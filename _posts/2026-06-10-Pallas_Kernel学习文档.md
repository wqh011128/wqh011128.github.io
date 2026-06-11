---
layout: post
title: "Pallas TPU Kernel 写法与优化文档"
date: 2026-06-10
categories:
  - blog
---

# Pallas TPU Kernel 写法与优化文档

## 0. 适用范围

本文总结 Pallas TPU kernel 的常用生产级写法、设计规范与优化思路，重点面向以下类型的 kernel：

- Attention / FlashAttention / GQA / MQA；
- Matmul-heavy backward kernel；
- 带 reduction 的 block kernel；
- 需要在 TPU MXU 上获得较高利用率的 Pallas kernel；
- 需要控制 HBM、VMEM、register、scratch、tile shape 的性能敏感 kernel。

本文尤其关注：

```text
1. 如何设计 grid
2. 如何设计 BlockSpec
3. 如何使用 scratch 做 partial accumulation
4. 如何选择 major/minor tile
5. 如何避免单个 program 内部 loop 太长
6. 如何提升 MXU utilization
7. 如何 benchmark 和调 block size
8. 如何写出更接近生产级的 Pallas kernel
```

---

# 1. Pallas Kernel 的基本心智模型

写 Pallas kernel 时，建议始终从下面几个概念出发。

## 1.1 Program

一个 Pallas `program` 可以理解成一个小的 tile 程序实例。

它由 `grid` 决定数量，例如：

```python
grid = (
    batch_blocks,
    num_heads,
    q_blocks,
    kv_blocks,
)
```

每个 program 会通过：

```python
pl.program_id(axis)
```

拿到当前自己负责的 tile index。

例如：

```python
q_block_index = pl.program_id(2)
kv_block_index = pl.program_id(3)
```

可以理解为：

```text
一个 program 负责一个局部 tile。
```

---

## 1.2 Grid

`grid` 决定有多少个 program 被调度。

一个好的 grid 应该：

```text
1. 暴露足够并行度；
2. 把重要的 block 维度显式暴露给 compiler；
3. 避免把过长的 reduction loop 藏在单个 program 里；
4. 让每个 program 的工作量适中；
5. 让输出 tile 和 reduction tile 的关系清楚。
```

对于 attention backward，常见 grid 设计是：

```text
DQ:
  grid = batch × head × q_block × kv_block

DKV:
  grid = batch × head_or_kv_head × kv_block × q_block
```

---

## 1.3 BlockSpec

`pl.BlockSpec` 决定每个 program 看到输入/输出 tensor 的哪一个 tile。

典型形式：

```python
q_spec = pl.BlockSpec(
    (block_b, group_size, block_q, head_dim),
    q_index_map,
)
```

其中：

```text
block_shape:
  当前 program 看到的局部 tile shape

index_map:
  从 program_id 映射到全局 tensor block index
```

生产级原则：

```text
BlockSpec 应该尽量描述局部 tile，而不是 full sequence。
```

例如，尽量避免：

```python
q_spec = pl.BlockSpec(
    (block_b, group_size, q_seq_len, head_dim),
    q_full_index_map,
)
```

因为这意味着一个 program 对 Q 的 sequence 维几乎没有 tile 化。

更推荐：

```python
q_spec = pl.BlockSpec(
    (block_b, group_size, block_q_major, head_dim),
    q_index_map,
)
```

---

## 1.4 MXU

TPU MXU 是主要做矩阵乘法的硬件单元。

Pallas kernel 想跑得快，通常要做到：

```text
1. dot_general / dot 的 shape 规整；
2. matmul tile 足够大；
3. matmul 之间不要被太多 elementwise / mask / exp 阻塞；
4. 数据能及时 load 到 VMEM；
5. 不要让 MXU 等待数据或等待复杂控制流；
6. 不要让单个 program 的 live range 太长。
```

如果 MXU utilization 很低，例如只有 13%，通常说明：

```text
MXU 大量时间没有被有效喂饱。
```

常见原因包括：

```text
1. grid 并行度不足；
2. 单个 program 内部 loop 太长；
3. temporary tensor 太大；
4. VMEM/register 压力过高；
5. BlockSpec 过大；
6. dot tile shape 不合适；
7. matmul 中间夹杂过多 exp/mask/elementwise；
8. causal 上三角无效 block 没有跳过。
```

---

# 2. 写 Kernel 前必须先做的设计

不要一上来就写 Pallas kernel。生产级 kernel 应先写清楚：

```text
1. 数学公式；
2. 输入输出 shape；
3. 哪些维度是并行维；
4. 哪些维度是 reduction 维；
5. 是否需要跨 block 累加；
6. 是否需要 scratch；
7. 是否需要保存 residuals；
8. 是否需要 mask/bias/causal/segment；
9. 是否需要支持 padding；
10. 是否需要支持 GQA/MQA。
```

---

## 2.1 先写数学公式

以 attention backward 为例：

前向：

```text
S = QK^T
P = softmax(S)
O = P V
```

反向：

```text
dV = P^T dO
dP = dO V^T
D  = sum(O * dO, axis=-1)
dS = P * (dP - D)
dQ = dS K
dK = dS^T Q
```

如果有 scale：

```text
S = QK^T * sm_scale
```

则：

```text
dS_unscaled = dS_scaled * sm_scale
```

如果有 additive attention bias：

```text
S = QK^T + AB
```

则：

```text
dAB = dS
```

---

## 2.2 明确输出决定 kernel 方向

对于 backward，常见拆法是：

```text
DQ kernel:
  输出 dQ
  沿 KV 维 reduction

DKV kernel:
  输出 dK, dV
  沿 Q 维 reduction
```

因此：

```text
DQ 的 grid 应该优先围绕 Q output tile 设计；
DKV 的 grid 应该优先围绕 KV output tile 设计。
```

---

# 3. Grid 设计规范

## 3.1 不要把长 reduction 全塞进单个 program

不推荐：

```python
grid = (
    batch_blocks,
    num_heads,
    q_blocks,
)

for kv_start in range(0, kv_seq_len, block_kv):
    ...
```

因为这意味着：

```text
一个 program 负责一个 q_block，
然后在 program 内部扫完整 kv_seq。
```

如果 `kv_seq_len` 很长，program 内部 loop 会很长。

更推荐：

```python
grid = (
    batch_blocks,
    num_heads,
    q_blocks,
    kv_blocks,
)
```

让 `kv_blocks` 成为 grid 维度。

---

## 3.2 DQ 推荐 grid

DQ 的公式：

```text
dQ = sum_over_KV dS @ K
```

推荐：

```python
grid_dq = (
    batch_blocks,
    num_kv_heads,
    q_seq_len // block_q_major_dq,
    kv_seq_len // block_kv_major_dq,
)
```

含义：

```text
axis 0: batch block
axis 1: kv head
axis 2: q major block
axis 3: kv major block
```

如果是 GQA：

```text
一个 kv_head 对应 group_size 个 q_heads。
```

所以 Q tile 可以是：

```text
[block_b, group_size, block_q_major_dq, head_dim]
```

---

## 3.3 DKV 推荐 grid

DKV 的公式：

```text
dK = sum_over_Q dS^T @ Q
dV = sum_over_Q P^T @ dO
```

推荐：

```python
grid_dkv = (
    batch_blocks,
    num_kv_heads,
    kv_seq_len // block_kv_major_dkv,
    q_seq_len // block_q_major_dkv,
)
```

含义：

```text
axis 0: batch block
axis 1: kv head
axis 2: kv major block
axis 3: q major block
```

这样每个 program 负责：

```text
一个 KV output tile
一个 Q reduction tile
```

再通过 scratch 跨 Q block 累加到最终 dK/dV。

---

## 3.4 哪些维度放 grid，哪些维度放 loop？

经验规则：

```text
短 loop 可以放 program 内部；
长 reduction loop 尽量暴露到 grid；
需要跨 block 累加时，用 scratch；
如果 scratch/ordering 很难处理，再考虑内部 loop。
```

推荐：

```text
batch/head/q_block/kv_block:
  通常适合放 grid

minor tile:
  通常适合放 program 内部 loop

完整 seq_len:
  不建议放单个 program 内部扫完
```

---

# 4. Scratch 设计规范

## 4.1 Scratch 的作用

Scratch 不是为了“直接加速”，而是为了允许：

```text
把 reduction 维拆到 grid
+
在 VMEM 中保存 partial accumulation
+
避免 atomic
+
最后一个 reduction block 写出结果
```

如果没有 scratch，通常只能让一个 program 扫完整 reduction 维。

这容易造成：

```text
1. 单个 program 太重；
2. 内部 loop 太长；
3. compiler 调度困难；
4. live range 变长；
5. MXU 利用率低。
```

---

## 4.2 DQ scratch 模式

DQ 需要沿 KV 维累加：

```text
dQ = sum_over_KV dS @ K
```

所以 DQ scratch 可以设计为：

```python
dq_scratch = pltpu.VMEM(
    (block_b, group_size, block_q_major_dq, head_dim),
    jnp.float32,
)
```

逻辑：

```python
kv_block_index = pl.program_id(3)

@pl.when(kv_block_index == 0)
def init():
    dq_scratch[...] = 0

@pl.when(should_run)
def run():
    dq_scratch[...] += partial_dq

@pl.when(kv_block_index == last_kv_block)
def store():
    dq_ref[...] = dq_scratch.astype(dq_ref.dtype)
```

---

## 4.3 DKV scratch 模式

DKV 需要沿 Q 维累加：

```text
dK = sum_over_Q dS^T @ Q
dV = sum_over_Q P^T @ dO
```

所以 DKV scratch 可以设计为：

```python
dk_scratch = pltpu.VMEM(
    (block_b, block_kv_major_dkv, head_dim),
    jnp.float32,
)

dv_scratch = pltpu.VMEM(
    (block_b, block_kv_major_dkv, head_dim),
    jnp.float32,
)
```

逻辑：

```python
q_block_index = pl.program_id(3)

@pl.when(q_block_index == 0)
def init():
    dk_scratch[...] = 0
    dv_scratch[...] = 0

@pl.when(should_run)
def run():
    dk_scratch[...] += partial_dk
    dv_scratch[...] += partial_dv

@pl.when(q_block_index == last_q_block)
def store():
    dk_ref[...] = dk_scratch.astype(dk_ref.dtype)
    dv_ref[...] = dv_scratch.astype(dv_ref.dtype)
```

---

## 4.4 使用 scratch 的注意事项

使用 scratch 跨 grid 维累加时，要注意：

```text
1. reduction grid 维必须有明确执行顺序；
2. 不能在完全 parallel 的 grid 维上假设先后顺序；
3. 初始化、累加、写出条件必须严格正确；
4. scratch shape 不要过大；
5. scratch 通常用 fp32 保证累加精度；
6. 最终写出时再 cast 到输出 dtype。
```

如果依赖某个 grid 维顺序推进，需要在 TPU compiler params 中正确设置该维的语义，例如将 reduction 维设置为 `"arbitrary"`，而不是所有维都当成完全 parallel。

---

# 5. BlockSpec 设计规范

## 5.1 BlockSpec 应该 tile 化

推荐：

```python
q_spec = pl.BlockSpec(
    (block_b, group_size, block_q_major, head_dim),
    q_index_map,
)
```

不推荐：

```python
q_spec = pl.BlockSpec(
    (block_b, group_size, q_seq_len, head_dim),
    q_full_index_map,
)
```

因为后者意味着：

```text
对 q_seq_len 这个维度几乎没有 tile 化。
```

这会让一个 program 看到完整 sequence，容易导致：

```text
1. input window 过大；
2. prefetch 粒度过粗；
3. compiler 难以优化；
4. live range 变长；
5. VMEM/register 压力变大；
6. MXU pipeline 不稳定。
```

---

## 5.2 DQ BlockSpec 示例

```python
def q_index_map(batch_index, kv_head_index, q_block_index, kv_block_index):
    return batch_index, kv_head_index, q_block_index, 0

def kv_index_map(batch_index, kv_head_index, q_block_index, kv_block_index):
    return batch_index, kv_head_index, kv_block_index, 0

q_spec = pl.BlockSpec(
    (block_b, group_size, block_q_major_dq, head_dim),
    q_index_map,
)

kv_spec = pl.BlockSpec(
    (block_b, 1, block_kv_major_dq, head_dim),
    kv_index_map,
)

lse_spec = pl.BlockSpec(
    (block_b, group_size, block_q_major_dq, MIN_BLOCK_SIZE),
    q_index_map,
)
```

---

## 5.3 DKV BlockSpec 示例

```python
def q_index_map(batch_index, kv_head_index, kv_block_index, q_block_index):
    return batch_index, kv_head_index, q_block_index, 0

def kv_index_map(batch_index, kv_head_index, kv_block_index, q_block_index):
    return batch_index, kv_head_index, kv_block_index, 0

q_spec = pl.BlockSpec(
    (block_b, group_size, block_q_major_dkv, head_dim),
    q_index_map,
)

kv_spec = pl.BlockSpec(
    (block_b, 1, block_kv_major_dkv, head_dim),
    kv_index_map,
)

lse_spec = pl.BlockSpec(
    (block_b, group_size, block_q_major_dkv, MIN_BLOCK_SIZE),
    q_index_map,
)
```

---

## 5.4 Skipped block 的 safe index

如果 causal 下某些 block 完全无效，Pallas 仍可能需要 index_map 返回合法 tile。

可以使用 safe index：

```python
def kv_index_map(batch_index, kv_head_index, q_block_index, kv_block_index):
    should_run = below_or_on_diag(
        q_block_index,
        block_q_major,
        kv_block_index,
        block_kv_major,
    )
    safe_kv_index = lax.select(should_run, kv_block_index, 0)
    return batch_index, kv_head_index, safe_kv_index, 0
```

这样可以避免无效或越界 prefetch。

---

# 6. Major / Minor Tile 分层

## 6.1 为什么要分 major/minor？

不要只用一层：

```text
block_q
block_kv
```

更推荐两层：

```text
major block:
  grid 级别 tile

minor block:
  program 内部 dot tile
```

好处：

```text
1. grid 粒度和 dot 粒度分开调；
2. 更容易平衡并行度和数据复用；
3. 更容易控制 VMEM/register 压力；
4. 更容易找到 MXU 友好的 matmul shape；
5. 更容易做 block size sweep；
6. 更接近成熟 FlashAttention kernel 的写法。
```

---

## 6.2 DQ tile 参数

推荐：

```python
block_q_major_dq
block_kv_major_dq
block_kv_dq
```

含义：

```text
block_q_major_dq:
  DQ kernel 一个 q grid block 覆盖的 Q 长度

block_kv_major_dq:
  DQ kernel 一个 kv grid block 覆盖的 KV 长度

block_kv_dq:
  DQ kernel 内部每次 dot 的 KV minor tile
```

示例：

```python
block_q_major_dq = 128
block_kv_major_dq = 256
block_kv_dq = 128
```

表示：

```text
一个 DQ program 覆盖 128 个 Q token 和 256 个 KV token，
但内部每次只用 128 个 KV token 做一次 dot。
```

伪代码：

```python
for kv_minor in range(0, block_kv_major_dq, block_kv_dq):
    k = k_tile[:, :, kv_minor : kv_minor + block_kv_dq, :]
    v = v_tile[:, :, kv_minor : kv_minor + block_kv_dq, :]
    scores = q @ k.T
    ...
    dq_scratch += ds @ k
```

---

## 6.3 DKV tile 参数

推荐：

```python
block_q_major_dkv
block_kv_major_dkv
block_q_dkv
block_kv_dkv
```

含义：

```text
block_q_major_dkv:
  DKV kernel 一个 q grid block 覆盖的 Q 长度

block_kv_major_dkv:
  DKV kernel 一个 kv grid block 覆盖的 KV 长度

block_q_dkv:
  DKV kernel 内部每次处理的 Q minor tile

block_kv_dkv:
  DKV kernel 内部每次处理的 KV minor tile
```

示例：

```python
block_q_major_dkv = 128
block_kv_major_dkv = 256
block_q_dkv = 128
block_kv_dkv = 128
```

表示：

```text
一个 DKV program 覆盖 128 个 Q token 和 256 个 KV token，
但内部把 KV 再拆成两个 128 的 minor tile。
```

伪代码：

```python
for q_minor in range(0, block_q_major_dkv, block_q_dkv):
    for kv_minor in range(0, block_kv_major_dkv, block_kv_dkv):
        q = q_tile[:, :, q_minor : q_minor + block_q_dkv, :]
        k = k_tile[:, :, kv_minor : kv_minor + block_kv_dkv, :]
        v = v_tile[:, :, kv_minor : kv_minor + block_kv_dkv, :]

        scores = q @ k.T
        ...
        dk_scratch[kv_minor] += ds.T @ q
        dv_scratch[kv_minor] += p.T @ do
```

---

# 7. Causal Mask 优化规范

## 7.1 区分 block-level mask 和 element-level mask

Element-level causal mask：

```python
scores = scores + jnp.where(mask, 0.0, DEFAULT_MASK_VALUE)
```

这是必要的，因为对角线附近的 partial block 仍需要逐元素 mask。

但对于完全在 causal 上三角的 block，应该 block-level skip：

```text
如果 q_block_end < kv_block_start，
则整个 block 无效，不需要做 QK^T。
```

---

## 7.2 Block-level skip

推荐函数：

```python
def below_or_on_diag(q_block_index, block_q, kv_block_index, block_kv):
    q_block_end = (q_block_index + 1) * block_q - 1
    kv_block_start = kv_block_index * block_kv
    return q_block_end >= kv_block_start
```

DQ 中：

```python
should_run = below_or_on_diag(
    q_block_index,
    block_q_major_dq,
    kv_block_index,
    block_kv_major_dq,
)
```

DKV 中：

```python
should_run = below_or_on_diag(
    q_block_index,
    block_q_major_dkv,
    kv_block_index,
    block_kv_major_dkv,
)
```

然后：

```python
@pl.when(should_run)
def run():
    ...
```

---

## 7.3 为什么 block-level skip 很重要？

对于 causal prefill，理论上上三角大约一半 attention block 无效。

如果不 skip，而是：

```python
scores = q @ k.T
scores = apply_causal_mask(scores)
```

则无效 block 仍然消耗：

```text
1. QK^T matmul
2. exp
3. dp
4. ds
5. dQ/dK/dV partial accumulation
```

所以 block-level skip 可以直接降低 latency。

---

# 8. Attention Backward 常用生产级模板

## 8.1 DQ kernel 模板

数学：

```text
dQ = sum_over_KV dS @ K
```

结构：

```python
def dq_kernel(
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    lse_ref,
    do_ref,
    dq_ref,
    dq_scratch_ref,
    *,
    sm_scale,
    block_kv_minor,
    kv_seq_len,
):
    q_block_index = pl.program_id(2)
    kv_block_index = pl.program_id(3)

    @pl.when(kv_block_index == 0)
    def init():
        dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)

    should_run = below_or_on_diag(
        q_block_index,
        block_q_major_dq,
        kv_block_index,
        block_kv_major_dq,
    )

    @pl.when(should_run)
    def run():
        q = q_ref[...]
        o = o_ref[...]
        do = do_ref[...]
        lse = lse_ref[...]

        di = jnp.sum(
            o.astype(jnp.float32) * do.astype(jnp.float32),
            axis=-1,
        )

        for kv_minor in range(0, block_kv_major_dq, block_kv_minor):
            k = k_ref[..., kv_minor : kv_minor + block_kv_minor, :]
            v = v_ref[..., kv_minor : kv_minor + block_kv_minor, :]

            scores = dot(q, k.T)
            scores *= sm_scale
            scores = apply_causal_mask_if_needed(scores)

            p = exp(scores - lse)
            dp = dot(do, v.T)
            ds = (dp - di[:, None]) * p
            ds *= sm_scale

            dq_scratch_ref[...] += dot(ds, k)

    @pl.when(kv_block_index == kv_seq_len // block_kv_major_dq - 1)
    def store():
        dq_ref[...] = dq_scratch_ref.astype(dq_ref.dtype)
```

---

## 8.2 DKV kernel 模板

数学：

```text
dK = sum_over_Q dS^T @ Q
dV = sum_over_Q P^T @ dO
```

结构：

```python
def dkv_kernel(
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    lse_ref,
    do_ref,
    dk_ref,
    dv_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    *,
    sm_scale,
    block_q_minor,
    block_kv_minor,
    q_seq_len,
):
    kv_block_index = pl.program_id(2)
    q_block_index = pl.program_id(3)

    @pl.when(q_block_index == 0)
    def init():
        dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
        dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)

    should_run = below_or_on_diag(
        q_block_index,
        block_q_major_dkv,
        kv_block_index,
        block_kv_major_dkv,
    )

    @pl.when(should_run)
    def run():
        for q_minor in range(0, block_q_major_dkv, block_q_minor):
            q = q_ref[..., q_minor : q_minor + block_q_minor, :]
            o = o_ref[..., q_minor : q_minor + block_q_minor, :]
            do = do_ref[..., q_minor : q_minor + block_q_minor, :]
            lse = lse_ref[..., q_minor : q_minor + block_q_minor, :]

            di = jnp.sum(
                o.astype(jnp.float32) * do.astype(jnp.float32),
                axis=-1,
            )

            for kv_minor in range(0, block_kv_major_dkv, block_kv_minor):
                k = k_ref[..., kv_minor : kv_minor + block_kv_minor, :]
                v = v_ref[..., kv_minor : kv_minor + block_kv_minor, :]

                scores = dot(q, k.T)
                scores *= sm_scale
                scores = apply_causal_mask_if_needed(scores)

                p = exp(scores - lse)
                dp = dot(do, v.T)
                ds = (dp - di[:, None]) * p
                ds *= sm_scale

                dk_scratch_ref[..., kv_minor, :] += dot(ds.T, q)
                dv_scratch_ref[..., kv_minor, :] += dot(p.T, do)

    @pl.when(q_block_index == q_seq_len // block_q_major_dkv - 1)
    def store():
        dk_ref[...] = dk_scratch_ref.astype(dk_ref.dtype)
        dv_ref[...] = dv_scratch_ref.astype(dv_ref.dtype)
```

---

# 9. LSE / m / l 设计规范

## 9.1 LSE 是什么？

LSE 是：

```text
logsumexp(scores)
```

也就是：

```text
lse_i = log(sum_j exp(scores_i,j))
```

backward 中可以用它重建 softmax：

```python
p = jnp.exp(scores - lse[:, None])
```

---

## 9.2 m/l 和 LSE 的关系

FlashAttention forward 有时保存：

```text
m = row max
l = sum(exp(scores - m))
```

那么：

```text
lse = m + log(l)
```

两种 backward 重建 P 的方式等价：

```python
p = exp(scores - lse)
```

等价于：

```python
p = exp(scores - m) / l
```

---

## 9.3 生产级建议

```text
1. 如果已有 LSE，backward 用 exp(scores - lse) 最简单；
2. 如果 forward 是 streaming softmax，可保存 m/l；
3. residuals 应尽量比完整 P 小；
4. 不要保存完整 [B, H, Q, K] attention matrix；
5. LSE/m/l 通常用 fp32。
```

---

# 10. Dtype 与数值稳定性规范

## 10.1 Matmul accumulation

建议：

```python
preferred_element_type=jnp.float32
```

尤其是：

```text
QK^T
P @ V
dO @ V^T
dS @ K
dS^T @ Q
P^T @ dO
```

---

## 10.2 Softmax 相关

建议：

```text
scores / p / ds / accumulators 尽量使用 fp32
最终输出再 cast 到 q/k/v dtype
```

常见写法：

```python
scores = dot(..., preferred_element_type=jnp.float32)
p = jnp.exp(scores - lse)
ds = (dp - di[:, None]) * p
dq_acc = dq_acc.astype(jnp.float32)
```

---

## 10.3 mask value

常见写法：

```python
DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)
```

避免直接使用 `-inf` 导致某些平台上的数值或编译问题。

---

# 11. GQA / MQA Kernel 设计规范

## 11.1 GQA 的核心

GQA 中：

```text
num_q_heads > num_kv_heads
group_size = num_q_heads // num_kv_heads
```

每个 KV head 被多个 Q heads 共享。

因此：

```text
dK/dV 需要聚合来自同一 group 内多个 Q heads 的梯度。
```

---

## 11.2 推荐 layout

对于一个 KV head：

```text
Q tile:
  [group_size, block_q, head_dim]

K/V tile:
  [1, block_kv, head_dim]
```

可以 reshape：

```text
[group_size, block_q, head_dim]
->
[group_size * block_q, head_dim]
```

这样一次 matmul 覆盖整个 Q-head group。

---

## 11.3 group_size 的风险

如果：

```text
group_size * block_q
```

太大，则：

```text
scores / p / dp / ds
```

会非常大。

例如：

```text
group_size = 8
block_q = 128
block_kv = 256
scores = [1024, 256]
```

这可能导致 VMEM/register 压力过大。

生产级策略：

```text
第一阶段:
  block_g = full group_size
  简化 dK/dV reduction

第二阶段:
  如果 group_size 很大且 VMEM 压力高，再引入 block_g
```

---

## 11.4 block_g 的代价

如果引入：

```python
block_g < group_size
```

则 DKV 需要额外处理：

```text
不同 group block 对同一个 dK/dV 的累加。
```

这可能需要：

```text
1. 更复杂的 scratch；
2. 额外 reduction kernel；
3. atomic-like accumulation 设计；
4. 更复杂的 correctness 测试。
```

所以不建议第一版就引入 `block_g`。

---

# 12. 性能优化 Checklist

## 12.1 MXU 利用率低时优先检查

```text
1. dot_general shape 是否太小？
2. dot_general shape 是否不规整？
3. block size 是否 128 对齐？
4. 是否有过长 program 内部 loop？
5. reduction 维是否应该放进 grid？
6. 是否用了 full-seq BlockSpec？
7. scores/p/dp/ds 是否太大？
8. VMEM/register 是否压力太高？
9. 是否有 spill 或 compiler warning？
10. causal 上三角是否被 matmul 计算了？
11. matmul 中间是否夹杂太多 elementwise？
12. grid program 数是否太少？
13. batch/head 并行度是否足够？
14. block_b 是否过大或过小？
```

---

## 12.2 Latency 高时优先检查

```text
1. 是否没有 block-level causal skip？
2. 是否物化了 [B, H, Q, K] 大 tensor？
3. 是否写出了 dAB/dS？
4. 是否 HBM 读写过多？
5. 是否重复读取 full Q/K/V？
6. scratch 是否太大？
7. block_kv 是否太大导致临时矩阵膨胀？
8. block_q 是否太大导致 group_size * block_q 过大？
9. 是否 padding 后多算太多 token？
10. DQ 和 DKV 哪一个更慢？
```

---

## 12.3 Correctness 失败时优先检查

```text
1. causal mask 的 row/col offset 是否正确？
2. q_start / kv_start 是否正确？
3. GQA head mapping 是否正确？
4. lse/m/l shape 是否正确 broadcast？
5. sm_scale 是否 forward/backward 一致？
6. ds 是否乘了 sm_scale？
7. di = sum(o * do) 是否按 row 计算？
8. dtype cast 是否过早？
9. padding 后是否正确 slice 回原 shape？
10. skipped block 是否误跳过了 diagonal block？
11. scratch init/store 条件是否正确？
```

---

# 13. Benchmark 规范

## 13.1 latency 是主指标

当比较 Pallas kernel 和 JAX reference 时，不要直接混用 FLOP 口径。

常见 FLOP 口径包括：

```text
1. JAX compiler estimated FLOPs
2. manual Pallas matmul inventory
3. theoretical math FLOPs
4. actual executed FLOPs with causal skip
```

这些不是一回事。

速度比较以：

```text
latency
```

为主。

---

## 13.2 effective TFLOP/s

如果要计算：

```text
effective_TFLOP/s
```

必须固定 FLOP 口径。

建议使用：

```text
manual Pallas matmul inventory
```

公式：

```text
effective_TFLOP/s = manual_pallas_GFLOP / latency_ms
```

因为：

```text
1 GFLOP / 1 ms = 1 TFLOP/s
```

---

## 13.3 必须分别测 DQ 和 DKV

不要只看 total。

建议记录：

```text
DQ latency
DKV latency
Total latency
DQ MXU utilization
DKV MXU utilization
Total MXU utilization
```

原因：

```text
DQ 和 DKV 的瓶颈可能完全不同。
```

例如：

```text
DQ 可能卡在 KV reduction；
DKV 可能卡在 full Q BlockSpec、scratch、group reduction。
```

---

## 13.4 Benchmark 矩阵

建议至少测：

```text
batch:
  1, 2, 4

seq_len:
  512, 1024, 2048, 4096, 8192

head_dim:
  64, 128

num_q_heads / num_kv_heads:
  16/4, 32/8, 32/4

group_size:
  2, 4, 8

dtype:
  bf16 input + fp32 accumulate
```

---

# 14. Block Size Sweep 规范

## 14.1 Forward 和 backward 的最优 block size 可能不同

不要假设：

```text
forward 最优 block size = backward 最优 block size
```

原因：

```text
forward:
  QK^T + P@V + streaming softmax

backward:
  重建 P
  dV = P^T dO
  dP = dO V^T
  dS
  dK = dS^T Q
  dQ = dS K
```

backward 的 matmul 数量、reduction 方向、scratch 需求都不同。

---

## 14.2 DQ 和 DKV 的最优 block size 也可能不同

DQ：

```text
输出 dQ
沿 KV reduction
```

DKV：

```text
输出 dK/dV
沿 Q reduction
还要聚合 group_size 个 Q heads
```

所以：

```text
block_q_major_dq
block_kv_major_dq
block_kv_dq
```

和：

```text
block_q_major_dkv
block_kv_major_dkv
block_q_dkv
block_kv_dkv
```

应该分开调。

---

## 14.3 推荐初始 sweep

DQ：

```text
block_q_major_dq:
  64, 128

block_kv_major_dq:
  128, 256

block_kv_dq:
  128
```

DKV：

```text
block_q_major_dkv:
  64, 128, 256

block_kv_major_dkv:
  128, 256

block_q_dkv:
  64, 128

block_kv_dkv:
  128
```

如果 group_size 较大：

```text
block_q_major:
  优先尝试 64, 128

block_kv_major:
  优先尝试 128
```

---

# 15. 生产级代码结构建议

## 15.1 参数 dataclass

推荐：

```python
@dataclasses.dataclass(frozen=True)
class BlockSizes:
    block_b: int = 1

    block_q_major_dq: int = 128
    block_kv_major_dq: int = 128
    block_kv_dq: int = 128

    block_q_major_dkv: int = 128
    block_kv_major_dkv: int = 128
    block_q_dkv: int = 128
    block_kv_dkv: int = 128

    def __post_init__(self):
        ...
```

校验：

```text
1. 所有 block size > 0；
2. minor <= major；
3. major % minor == 0；
4. block_kv_* 是 128 的倍数；
5. q_seq_len / kv_seq_len 是否可整除，或是否需要 padding。
```

---

## 15.2 Shape validation

在外层 API 做 shape 校验：

```python
def validate_shapes(q, k, v, o, lse, do):
    assert q.shape == o.shape
    assert q.shape == do.shape
    assert k.shape == v.shape
    assert q.shape[1] % k.shape[1] == 0
    assert q.shape[-1] == k.shape[-1]
```

不要把 shape 错误留到 kernel 内部才失败。

---

## 15.3 Padding 和 slicing

生产级 kernel 通常需要支持非整除 sequence length。

做法：

```python
q_pad = pad_axis_to_multiple(q, axis=2, multiple=block_q)
k_pad = pad_axis_to_multiple(k, axis=2, multiple=block_kv)
...
out = out[:, :, :original_seq_len, :]
```

注意：

```text
padding token 必须被 mask 或保证不影响结果。
```

---

## 15.4 named_scope

推荐：

```python
name_scope = (
    f"gqa_bwd_dq_"
    f"{block_q_major_dq=}_"
    f"{block_kv_major_dq=}_"
    f"{block_kv_dq=}"
)

with jax.named_scope(name_scope):
    ...
```

好处：

```text
1. profile 更容易看；
2. HLO / trace 更容易定位；
3. benchmark 结果更清楚。
```

---

## 15.5 debug / interpret

建议外层 API 支持：

```python
debug: bool = False
interpret: bool = False
```

开发阶段：

```text
interpret=True 方便调试
```

性能测试阶段：

```text
interpret=False
debug=False
```

---

# 16. 常见反模式

## 16.1 一个 program 扫完整 seq_len

不推荐：

```python
for q_start in range(0, q_seq_len, block_q):
    ...
```

如果这个 loop 很长，应考虑把 `q_block` 放入 grid。

---

## 16.2 full-sequence BlockSpec

不推荐：

```python
pl.BlockSpec(
    (block_b, group_size, q_seq_len, head_dim),
    index_map,
)
```

推荐：

```python
pl.BlockSpec(
    (block_b, group_size, block_q_major, head_dim),
    index_map,
)
```

---

## 16.3 先 matmul 再 mask 掉整个无效 block

不推荐：

```python
scores = q @ k.T
scores = causal_mask(scores)
```

对于完全无效 block，应先 skip：

```python
if block_is_valid:
    scores = q @ k.T
```

---

## 16.4 写出 dS / dAB 大矩阵

如果不是必须，不要写：

```text
dAB = dS = [B, H, Q, K]
```

这会破坏 FlashAttention 不物化 attention matrix 的优势。

---

## 16.5 盲目增大 block_kv

`block_kv=256` 不一定比 `128` 快。

需要看：

```text
1. MXU utilization
2. VMEM pressure
3. scores/p/dp/ds size
4. causal skip 粒度
5. latency
```

---

# 17. 推荐优化顺序

如果一个 Pallas attention backward kernel MXU utilization 很低，建议按下面顺序优化。

## Step 1: 分离 DQ / DKV profiling

先确认：

```text
DQ 慢，还是 DKV 慢？
```

记录：

```text
DQ latency
DKV latency
DQ MXU utilization
DKV MXU utilization
```

---

## Step 2: 加 block-level causal skip

这是低风险高收益优化。

目标：

```text
完全 causal invalid 的 block 不做 QK^T。
```

---

## Step 3: 消除 full-seq BlockSpec

将：

```python
(block_b, group_size, q_seq_len, head_dim)
```

改成：

```python
(block_b, group_size, block_q_major, head_dim)
```

---

## Step 4: reduction 维放进 grid

DQ：

```text
把 kv_block 放进 grid。
```

DKV：

```text
把 q_block 放进 grid。
```

---

## Step 5: 用 scratch 做 partial accumulation

DQ：

```text
dq_scratch 跨 kv_block 累加。
```

DKV：

```text
dk_scratch / dv_scratch 跨 q_block 累加。
```

---

## Step 6: 引入 major/minor tile

把：

```text
block_q
block_kv
```

拆成：

```text
block_q_major
block_kv_major
block_q_minor
block_kv_minor
```

---

## Step 7: 系统 sweep block size

不要凭感觉判断最优 block。

至少 sweep：

```text
block_q_major = 64 / 128 / 256
block_kv_major = 128 / 256
block_q_minor = 64 / 128
block_kv_minor = 128
```

---

## Step 8: 视情况引入 block_g

只有当：

```text
group_size 很大
scores tile 太大
VMEM 压力明显
```

才考虑：

```python
block_g < group_size
```

---

# 18. 代码评审 Checklist

提交生产级 Pallas kernel 前，建议检查：

```text
[ ] 数学公式是否写清楚？
[ ] 输入输出 shape 是否校验？
[ ] GQA head mapping 是否正确？
[ ] grid 是否暴露足够并行度？
[ ] 是否避免 full-seq BlockSpec？
[ ] 是否避免单 program 内部长 reduction loop？
[ ] 是否正确使用 scratch？
[ ] scratch init/store 条件是否正确？
[ ] reduction grid 维是否有正确 dimension semantics？
[ ] causal block-level skip 是否正确？
[ ] diagonal block 是否仍然做 element-level mask？
[ ] dtype cast 是否合理？
[ ] accumulation 是否使用 fp32？
[ ] 是否避免写出不必要的大 tensor？
[ ] padding 后是否 slice 回原 shape？
[ ] 是否有 reference correctness test？
[ ] 是否有 DQ/DKV 单独 benchmark？
[ ] 是否记录 latency / MXU utilization / effective TFLOP/s？
[ ] 是否明确 FLOP 统计口径？
[ ] 是否 sweep 过核心 block size？
[ ] 是否有 named_scope 方便 profile？
```

---

# 19. 一句话总结

生产级 Pallas kernel 的核心不是“把公式翻译成 kernel”，而是：

```text
把数学计算拆成硬件友好的 tile，
把长 reduction 暴露成 grid，
用 scratch 做局部累加，
避免 full-sequence input window，
尽量让 MXU 连续吃到规整 matmul，
同时控制 VMEM/register/HBM 压力。
```

对于 FlashAttention / GQA backward，最常用、最值得掌握的模式是：

```text
DQ:
  grid over q_block × kv_block
  dq_scratch 跨 kv_block 累加

DKV:
  grid over kv_block × q_block
  dk_scratch / dv_scratch 跨 q_block 累加

Causal:
  block-level skip + diagonal block element-level mask

Tiling:
  major block 控制 grid 粒度
  minor block 控制 program 内部 dot 粒度

Benchmark:
  latency 为主
  FLOP 口径固定
  DQ/DKV 分开看
```