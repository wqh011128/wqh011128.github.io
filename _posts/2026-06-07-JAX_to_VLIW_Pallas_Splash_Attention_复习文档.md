---
layout: post
title: "JAX 到 VLIW，以及 Pallas / Splash Attention 复习文档"
date: 2026-06-04
categories:
  - blog
---

# JAX 到 VLIW，以及 Pallas / Splash Attention 复习笔记

本文整理两篇 Patrick Toulme 的文章：

- [From JAX to VLIW: Tracing a Computation Through the TPU Compiler](https://patricktoulme.substack.com/p/from-jax-to-vliw-tracing-a-computation)
- [When XLA Isn't Enough: From Pallas to VLIW](https://patricktoulme.substack.com/p/when-xla-isnt-enough-from-pallas)

目标不是逐字翻译，而是按模型框架开发者的视角复习：重点理解 JAX/HLO/XLA/Pallas 这些上层如何影响底层执行；LLO/VLIW 只学会基础读法，不陷入指令细节。

---

## 0. 两条总线路：JAX to VLIW 与 Pallas to VLIW

先把两篇文章压缩成两张图。

### 0.1 普通 JAX / XLA 路线

```text
Python JAX function
  -> JAX tracing / Jaxpr
  -> StableHLO / HLO
  -> HLO optimization passes
       algebraic simplification
       layout assignment
       tiling
       fusion
       memory assignment
       copy scheduling
  -> TPU backend LLO
  -> VLIW bundles
  -> TPU hardware
       HBM / VMEM / MXU / VPU / XLU / DMA
```

直觉：

```text
JAX 代码描述“我要算什么”
HLO 描述“张量图是什么”
XLA 优化“怎样重排、融合、分配内存”
LLO 描述“TPU 上具体用哪些硬件动作”
VLIW bundle 描述“同一个时刻哪些硬件指令一起发射”
```

### 0.2 Pallas / Mosaic 路线

```text
Python Pallas kernel
  -> pallas_call
  -> HLO custom-call
       XLA 只看到一个 opaque call
  -> Mosaic MLIR
       Mosaic 能看到 Pallas kernel body
  -> TPU backend LLO
  -> VLIW bundles
  -> TPU hardware
```

直觉：

```text
普通 JAX:
  你写完整 tensor program，让 XLA 自动找 fusion / tiling。

Pallas:
  你直接写 tiled kernel program，让 Mosaic 和 TPU backend 继续降到底层。
```

### 0.3 两篇文章合起来的核心

第一篇的主题：

```text
XLA 很强。
普通 JAX 程序可以被自动优化成 TPU 上的高质量 VLIW 程序。
```

第二篇的主题：

```text
XLA 有边界。
XLA 会优化你写出来的算法，但通常不会自动发明另一个算法。
Splash / FlashAttention 这类 online softmax 是算法级改写，因此需要 Pallas 表达。
```

最重要的对比：

```text
XLA fusion:
  减少已经存在的中间 tensor 的读写。

Splash / FlashAttention:
  改写算法，让完整 attention matrix 根本不成为程序语义里的 tensor。
```

---

# 第一篇：From JAX to VLIW

## 1. 例子在算什么

文章使用一个很小的 attention-like block，便于观察整个 TPU 编译链路：

```python
h = x @ w1
h = h / sqrt(mean(h ** 2) + eps)
h = softmax(h)
out = h @ w2
```

形状大致是：

```text
x:   [16, 64]
w1:  [64, 64]
w2:  [64, 32]
out: [16, 32]
```

它包含：

```text
matmul_1
RMSNorm-like normalization
softmax
matmul_2
```

作者使用 `jax.jit` 触发编译，用 dump flags 输出 HLO 和 TPU backend 的 LLO。`jax.named_call` 的作用是让 IR metadata 中保留人类可读名字，例如 `matmul_1`、`rms_norm`、`softmax`。

对框架开发者来说，这个例子的价值在于：它虽然小，但覆盖了模型里常见的关键模式：

```text
matmul producer
elementwise chain
reduce
broadcast
multi-output use
softmax
second matmul
```

---

## 2. HLO 是什么

HLO 是 High Level Optimizer IR。可以把它理解成：

```text
带 shape / dtype / layout / metadata 的 SSA 张量计算图
```

SSA 表示每个中间值只定义一次：

```text
%dot = dot(%x, %w1)
%square = multiply(%dot, %dot)
%sum = reduce(%square)
%sqrt = sqrt(...)
```

HLO 仍然是比较高层的 IR。它关心：

```text
op 类型：dot / reduce / broadcast / exp / sqrt
tensor shape：f32[16,64]
layout：{1,0}
reduce 维度
metadata：来自哪一行 Python，来自哪个 named_call
```

它暂时不关心：

```text
哪个 MXU 执行 matmul
哪个 cycle 发射指令
何时 vmatpush / vpop
哪个 VLIW bundle 同时发射 DMA 和 vector op
```

因此，HLO 是模型框架开发者很值得读的层级。很多性能问题在 HLO 层已经能判断：

```text
是否 materialize 了巨大中间矩阵
是否有不必要 broadcast
是否存在 producer 被多个 consumer 使用
是否可以 fusion
layout 是否合理
copy-start / copy-done 是否能 overlap
```

---

## 3. 初始 HLO：从 Python 语义到显式张量图

### 3.1 Matmul

Python：

```python
h = x @ w1
```

HLO 中一般表现为：

```text
dot(x, w1)
```

HLO 会带上 contracting dimensions，例如：

```text
lhs_contracting_dims={1}
rhs_contracting_dims={0}
```

也就是：

```text
x:  [16,64]
w1: [64,64]
沿 x 的第 1 维和 w1 的第 0 维相乘求和
输出 [16,64]
```

### 3.2 RMSNorm-like 部分

Python 直觉：

```python
square = h ** 2
mean = sum(square, axis=-1, keepdims=True) / 64
rms = sqrt(mean + eps)
h_norm = h / rms
```

HLO 里会显式拆成：

```text
multiply(h, h)
reduce_sum(axis=1)
multiply by 1/64 或 divide by 64
add eps
sqrt
broadcast
divide
```

注意 `broadcast`：Python 写 `keepdims=True` 或隐式 broadcasting 时看起来很自然，但 HLO 必须显式说明哪个 shape 扩展到哪个 shape。

### 3.3 Softmax

Python：

```python
m = max(h, axis=-1, keepdims=True)
e = exp(h - m)
p = e / sum(e, axis=-1, keepdims=True)
```

HLO：

```text
reduce_max
broadcast max
subtract
exp
reduce_sum
broadcast sum
divide
```

HLO 中 `reduce` 通常会带一个小 computation region，表示规约函数是 `add` 还是 `maximum`。

---

## 4. HLO optimization passes 做了什么

文章展示了几类重要优化。

### 4.1 Algebraic simplification

例如：

```text
x / 64
```

会变成：

```text
x * 0.015625
```

因为乘法通常比除法更便宜。这类优化在 HLO 层完成，不是 TPU 特有。

### 4.2 Layout assignment

HLO shape 可能出现：

```text
f32[16,64]{1,0}
```

解释：

```text
f32[16,64]  dtype + shape
{1,0}       physical layout order
```

`{1,0}` 可以粗略理解为最后一维更连续，接近 row-major 的直觉。

### 4.3 TPU tiling

优化后可能出现：

```text
f32[16,64]{1,0:T(8,128)}
```

`T(8,128)` 是 TPU tile annotation。它和 TPU VPU 的向量组织有关：常见向量寄存器结构可以理解为 8 个 sublanes，每个 sublane 128 lanes。

这层含义是：

```text
HLO tensor 不是只作为抽象 ndarray 存放，
backend 已经开始决定怎样把它切成更贴近 TPU 硬件的 tile。
```

### 4.4 Fusion

Fusion 是第一篇的主角之一。

没有 fusion：

```text
op A 产生中间 tensor
写出去
op B 读回来
产生另一个中间 tensor
写出去
op C 再读回来
```

有 fusion：

```text
op A / B / C 进入同一个 fusion kernel
中间值尽量留在寄存器、VMEM 或临时片上状态
只写 fusion 的对外输出
```

文章里的最终 HLO 大致被压成几个 fusion：

```text
multiply_reduce_fusion:
  matmul_1 + square + reduce_sum
  -> (sum_of_squares, matmul_result)

add_sqrt_fusion:
  sum_of_squares * 1/64 + eps -> sqrt

fusion.5:
  normalized h -> reduce_max

fusion.2:
  exp(h - max) -> reduce_sum

fusion:
  normalize softmax + matmul_2 -> output
```

---

## 5. Multi-output fusion：为什么重要

第一次 matmul 的结果：

```python
h = x @ w1
```

有两个用途：

```text
用途 1：h -> h ** 2 -> reduce_sum，用于 RMSNorm
用途 2：h -> 后续 normalize / softmax
```

如果处理不好，编译器可能遇到两难：

```text
要么重复计算 matmul
要么把 matmul result 写到大 buffer，后续再读
```

multi-output fusion 的做法是：

```text
multiply_reduce_fusion -> (reduce_sum(h*h), h)
```

它让 matmul 只算一次，同时产出：

```text
1. 每行平方和
2. matmul result 本身
```

对模型框架开发者来说，这对应一个常见图优化问题：

```text
一个 producer 有多个 consumer。
```

好的优化器需要在以下选择间做代价判断：

```text
duplicate producer
materialize producer
multi-output fusion
recompute
```

---

## 6. Memory assignment：HBM、VMEM、copy-start、copy-done

文章中 HLO 会出现类似 memory space 的 annotation。粗略理解：

```text
S(0): HBM，片外大内存，通常是默认
S(1): VMEM，片上 SRAM
S(2): sync token 或 backend-specific space
```

HBM 和 VMEM 的关系：

```text
HBM:
  大，慢，跨 kernel 稳定可见

VMEM:
  小，快，片上，适合 tile 和短生命周期中间值
```

HLO 中还会出现：

```text
copy-start(w1)
copy-done(w1)
```

含义：

```text
copy-start:
  发起从 HBM 到 VMEM 的异步拷贝

copy-done:
  等待这个拷贝完成，之后才能使用
```

这允许 backend 做 overlap：

```text
一边计算当前阶段
一边 DMA 搬下一阶段需要的数据
```

例如：

```text
先 copy-start(w1)
copy-done(w1) 后调用第一个 matmul fusion

尽早 copy-start(w2)
在 RMSNorm / softmax 期间后台搬 w2
最后 matmul 前 copy-done(w2)
```

这就是编译器自动做的 prefetch / overlap。

---

## 7. Fusion 结束是否一定写回 HBM

这是非常容易误解的点。

更准确的说法：

```text
每个 HLO op 结束不一定写 HBM。
fusion 内部中间值通常不写 HBM。
fusion 的对外输出必须 materialize 到某个 buffer。
这个 buffer 可能是 HBM，也可能是 VMEM，取决于 memory assignment、大小、生命周期和后端能力。
```

因此：

```text
不是“每个 op 写 HBM”
也不是“每个 fusion 一定写 HBM”
而是“fusion/custom-call 边界通常是 materialization 边界”
```

在第一篇 toy program 里，一些中间 buffer 可以放到 VMEM。TLP 可以把 VMEM buffer 地址传给后续 fusion，所以并不是所有 fusion output 都落 HBM。

但在第二篇 naive attention 中，`scores` 形状是：

```text
[heads, seq_len, seq_len]
```

具体例子：

```text
[8, 2048, 2048] f32
= 8 * 2048 * 2048 * 4 bytes
= 128MB
```

这个中间矩阵太大，而且跨多个 fusion 被使用：

```text
fusion.5 产生 scores
fusion.2 需要 scores 来做 exp 和 reduce_sum
final fusion 需要 softmax 后的信息继续乘 V
```

所以它通常会 materialize 到 HBM。第二篇的性能问题本质就是这里。

判断规则：

```text
小的、短生命周期的、编译器能 schedule 在片上的值：
  可能留在 VMEM / register / scratch。

大的、跨 fusion 使用的、生命周期较长的 tensor：
  往往落 HBM。

Flash/Splash Attention 的胜利：
  不是让 128MB scores 写得更快，
  而是让完整 scores tensor 不存在。
```

---

## 8. 从 HLO 到 LLO：层级突然变低

HLO 还像数学图：

```text
dot
reduce
sqrt
exp
fusion
```

LLO 开始像硬件动作：

```text
vld
vst
vmatpush
vmatmul
vpop
vxpose
vrot.slane
dma.hbm_to_vmem
```

LLO 是 TPU-specific Low Level Operators。你不需要把每条指令都背下来，但要能识别几个模式。

### 8.1 TPU 硬件单元粗略图

| 单元 | 作用 |
|---|---|
| MXU | Matrix Multiply Unit，负责矩阵乘，TPU 主要算力来源 |
| VPU | Vector Processing Unit，负责 elementwise、vector add/mul/select 等 |
| XLU | 负责 transpose、shuffle、cross-lane movement |
| Scalar Unit | 标量计算、地址、控制流 |
| DMA | HBM 与 VMEM 之间异步搬运 |

### 8.2 LLO 中识别 matmul

常见模式：

```text
vld        从 VMEM load tile
vmatpush   把 tile push 到 MXU
vmatmul    触发矩阵乘
vpop       从 MXU 取回 accumulator / result
```

如果看到：

```text
vmatpush
vmatmul
vpop.mrf / vpop.f32
```

基本可以判断这里在做 MXU matmul。

### 8.3 LLO 中识别 vector compute

常见：

```text
vadd
vmul
vsub
vsel
vcmp
vpow2
vrsqrt
```

这些通常是 VPU 上的向量操作。比如 RMSNorm 的 `sqrt`，后端可能用 `vrsqrt` 先算 reciprocal sqrt，再组合得到需要的结果。

### 8.4 LLO 中识别 reduction / transpose

RMSNorm 要做：

```text
sum(h ** 2, axis=-1)
```

TPU 的 lane/sublane 数据布局和 Python 行列不完全一致，所以 reduction 之前常会出现：

```text
vxpose
vpop.trf
```

然后是 tree reduction：

```text
vadd
vrot.slane 4
vadd
vrot.slane 2
vadd
vrot.slane 1
vadd
```

这是并行规约：8 个 sublane 求和，不是串行加 7 次，而是 3 轮合并。

---

## 9. VLIW bundle 的特点

VLIW 是 Very Long Instruction Word。

一个 bundle 可以包含多条指令：

```text
bundle:
  dma.hbm_to_vmem
  vld
  vld
  vmov
  scalar address op
  maybe vmatmul
```

它的核心特点：

```text
1. bundle 内部的指令可以并行发射。
2. 编译器静态决定哪些指令能放进同一个 bundle。
3. 硬件不依赖复杂乱序执行，而是按编译器安排执行。
4. 一个 bundle 可以同时使用多个硬件单元，例如 DMA、MXU、VPU、Scalar。
5. 好的 schedule 会把数据搬运、地址计算、当前 tile 计算、下一 tile 预取交织起来。
```

所以看到 bundle 时，不要把它理解成：

```text
bundle 中的指令一条条串行执行
```

更应该理解成：

```text
这是编译器打包好的并行发射包。
```

文章第一篇说明，toy program 最终被拆成多个 fusion 和一个 TLP；`multiply_reduce_fusion` 这样的 kernel 会生成几十个 bundle，TLP 负责把多个 fusion 和 DMA 调度起来。

bundle count 有参考意义，但不是唯一性能指标：

```text
bundle 少不一定快
bundle 多不一定慢
HBM traffic、MXU 利用率、DMA overlap、stall、register pressure 都会影响性能
```

---

## 10. TLP：Top Level Program

TLP 是整个 compiled computation 的总控程序。它不是某个 fusion 的内部实现，而是调度多个 fusion/custom-call 和 DMA。

第一篇中可以粗略理解为：

```text
copy w1 HBM -> VMEM
call multiply_reduce_fusion

start copy w2 HBM -> VMEM

call add_sqrt_fusion
call reduce_max fusion
call exp/reduce_sum fusion

wait w2 copy done
call final matmul fusion

final sync
```

对框架开发者来说：

```text
HLO graph:
  决定有哪些 computation 节点和依赖。

HLO fusion:
  决定哪些 op 合成 kernel。

TLP:
  决定 kernel 和 DMA 的 top-level schedule。

LLO/VLIW:
  决定 kernel 内部如何使用 TPU 硬件单元。
```

---

# 第二篇：When XLA Isn't Enough

## 11. 第二篇的问题意识

第一篇告诉我们：

```text
XLA/TPU compiler 能把普通 JAX 程序优化得很深。
```

第二篇问：

```text
既然 XLA 这么强，为什么还需要 Pallas？
```

答案：

```text
XLA 能优化你写出来的计算图，
但通常不会自动把 naive attention 改写成 FlashAttention / Splash Attention。
```

这是“图优化”和“算法改写”的边界。

---

## 12. Naive attention 的问题

标准 attention：

```python
scores = Q @ K.T
scores = scores + mask
m = max(scores, axis=-1, keepdims=True)
e = exp(scores - m)
l = sum(e, axis=-1, keepdims=True)
p = e / l
out = p @ V
```

核心中间矩阵：

```text
scores: [heads, q_len, kv_len]
```

文章使用的例子中：

```text
heads = 8
seq_len = 2048
head_dim = 128
```

所以：

```text
scores = [8, 2048, 2048] f32
       = 128MB
```

XLA 会做真正的优化，例如：

```text
fusion.5:
  Q @ K^T + mask + reduce_max
  -> (max, scores)

fusion.2:
  exp(scores - max) + reduce_sum
  -> sum

fusion:
  normalize + S @ V
  -> output
```

这里的 XLA 已经很努力：

```text
matmul 和 mask / max 融合
exp 和 sum 融合
normalize 和 final matmul 融合
```

但问题仍然存在：

```text
完整 scores matrix 仍然被创建出来，并跨 fusion 使用。
```

所以 attention 的瓶颈不只是“op 没 fusion”，而是：

```text
算法本身 materialize 了巨大的 [H,S,S] 矩阵。
```

---

## 13. 为什么不能简单依赖 XLA

XLA 的优化通常是在等价计算图范围内做：

```text
融合 producer/consumer
消除冗余 broadcast
代数化简
layout assignment
tile scheduling
memory placement
```

但 FlashAttention / Splash Attention 本质上做了更深的事情：

```text
把 attention 的执行方式改成 streaming over KV blocks。
```

换句话说：

```text
naive:
  先完整生成 scores
  再 softmax
  再乘 V

Splash:
  每次只生成一个 scores tile
  边扫 KV tile 边维护 online softmax 状态
  从不生成完整 scores
```

这属于算法级重写。普通 XLA pass 很难从任意 attention graph 自动证明并改写成这种形式，尤其还要考虑 mask、sparsity、GQA/MQA、numerical stability、tile size、memory constraints。

---

## 14. 先解决一个直觉误区：KV tile 结果为什么是累加，不是 cat

对一个 query 向量 `q`，attention 是：

```text
s_j = q @ k_j
p_j = softmax(s)_j
out = sum_j p_j * v_j
```

如果 KV 被切成两个 tile：

```text
tile 0: k0, k1, k2
tile 1: k3, k4, k5
```

那么 scores 可以概念上拼接：

```text
scores_tile0 = [s0, s1, s2]
scores_tile1 = [s3, s4, s5]
scores = cat([scores_tile0, scores_tile1])
```

但是最终输出不是 scores。最终输出是：

```text
out =
  p0 * v0 + p1 * v1 + p2 * v2
  + p3 * v3 + p4 * v4 + p5 * v5
```

按 tile 写：

```text
out =
  sum_{j in tile0} p_j * v_j
  + sum_{j in tile1} p_j * v_j
```

所以：

```text
scores 可以在 KV 维度 cat。
output 是沿 KV 维度 weighted sum，因此不同 KV tile 对 output 的贡献要相加。
```

如果把每个 tile 的 output cat 起来，shape 都错了。对每个 query，attention 最终只输出一个 `head_dim` 向量。

---

## 15. Online softmax：为什么需要 m、l、o

先只看一个 query。它要 attend 到很多 key/value：

```text
key/value 位置：1, 2, 3, ..., N
score：a_1, a_2, a_3, ..., a_N
value：v_1, v_2, v_3, ..., v_N
```

标准 attention 输出是：

$$
\mathrm{Out}
=
\sum_{t=1}^{N}
\frac{e^{a_t}}{\sum_{r=1}^{N} e^{a_r}}
v_t
$$

为了数值稳定，通常减去最大值：

$$
M = \max_{1 \le t \le N} a_t
$$

于是：

$$
\mathrm{Out}
=
\frac{
\sum_{t=1}^{N} e^{a_t - M} v_t
}{
\sum_{t=1}^{N} e^{a_t - M}
}
$$

这里可以拆出两个量：

$$
O = \sum_{t=1}^{N} e^{a_t - M} v_t
$$

$$
L = \sum_{t=1}^{N} e^{a_t - M}
$$

最后：

$$
\mathrm{Out} = \frac{O}{L}
$$

所以：

```text
L = softmax denominator，分母，也可以理解成总权重
O = softmax @ V 的 numerator，分子，也就是未归一化的加权 V 总和
```

注意 shape：

```text
L:
  对每个 query 是一个标量。
  对一个 Q block 是 [bq] 或 [bq, 1]。

O:
  对每个 query 是一个 head_dim 向量。
  对一个 Q block 是 [bq, head_dim]。
```

### 15.1 分块之后，local_sum 和 local_out 是什么

现在把 KV 分成两个 block：

```text
block A: 位置 1,2,3
block B: 位置 4,5,6
```

如果已经处理完 block A，我们维护：

$$
m_A = \max(a_1,a_2,a_3)
$$

$$
L_A =
e^{a_1-m_A} +
e^{a_2-m_A} +
e^{a_3-m_A}
$$

$$
O_A =
e^{a_1-m_A}v_1 +
e^{a_2-m_A}v_2 +
e^{a_3-m_A}v_3
$$

这里：

```text
m_A = 已处理 blocks 的最大 score
L_A = 已处理 blocks 的分母累计
O_A = 已处理 blocks 的加权 V 分子累计
```

现在来了 block B：

$$
m_B = \max(a_4,a_5,a_6)
$$

新的全局最大值是：

$$
m_{AB} = \max(m_A, m_B)
$$

block B 自己的贡献必须用新的最大值 \(m_{AB}\) 来算：

$$
L_B =
e^{a_4-m_{AB}} +
e^{a_5-m_{AB}} +
e^{a_6-m_{AB}}
$$

$$
O_B =
e^{a_4-m_{AB}}v_4 +
e^{a_5-m_{AB}}v_5 +
e^{a_6-m_{AB}}v_6
$$

这两个就是代码里的：

```text
local_sum = L_B
local_out = O_B
```

也就是：

```text
local_sum:
  当前 KV block 对 softmax 分母的贡献。

local_out:
  当前 KV block 对 attention 输出分子的贡献。
```

对应到 tile 写法：

```text
scores_block = Q_block @ K_block^T
p_block = exp(scores_block - m_new)

local_sum = reduce_sum(p_block, axis=KV_block_dim)
local_out = p_block @ V_block
```

### 15.2 为什么旧的 O/L 要乘 correction

问题是：

```text
旧的 L_A / O_A 是用旧最大值 m_A 算的。
现在新的最大值变成了 m_AB。
```

为了把旧贡献换到新的 max 坐标系，需要缩放：

$$
\alpha = e^{m_A - m_{AB}}
$$

于是：

$$
L_A' = \alpha L_A
$$

$$
O_A' = \alpha O_A
$$

这是因为：

$$
e^{a_t - m_{AB}}
=
e^{a_t - m_A} \cdot e^{m_A - m_{AB}}
$$

所以旧累计值都要乘同一个缩放因子。

### 15.3 l_new 和 o_new 是什么

合并旧 blocks 与当前 block：

$$
L_{AB}
=
\alpha L_A + L_B
$$

$$
O_{AB}
=
\alpha O_A + O_B
$$

这就是代码：

```python
l_new = correction * l_prev + local_sum
o_new = correction * o_prev + local_out
```

变量对照：

```text
correction = alpha
l_prev     = L_A
o_prev     = O_A
local_sum  = L_B
local_out  = O_B
l_new      = L_AB
o_new      = O_AB
```

最后扫完所有 KV blocks：

$$
\mathrm{Out}
=
\frac{O_{\mathrm{final}}}{L_{\mathrm{final}}}
$$

也就是：

```python
output = o_final / l_final
```

一句话：

```text
Attention = 加权 value 之和 / 权重之和

O = 加权 value 之和，也就是分子
L = 权重之和，也就是分母

online softmax 只是分块处理时，边走边维护 O 和 L。
```

---

## 16. Pallas 是什么

Pallas 是 JAX 的 custom kernel language。官方文档将它描述为 JAX extension，用于为 GPU/TPU 写 custom kernels，同时保留一部分 JAX tracing 和 `jax.numpy` 风格。Pallas API 仍然是 experimental。

简化理解：

```text
JAX:
  写 tensor program。

Pallas:
  写 tile program。
```

Pallas 不是手写汇编。你不直接写 VLIW 指令，而是显式表达：

```text
grid 怎么划分
每个 program 处理哪个 tile
输入输出如何 block
scratch 如何跨 iterations 保存
哪些 metadata 放 SMEM
```

后端仍然负责：

```text
MXU/VPU 映射
DMA scheduling
VMEM layout
VLIW bundle packing
```

---

## 17. Pallas kernel 语法核心

一个典型 Pallas kernel 长这样：

```python
def kernel(q_ref, k_ref, v_ref, o_ref, scratch_ref):
    q = q_ref[...]
    k = k_ref[...]
    v = v_ref[...]

    result = ...

    o_ref[...] = result
```

这里的参数不是普通 JAX array，而是 `Ref`。

`Ref` 可以理解成：

```text
kernel 内部看到的一块可读写内存视图
```

读取：

```python
x = x_ref[...]
```

写入：

```python
o_ref[...] = y
```

调用时用：

```python
pl.pallas_call(
    kernel,
    out_shape=...,
    grid=...,
    in_specs=...,
    out_specs=...,
    scratch_shapes=...,
    compiler_params=...,
)(q, k, v, ...)
```

核心概念表：

| 概念 | 作用 |
|---|---|
| `Ref` | kernel 内部的可读写内存视图 |
| `grid` | kernel program 的多维迭代空间 |
| `BlockSpec` | 每个 grid point 如何映射到 input/output 的 tile |
| `index_map` | 从 grid indices 返回 array block indices |
| `scratch_shapes` | 跨 grid iteration 持久存在的临时 buffer |
| `memory_space` | 指定 VMEM / SMEM 等 |
| `compiler_params` | 给 TPU backend 的编译 hint |

---

## 18. grid 怎么理解

文章中的 Splash grid 可以抽象成：

```text
grid = (num_heads, num_q_blocks, num_kv_blocks)
```

一个 grid point：

```text
(h, i, j)
```

表示：

```text
h: 第几个 attention head
i: 第几个 Q block
j: 第几个 KV block
```

概念上像三层循环：

```python
for h in range(num_heads):
    for i in range(num_q_blocks):
        initialize m, l, o for this (h, i)

        for j in range(num_kv_blocks):
            scores = Q[h, i] @ K[h, j].T
            update m, l, o

        O[h, i] = o / l
```

实际 Pallas/TPU backend 可以根据 dimension semantics 和 pipeline 策略重排或并行化某些维度。

---

## 19. dimension_semantics 怎么理解

文章提到类似：

```text
dimension 0: heads       -> parallel
dimension 1: q_blocks    -> arbitrary
dimension 2: kv_blocks   -> arbitrary
```

先不要把它理解成 tensor shape 的维度。它是 `grid` 的维度。

```text
grid = (h, i, j)
```

因此：

```text
dimension 0 = h = heads
dimension 1 = i = q_blocks
dimension 2 = j = kv_blocks
```

### 19.1 heads 维为什么 parallel

不同 attention head 之间没有依赖：

```text
head 0 不需要 head 1 的 m/l/o
head 1 不需要 head 0 的 m/l/o
```

所以可以告诉编译器：

```text
heads 维 iteration 可以并行或自由调度。
```

### 19.2 q_blocks 维为什么数学上独立

不同 Q block 也有自己的输出和 scratch：

```text
Q block 0 有自己的 m/l/o
Q block 1 有自己的 m/l/o
```

数学上它们是独立的。不过文章示例中可能仍把它标成 `arbitrary`，这是保守选择，表示不要让编译器对这一维做过强假设。

### 19.3 kv_blocks 维为什么有依赖

对固定 `(h, i)`，要沿 `j` 扫过 KV blocks：

```text
j = 0 -> 得到 m0, l0, o0
j = 1 -> 需要读 m0, l0, o0，更新成 m1, l1, o1
j = 2 -> 需要读 m1, l1, o1，更新成 m2, l2, o2
```

所以 KV block 维不能随便并行或乱序，因为 online softmax 的状态沿这一维传递。

一句话：

```text
parallel across heads
independent across Q blocks
accumulate across KV blocks
```

---

## 20. BlockSpec 怎么理解

`BlockSpec` 定义：

```text
某个 grid point 应该看到数组的哪一个 tile。
```

例如 Q：

```python
pl.BlockSpec(
    block_shape=(None, bq, head_dim),
    index_map=lambda h, i, j, *_: (h, i, 0),
)
```

假设 Q shape 是：

```text
Q: [num_heads, q_len, head_dim]
```

`index_map` 返回：

```text
(h, i, 0)
```

分别对应：

```text
axis 0: head 轴       -> 第 h 个 block
axis 1: sequence 轴   -> 第 i 个 Q block
axis 2: head_dim 轴   -> 第 0 个 feature block
```

结合：

```text
block_shape=(None, bq, head_dim)
```

可以近似理解为：

```python
Q[h, i*bq:(i+1)*bq, 0:head_dim]
```

注意：

```text
None 不是“读完整维度”。
None 表示该维度取 size-1 的 slice，并在 kernel 内部 squeeze 掉。
```

所以 kernel 内部看到的 Q tile 通常是：

```text
[bq, head_dim]
```

而不是：

```text
[1, bq, head_dim]
```

### 20.1 为什么 `lambda h, i, j: (h, j, 0)` 中的 0 表示读完整 head_dim

它不是因为 `0` 有“完整”的特殊含义。

对于 K：

```text
K: [num_kv_heads, kv_len, head_dim]
```

BlockSpec：

```python
pl.BlockSpec(
    block_shape=(None, bkv, head_dim),
    index_map=lambda h, i, j: (h, j, 0),
)
```

真实 slice 近似是：

```python
K[h, j*bkv:(j+1)*bkv, 0*head_dim:1*head_dim]
```

也就是：

```python
K[h, j*bkv:(j+1)*bkv, 0:head_dim]
```

所以：

```text
0 只是 head_dim 这一轴的 block index。
因为 block_shape 的最后一维恰好等于完整 head_dim，
所以从第 0 个 block 开始读，就读到了完整 feature 维。
```

如果 block_shape 写成：

```python
block_shape=(None, bkv, 32)
```

而 `head_dim=128`，那么：

```text
0 -> 读 0:32
1 -> 读 32:64
2 -> 读 64:96
3 -> 读 96:128
```

此时 `0` 就不代表完整 head_dim 了。

---

## 21. K/V 的 sparse indirection

Dense attention 里，K/V 的 index map 可以很简单：

```python
lambda h, i, j: (h, j, 0)
```

意思是：

```text
当前 program 是 (h, i, j)
就读取第 h 个 head、第 j 个 KV block
```

Sparse attention 里，某些 KV block 对当前 Q block 完全被 mask，没必要加载和计算。Splash 使用 `data_next_ref` 做间接寻址：

```python
def k_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref):
    next_j, *_ = _next_nonzero(
        h, i, j,
        data_next_ref,
        block_mask_ref,
        mask_next_ref,
    )
    return (h // q_heads_per_kv_head, next_j, 0)
```

直觉：

```text
本来要读 KV block j
但如果 j 是 fully masked block
就通过 data_next_ref 跳到下一个有效 block next_j
```

例如：

```text
j:       0  1  2  3  4  5
valid:   1  0  0  1  0  1
```

那么：

```text
data_next[h,i,1] 可能指向 3
data_next[h,i,2] 也可能指向 3
```

这样 kernel 在寻址阶段就跳过无效块：

```text
不是 load 之后发现不用，
而是根本不 load 那些 fully masked KV blocks。
```

`h // q_heads_per_kv_head` 用于 GQA/MQA：

```text
Q heads 可能比 KV heads 多。
多个 Q heads 共享同一个 KV head。
```

例如：

```text
Q heads = 8
KV heads = 2
q_heads_per_kv_head = 4
```

则：

```text
Q head 0,1,2,3 -> KV head 0
Q head 4,5,6,7 -> KV head 1
```

---

## 22. Pallas 暴露的 TPU memory hierarchy

文章中的核心内存概念：

```text
HBM:
  片外大内存。
  Q/K/V/O 原始大数组通常在这里。

VMEM:
  片上 SRAM。
  默认 BlockSpec refs 指向 VMEM tile。

SMEM:
  scalar memory。
  适合小索引、mask metadata、next pointer、control-flow decision。

Scratch:
  kernel iterations 之间持久存在的临时 buffer。
```

### 22.1 VMEM：BlockSpec refs 默认在这里

例如：

```python
pl.BlockSpec((bq, head_dim), index_map)
```

外面的 Q/K/V 大数组在 HBM，进入 kernel 时当前 tile 被搬到 VMEM。kernel 内部看到的是：

```text
Q tile: [bq, head_dim]
K tile: [bkv, head_dim]
V tile: [bkv, head_dim]
```

### 22.2 SMEM：放小的控制信息

例如：

```python
pl.BlockSpec(
    (num_heads,),
    lambda *_: (0,),
    memory_space=pltpu.SMEM,
)
```

适合：

```text
block mask metadata
data_next pointer
mask_next pointer
小整数索引
控制流判断
```

这些不是大矩阵数据，而是告诉 kernel：

```text
当前 block 是否有效
下一个有效 block 是谁
怎样跳过 masked region
```

### 22.3 Scratch：online softmax 的记忆

文章中的：

```python
m_scratch_ref
l_scratch_ref
o_scratch_ref
```

分别保存：

```text
m: running max
l: running denominator
o: running unnormalized output numerator
```

对固定 `(h, i)`，沿 `j` 扫 KV blocks：

```text
j=0:
  初始化并写 m/l/o scratch

j=1:
  读旧 m/l/o scratch
  更新成新的 m/l/o

j=2:
  继续

最后:
  output = o / l
  写最终 O
```

关键：

```text
这些 scratch 在 VMEM 里跨 grid iterations 保留，
不需要每个 KV tile 都把 m/l/o 写回 HBM 再读回来。
```

这就是 online softmax 能高效实现的关键之一。

---

## 23. Splash Attention 的执行逻辑

对一个 Q block 和一个 KV block：

```text
Q_tile: [bq, head_dim]
K_tile: [bkv, head_dim]
V_tile: [bkv, head_dim]
```

当前 tile scores：

```text
scores_tile = Q_tile @ K_tile.T
```

shape：

```text
[bq, bkv]
```

这个 scores tile 只在 VMEM 中短暂存在。然后：

```text
更新 running max m
更新 running denominator l
更新 running output numerator o
```

再处理下一个 KV tile。

最终：

```text
O_tile = o / l
```

写回 HBM。

对比 naive：

```text
naive:
  scores_full = Q @ K.T
  scores_full shape = [H, S, S]
  scores_full 会跨 fusion materialize

Splash:
  scores_tile = Q_tile @ K_tile.T
  scores_tile shape = [bq, bkv]
  scores_tile 不跨 kernel 边界
```

这就是 HBM traffic 大幅下降的原因。

---

## 24. Pallas kernel 到 HLO：为什么是 custom-call

Pallas kernel 在 HLO 中通常表现为：

```text
custom-call(...)
custom_call_target="tpu_custom_call"
```

这意味着：

```text
XLA HLO graph 不再展开 Pallas kernel 内部细节。
```

XLA 看到的是：

```text
这里有一个 custom-call
输入是 Q/K/V/mask metadata
输出是 O
```

Pallas kernel body 作为 MLIR payload 交给 Mosaic。之后 Mosaic 和 TPU backend 继续降到：

```text
LLO -> VLIW bundles
```

所以 Pallas 的定位不是：

```text
绕开整个 TPU compiler
```

而是：

```text
绕开 XLA 高层图优化对 kernel 内部算法的表达限制，
但仍然使用 Mosaic / TPU backend 做底层 lowering 和 scheduling。
```

---

# 25. 两篇文章的最终对比

| 维度 | 普通 JAX / XLA | Pallas / Splash |
|---|---|---|
| 你写的东西 | 高层 tensor program | tiled kernel program |
| XLA 是否看见内部 | 看见完整 HLO graph | HLO 只看见 custom-call |
| 优化主要来自 | fusion、layout、tiling、memory assignment、VLIW scheduling | 你显式表达算法级 tiling / streaming，后端继续调度 |
| attention scores | 完整 `[H,S,S]` tensor 存在 | 只有 `[bq,bkv]` tile 短暂存在 |
| softmax | 对完整 scores 分阶段处理 | online softmax，维护 `m/l/o` |
| 中间矩阵 HBM traffic | 大 | 小 |
| 程序员负担 | 低，主要写 JAX | 高，需要理解 grid、BlockSpec、scratch、memory space |
| 适用场景 | 常规模型算子、可由 XLA 表达的图优化 | 编译器不会自动发现的算法级优化 |

---

# 26. 给模型框架开发者的复习清单

## 26.1 HLO 层重点看什么

读 HLO 时优先看：

```text
1. 是否出现巨大中间 tensor
   例如 attention scores [H,S,S]

2. producer/consumer 是否 fusion
   elementwise + reduction + broadcast 是否合并

3. common producer 如何处理
   multi-output fusion / duplicate / materialize / recompute

4. layout 是否合理
   是否出现 TPU tile annotation

5. memory assignment
   大 tensor 是否 HBM
   小中间结果是否 VMEM

6. copy-start/copy-done
   是否有异步拷贝和 compute overlap
```

## 26.2 LLO 层只需先会识别模式

```text
vmatpush / vmatmul / vpop
  -> MXU matmul

vadd / vmul / vsel / vcmp / vpow2 / vrsqrt
  -> VPU vector compute

vxpose / vpop.trf / vrot.slane
  -> transpose / shuffle / reduction

dma.hbm_to_vmem / dma.done.wait
  -> HBM 与 VMEM 之间数据搬运

bundle { ... }
  -> VLIW 静态并行发射包
```

## 26.3 Pallas 层重点看什么

```text
1. grid
   每个 program id 代表什么？
   哪些维度 independent，哪些维度 carry state？

2. BlockSpec
   每个 grid point 看到数组的哪个 tile？

3. index_map
   是否存在 sparse indirection？
   是否有 GQA/MQA head mapping？

4. scratch
   哪些状态跨 grid iteration 保留？
   是否避免 HBM round-trip？

5. memory_space
   哪些 metadata 放 SMEM？
   哪些 tile 放 VMEM？

6. custom-call
   HLO 是否只看见 opaque call？
   kernel body 是否进入 Mosaic？
```

---

# 27. 最后用一句话复习

第一篇：

```text
JAX 写自然的 tensor program，XLA 可以把它优化成带 fusion、VMEM placement、DMA overlap 和 VLIW bundle 的 TPU 程序。
```

第二篇：

```text
当性能瓶颈来自算法级 materialization，例如完整 attention matrix，XLA 的 fusion 仍然不够；Pallas 让我们直接表达 streaming tiled algorithm，用 online softmax 避免完整 scores 落 HBM。
```

最核心判断：

```text
如果问题是“这些 op 能不能合并得更好”，先看 XLA/HLO fusion。
如果问题是“这个巨大中间 tensor 是否本来就不该存在”，考虑算法改写和 Pallas。
```

---

## 参考资料

- Patrick Toulme, [From JAX to VLIW: Tracing a Computation Through the TPU Compiler](https://patricktoulme.substack.com/p/from-jax-to-vliw-tracing-a-computation)
- Patrick Toulme, [When XLA Isn't Enough: From Pallas to VLIW](https://patricktoulme.substack.com/p/when-xla-isnt-enough-from-pallas)
- JAX documentation, [Pallas: a JAX kernel language](https://docs.jax.dev/en/latest/pallas/index.html)
- JAX documentation, [Pallas Quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html)
- JAX documentation, [Grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
- JAX documentation, [jax.experimental.pallas.BlockSpec](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.BlockSpec.html)
- JAX documentation, [Pallas TPU details](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
- JAX documentation, [Pallas TPU pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html)
