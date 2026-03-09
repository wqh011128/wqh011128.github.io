---
layout: post
title: "JAX and TPU学习记录"
date: 2026-03-26
categories:
  - blog
---

# JAX + TPU 两个重点问题

这次其实只需要抓住两个问题：

1. **如何监控：我到底有没有真正用上 TPU，运行时状态怎么看？**
2. **`XLA_FLAGS` 和 `LIBTPU_INIT_ARGS` 到底分别控制什么，会额外产出什么？**

---

## 1. 如何监控 JAX + TPU

我现在更倾向于把“监控”分成三层看，因为很多混淆都来自把这三层混在一起。

### 1.1 第一层：机器上有没有 TPU 设备

这一层回答的是：

> **机器有没有挂上 TPU 芯片。**

常用命令：

```bash
tpu-info
tpu-info --streaming --rate 2
```

如果 `tpu-info` 能看到型号、芯片数量、`/dev/vfio/*`，说明**设备层面**是存在的。

但这里要注意一个关键点：

- **能看到 TPU 芯片，不等于能看到 TPU 利用率**
- `HBM Usage`、`Duty cycle`、`TensorCore Utilization` 出现 `N/A`，通常说明 **runtime 指标没有正确暴露出来**

最常见原因有这些：

- 当前环境没有正确接上 `libtpu`
- `tpu-info` 和当前环境不兼容
- 当前没有真正跑 TPU workload
- runtime metrics 本身还不可用

可以先做一个最小检查：

```bash
python - <<'PY'
try:
    import libtpu
    print("libtpu OK:", libtpu.__file__)
except Exception as e:
    print("libtpu import failed:", repr(e))
PY
```

如果这里连 `libtpu` 都导不进来，那么 `tpu-info` 读不到完整指标就不奇怪。

### 1.2 第二层：JAX 有没有成功连到 TPU backend

这一层回答的是：

> **JAX 这个进程，是否真的初始化了 TPU backend。**

最小判断代码：

```python
import jax

print("backend:", jax.default_backend())
print("devices:", jax.devices())

assert jax.devices()[0].platform == "tpu"
```

如果输出类似：

```python
backend: tpu
devices: [TpuDevice(...)]
```

说明的事情只有一件：

> **JAX backend 已经接到 TPU 了。**

但这还不等于“整段 Python 程序都在 TPU 上跑”。  
真正上 TPU 的，只是 JAX/XLA 编译后的数组计算部分；数据加载、Python 循环、日志打印仍然主要在 host 侧。

### 1.3 第三层：具体这一步计算是否真的在 TPU 上执行

这一层回答的是：

> **不是环境认到了 TPU，而是这一步 `step()`、这次 matmul、这次训练迭代，是否真的在 TPU 上算了。**

最实用的判断方式有三种。

#### 方法 1：看输出数组落在哪个设备上

```python
import jax
import jax.numpy as jnp

@jax.jit
def step(x):
    return x @ x + 1

x = jnp.ones((4096, 4096))
y = step(x)
y.block_until_ready()

print("device:", y.device)
print("sharding:", y.sharding)
print("addressable shard devices:", [s.device for s in y.addressable_shards])
```

如果 `device` 或 `addressable_shards` 对应的是 `TpuDevice(...)`，说明这一步结果确实落在 TPU 上。

#### 方法 2：显式同步 `block_until_ready()`

这是判断 TPU 计算是否**真的完成**时最容易漏掉的一步。

JAX 默认是 **asynchronous dispatch**。也就是说：

- Python 线程可能只是把任务提交给 TPU
- 任务还在设备端排队或执行
- 主线程已经继续往下跑了

所以像下面这种代码：

```python
y = step(x)
print("done")
```

很多时候只能说明 **dispatch 完成了**，不能说明 **TPU 已经算完了**。

如果要做准确判断或准确计时，应该这样写：

```python
import time
import jax
import jax.numpy as jnp

@jax.jit
def step(x):
    return x @ x

x = jnp.ones((4096, 4096))

t0 = time.time()
y = step(x)
t1 = time.time()

y.block_until_ready()
t2 = time.time()

print("dispatch time:", t1 - t0)
print("real execution time:", t2 - t0)
```

通常：

- `dispatch time` 只是任务提交时间
- `real execution time` 才更接近真实 TPU 执行时间

#### 方法 3：需要铁证时，用 profiler

如果要看更硬的证据，直接做 profile。

```python
import jax

jax.profiler.start_server(9999)
```

或者：

```python
import jax

jax.profiler.start_trace("/tmp/profile-data")
# run workload
jax.profiler.stop_trace()
```

这时候关心的就不是“数组在哪”，而是 trace 里有没有真正的 TPU activity。

### 1.4 监控时最容易误判的三件事

#### 误判 1：`tpu-info` 有芯片信息，就以为 TPU 一定在正常工作

不对。  
这最多说明**设备存在**，不说明 **runtime metrics 可用**，更不说明 **JAX 已经在跑 TPU 计算**。

#### 误判 2：程序很快打印 `done`，就以为 TPU 已经算完

不对。  
这通常只是 asynchronous dispatch 的表现，必须配合 `block_until_ready()` 看。

#### 误判 3：报 `The TPU is already in use by process with pid ...`，以为是 JAX 坏了

这类报错的核心含义其实很简单：

> **另一独立进程已经先初始化并占住了 TPU runtime。**

常用检查命令：

```bash
ps -fp 22589
sudo lsof -w /dev/vfio/*
tpu-info
```

这通常是另一个脚本、另一个终端、`tmux`、notebook kernel 还没退出，不是 TPU “坏了”。

### 1.5 一个最小但可靠的监控顺序

如果我只想快速判断 TPU 状态，我会按这个顺序看：

1. `tpu-info`
   先确认机器层面有没有 TPU 设备。
2. `jax.default_backend()` 和 `jax.devices()`
   确认当前 JAX 进程有没有成功连到 TPU backend。
3. 输出数组的 `device / sharding / addressable_shards`
   确认具体计算结果是不是落在 TPU 上。
4. `block_until_ready()`
   确认这一步不是只完成 dispatch，而是真的执行完。
5. profiler
   需要最硬证据时再上。

---

## 2. HLO / LLO / compile / backend 到底是什么关系

如果只想抓主线，可以先记这一条：

```text
trace -> lower -> HLO -> XLA 优化 -> backend lowering(LLO) -> executable
```

这一节其实就是在解释这条链路，以及它在项目里意味着什么。

### 2.1 四个词先分清

#### HLO 是什么

HLO 可以理解成 XLA 里的**高层中间表示**。  
它更接近张量计算图，也更适合看：

- 图优化
- fusion
- sharding
- layout 传播

在项目里，下面两种写法都会碰到 HLO：

```python
lowered_func.as_text("hlo")
lowered_func.compile().as_text()
```

区别是：

- `as_text("hlo")` 更像是直接查看 lower 后的 HLO
- `compile().as_text()` 则是在真正编译之后查看结果

#### compile() 是什么

`compile()` 不是“把内容打印出来”，而是：

- 真正触发一次编译
- 让 lowered computation 继续进入 XLA 和 backend 的编译链
- 最终生成可执行对象

所以这句代码：

```python
lowered_func.compile().as_text()
```

它的关键点不在 `as_text()`，而在 **`compile()` 已经把后面的编译过程跑起来了**。

#### backend 是什么

这里的 backend 可以理解为：

> **负责把编译器中间表示继续变成目标设备可执行结果的那一层。**

在这个问题里，主要就是 TPU backend / `libtpu`。

#### LLO 是什么

LLO 可以粗略理解成**更靠近后端和设备的一层表示**。  
它的位置比 HLO 更低，更接近最终 executable。

所以可以把它粗略记成：

```text
HLO -> 更低层后端表示(LLO) -> executable
```

LLO 通常不是项目自己构造的数据结构，而是 TPU backend / `libtpu` 在编译过程中额外吐出来的调试产物。

### 2.2 这对当前项目意味着什么

当前项目里，如果只是：

```python
lowered_func.as_text("hlo")
```

更偏向“看一份 HLO 文本”。

如果是：

```python
lowered_func.compile().as_text()
```

那就不只是看文本了，而是会真的触发一次编译过程。  
也正因为如此，编译链后面的很多东西都会被触发出来，比如：

- 优化后的 HLO
- `XLA_FLAGS` 对应的 HLO pass dump
- `LIBTPU_INIT_ARGS` 对应的 LLO dump

所以项目里“导出优化后 HLO”这件事，本质上已经进入了**真实编译**，而不只是字符串导出。

### 2.3 为什么会同时看到 `xla_dump` 和 `llo_dump`

因为一次 `compile()` 本来就会经过多层编译链路。

如果设置：

```bash
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_pass_re=.*"
LIBTPU_INIT_ARGS="--xla_jf_dump_to=/tmp/dump_llo"
```

那么通常会同时出现两类 dump：

- `xla_dump`
  更偏 XLA / HLO pass，适合看编译器中高层是怎么改图的。
- `llo_dump`
  更偏 TPU backend / `libtpu`，适合看更靠近设备的一层 lowering。

它们不是重复关系，而是编译链上**不同阶段**的调试产物。

### 2.4 `XLA_FLAGS` 和 `LIBTPU_INIT_ARGS` 分别控制什么

如果只看用途，可以直接记成：

- **`XLA_FLAGS`**：看 HLO / XLA pass
- **`LIBTPU_INIT_ARGS`**：看 TPU backend / LLO lowering

更重要的一点是：

> **它们通常不是在改变最终 HLO 的语义，而是在让你看到更多编译过程中的中间状态。**

所以同时打开时，通常会有三类结果：

1. 项目自己主动导出的 HLO 文本
2. `/tmp/xla_dump` 里的 HLO pass dump
3. `/tmp/dump_llo` 里的 TPU/LLO dump

前者适合做稳定分析，后两者适合排查“图到底是在哪一层被改掉的”。

### 2.5 为什么不能在每个 layer 的 `compile()` 前动态改 `LIBTPU_INIT_ARGS`

直觉上很容易想到这种写法：

```python
os.environ["LIBTPU_INIT_ARGS"] = "..."
lowered_func.compile()
os.environ.pop("LIBTPU_INIT_ARGS", None)
```

但这通常不可靠，原因很简单：

- `LIBTPU_INIT_ARGS` 属于 backend / runtime 初始化参数
- 它往往在 backend 初始化时就已经读取了
- 不是每次 `compile()` 都重新读一次

所以程序跑到 `save_hlo_and_data()` 时，backend 很可能早就初始化完了。  
这时再临时改环境变量，大概率不会影响当前这次 `compile()`。

因此它更像是：

- **进程启动前配置**
- **backend 初始化前配置**

而不是 layer 级别的 Python 参数。

### 2.6 这次修改真正解决的问题

当前项目在导出优化后 HLO 时，会触发真实编译：

```python
lowered_func.compile().as_text()
```

这次修改的目标不是改变编译行为本身，而是：

- 继续保留全局 `LIBTPU_INIT_ARGS="--xla_jf_dump_to=..."`
- 在每个 layer 的 `compile()` 前后观察全局 LLO dump 目录
- 把本轮新增或变化的文件归档到当前 layer 目录

也就是把“全局 dump”尽量整理成“按 layer 可读”。

### 2.7 当前项目采用的最小方案

因为 `LIBTPU_INIT_ARGS` 只能全局生效，所以当前最稳妥的办法不是“让 libtpu 直接按 layer dump”，而是：

> **全局 dump，再由项目代码按每次 `compile()` 的时间窗口做归档。**

具体流程是：

1. compile 前读取全局 LLO dump 根目录
2. 先做一次快照
3. 执行当前 layer 的 `compile()`
4. compile 后再做一次快照
5. 找出本轮新增或变化的文件
6. 复制到 `output/<layer_name>/llo_dump`

这个方案的优点是：

- 不改 backend 初始化逻辑
- 不需要多进程
- 改动面小

它的限制也很明确：

- 这是按“时间窗口”归因，不是按文件内容精确识别 layer
- 如果 backend 异步写文件，可能有少量归因误差
- 如果多个重复 block 共用同一个名字，仍然不容易区分实例

### 2.8 运行时怎么用

如果要让 LLO 归档生效，运行前需要先设置全局环境变量：

```bash
LIBTPU_INIT_ARGS="--xla_jf_dump_to=/tmp/dump_llo" \
uv run entrypoints.py \
  --model_type llama3 \
  --config-path configs/llama/llama3-8B-sq4k-bf16-L1.ini \
  --output_dir ./outputs/ \
  --export_after_optimize
```

如果还想同时看 HLO pass dump，再加：

```bash
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_pass_re=.*"
```

运行后通常会有两类目录：

- 全局原始目录：`/tmp/dump_llo`
- 项目归档目录：`./outputs/<config_name>/<layer_name>/llo_dump`

---

## 3. 最后压缩成几句话

1. **监控 TPU 要分三层：设备是否存在、JAX 是否连上、具体计算是否真的执行。**
2. **`compile()` 会真正触发编译，所以它不只是在“导出文本”，也会带出后端 dump。**
3. **HLO 更偏 XLA 高层表示，LLO 更偏 TPU backend 更低层的 lowering。**
4. **`XLA_FLAGS` 主要看 HLO/XLA pass，`LIBTPU_INIT_ARGS` 主要看 TPU backend/LLO。**
5. **当前项目不能让 libtpu 直接按 layer dump，所以采用的是“全局 dump + 每次 compile 后按 layer 归档”。**
