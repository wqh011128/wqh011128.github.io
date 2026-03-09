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

## 2. `XLA_FLAGS` 和 `LIBTPU_INIT_ARGS` 到底分别是什么

这两个变量最容易混淆的点在于：它们都和 XLA / TPU 编译过程有关，但**不在同一层**。

### 2.1 先记结论

- **`XLA_FLAGS`**：更偏 **XLA 编译器 / HLO pass** 这一层
- **`LIBTPU_INIT_ARGS`**：更偏 **TPU backend / libtpu runtime / LLO lowering** 这一层

所以它们不是同一个入口的两个写法，而是作用在**不同层级**的参数。

### 2.2 `XLA_FLAGS` 在控制什么

例如：

```bash
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_pass_re=.*"
```

大致含义是：

- 把 XLA 编译过程中的 dump 文件写到 `/tmp/xla_dump`
- 对匹配 `.*` 的 HLO pass 都进行 dump

所以 `XLA_FLAGS` 更像是在说：

> **把 XLA 编译器这一层的中间状态吐出来给我看。**

它主要对应的是：

- HLO dump
- HLO pass 前后变化
- 文本、图、proto 一类中间产物

### 2.3 `LIBTPU_INIT_ARGS` 在控制什么

例如：

```bash
LIBTPU_INIT_ARGS="--xla_jf_dump_to=/tmp/dump_llo/"
```

它更像是在说：

> **在 TPU 后端 / libtpu 初始化时，把更靠近设备 lowering 的中间结果也吐出来。**

粗略理解就是：

- `XLA_FLAGS` 更偏 **HLO / XLA pass**
- `LIBTPU_INIT_ARGS` 更偏 **TPU backend / LLO / 更接近设备代码生成**

也可以把它理解成：

- 前者更偏“通用编译器视角”
- 后者更偏“TPU 插件和后端视角”

### 2.4 它们会不会改变项目自己导出的 HLO

通常要分成两类看。

#### 第一类：项目代码自己主动导出的 HLO

比如项目里如果有这种逻辑：

- `lowered_func.as_text("hlo")`
- `lowered_func.compile().as_text()`

这类输出是**你的代码自己主动调用 API 导出的文本表示**。

这时 `XLA_FLAGS` 和 `LIBTPU_INIT_ARGS` 的作用并不是“把这个 API 变成另一套接口”，而是：

> **额外让编译器 / TPU backend 再多产出一批调试文件。**

换句话说，项目自己导出的 HLO 仍然是你的主输出；这些环境变量只是额外加旁路 dump。

#### 第二类：编译器和 TPU backend 额外 dump 的中间文件

如果你打开这些参数，通常会额外得到：

- `/tmp/xla_dump` 里的 HLO pass 过程文件
- `/tmp/dump_llo/` 里的 TPU/LLO lowering 过程文件

这些文件更适合回答的问题是：

- 图在某个 pass 前后怎么变了
- 某一步 lowering 到 TPU backend 后长什么样
- 到底是 HLO 层改了，还是更后面的 TPU lowering 改了

所以更准确的说法不是“这些变量改变了最终 HLO 的语义”，而是：

> **它们让你看见更多编译过程中的中间状态。**

### 2.5 同时打开时，实际会得到什么

如果同时设置：

```bash
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_pass_re=.*"
LIBTPU_INIT_ARGS="--xla_jf_dump_to=/tmp/dump_llo/"
```

那通常会得到三套东西：

1. 项目自己输出的稳定结果  
   例如你代码里主动导出的优化前/优化后 HLO 文本。

2. `/tmp/xla_dump` 里的 XLA/HLO pass dump  
   适合看编译器在每个 pass 里怎么改图。

3. `/tmp/dump_llo/` 里的 TPU/LLO dump  
   适合看更靠近 TPU backend 的 lowering 结果。

如果目标是：

- **拿一份稳定结果做分析、比对、复现**  
  优先看项目自己导出的 HLO。

- **排查到底哪个 pass 把图改了**  
  重点看 `XLA_FLAGS` 生成的 dump。

- **排查 TPU 后端 lowering 到底发生了什么**  
  再看 `LIBTPU_INIT_ARGS` 生成的 dump。

### 2.6 一个很重要的前提

这两个环境变量都必须在 **JAX / TPU backend 初始化之前** 就进入进程环境。

也就是说，它们本质上是：

- **进程启动前配置**
- **backend 初始化前配置**

而不是：

- `lowered_func.compile(...)` 的 Python 函数参数
- 代码跑到一半临时加上的调试开关

如果 backend 已经初始化完，再去设置它们，通常就太晚了，或者只会部分生效。

### 2.7 我的实用理解

如果只记一句话，我会记成：

> **`XLA_FLAGS` 用来看 HLO/XLA pass，`LIBTPU_INIT_ARGS` 用来看 TPU backend/LLO lowering。**

---

## 3. 最后压缩成几句话

1. **监控 TPU 要分三层：设备是否存在、JAX 是否连上、具体计算是否真的执行。**
2. **`tpu-info` 能看到芯片，只说明设备存在；看到实时利用率才说明 runtime 指标也可用。**
3. **判断某一步是否真的在 TPU 上完成，最实用的是 `device/sharding` 加 `block_until_ready()`。**
4. **`XLA_FLAGS` 更偏 HLO/XLA pass，`LIBTPU_INIT_ARGS` 更偏 TPU backend/LLO lowering。**
5. **这两个变量通常不是改变最终 HLO 语义，而是让你拿到更多编译中间产物。**
