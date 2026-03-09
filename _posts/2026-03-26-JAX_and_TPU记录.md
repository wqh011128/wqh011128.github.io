---
layout: post
title: "JAX and TPU学习记录"
date: 2026-03-26
categories:
  - blog
---


# 1. JAX + TPU 排查笔记

## 0. 先记住几个核心概念

- **`jax.default_backend()`**：返回当前默认的 XLA backend 名称，比如 `cpu`、`gpu`、`tpu`。  
- **`jax.devices()`**：返回某个 backend 下可见的设备列表。  
- **`jax.Array`**：JAX 的数组对象，数组本身会携带设备与 `sharding` 信息。  
- **`block_until_ready()`**：JAX 默认是 **asynchronous dispatch**，很多计算只是“下发”了，并不代表设备端已经真正算完；要做准确判断或准确计时，通常需要显式同步。  
- **`tpu-info`**：一个 Cloud TPU 侧的监控工具，它从 **`libtpu`** 读取运行时指标，比如 HBM usage、duty cycle、TensorCore utilization。  
- **`libtpu`**：Cloud TPU 的基础软件层，包含驱动、网络库、XLA compiler 和 TPU runtime。  

---

## 1. 问题一：`tpu-info` 能看到 TPU 芯片，但利用率指标是 `N/A`，或者提示 `libtpu` 相关问题

### 问题描述

典型现象是：

- `tpu-info` 能识别出 TPU 型号，比如 `v6e`
- 也能看到 `/dev/vfio/0`
- 但 `HBM Usage`、`Duty cycle`、`TensorCore Utilization` 显示为 `N/A`
- 或者出现类似 `libtpu not found`、`Libtpu metrics unavailable` 的提示

### 可能原因

1. **机器层面已经挂上 TPU 设备了，但当前环境没正确接上 `libtpu`**
2. **`tpu-info` 版本和当前环境不兼容**
3. **当前没有真正使用 TPU 的 framework 进程在跑**
4. **虽然 `libtpu` 存在，但 runtime 指标还没被有效暴露出来**

### JAX / TPU 专业知识

`tpu-info` 不是直接“看硬件寄存器”，它主要是从 **`libtpu`** 读取运行时指标。  
而 `libtpu` 本身不仅仅是一个 Python 包，它背后对应的是 TPU runtime 这一整层基础设施。  
因此，“能看到 TPU 芯片”只说明 **设备存在**；“能看到实时利用率”才说明 **runtime 指标也可用**。这两件事不是同一个层级。

### 简单模拟例子

```python
try:
    import libtpu
    print("libtpu OK:", libtpu.__file__)
except Exception as e:
    print("libtpu import failed:", repr(e))
```

如果这里就失败，说明当前 Python 环境根本没接好 `libtpu`。

### 实用检查命令

```bash
python - <<'PY'
try:
    import libtpu
    print("libtpu OK:", libtpu.__file__)
except Exception as e:
    print("libtpu import failed:", repr(e))
PY

tpu-info
tpu-info --streaming --rate 2
```

这些命令分别用于检查 `libtpu` 是否能导入，以及 `tpu-info` 是否能读到运行时指标。

---

## 2. 问题二：`jax.default_backend()` 是 `tpu`，`jax.devices()` 也有 `TpuDevice`，这说明什么？

### 问题描述

典型输出像这样：

```python
backend: tpu
device_count: 1
devices: [TpuDevice(id=0, process_index=0, ...)]
```

### 可能原因

这通常不是“问题”，而是说明：

1. **JAX 已经成功初始化了 TPU backend**
2. **当前 Python 进程可以看到 TPU 设备**
3. **至少从 backend 连接层面，JAX → TPU 是通的**

### JAX 专业知识

`jax.default_backend()` 只回答一个问题：  

**“JAX 默认打算把计算发到哪个 backend？”**

`jax.devices()` 回答的是：  

**“这个 backend 下有哪些设备对当前进程可见？”**

只要 `jax.devices()[0].platform == 'tpu'` 成立，就可以确认程序已经接到了 TPU。

### 需要特别注意

**这不等于“整个 Python 程序都在 TPU 上跑”。**

实际情况是：

- 数据加载、普通 Python 循环、字符串处理、日志打印，仍然主要发生在 CPU/host 侧
- 真正放到 TPU 上执行的是 **JAX/XLA 编译后的数组计算部分**

### 简单模拟例子

```python
import jax

print("backend:", jax.default_backend())
print("devices:", jax.devices())

assert jax.devices()[0].platform == "tpu"
```

如果断言通过，说明 **JAX backend 已经正确连到 TPU**。

---

## 3. 问题三：怎么确认“具体这一步计算”真的在 TPU 上执行，而不只是环境认到了 TPU？

### 问题描述

你可能已经看到：

- `backend == tpu`
- `devices == [TpuDevice(...)]`

但你还想进一步确认：

> “我的这段矩阵乘法、我的这一步 `step()`、我的训练 forward/backward，真的下发到 TPU 了吗？”

### 可能原因

之所以会产生这个疑问，是因为：

1. JAX 会把很多事情延后到编译或执行阶段
2. JAX 默认是 **asynchronous dispatch**
3. 不是所有 Python 代码都会上 TPU，真正上 TPU 的只是 JAX 数组计算

### JAX 专业知识

有三种常用判断方法：

#### 方法 1：看结果数组的设备 / `sharding`

JAX 的 `jax.Array` 会携带设备布局信息。  
因此，一个很实用的办法是：**看输出数组落在哪个设备上**。如果输出数组的 `device` 或 `addressable_shards` 对应 `TpuDevice`，那这一步结果就是在 TPU 侧持有的。

#### 方法 2：显式同步 `block_until_ready()`

JAX 的 **asynchronous dispatch** 意味着：  
Python 线程可能只是把工作“排队”给设备了，然后立刻继续往下跑；只有当你真的去取值、打印值、转成 NumPy，或者主动同步时，主机才会等待设备完成。  
因此，如果你不做 `block_until_ready()`，你看到的“运行完成”很可能只是 **dispatch 完成**，不是 **TPU 计算完成**。

#### 方法 3：做 profiler

如果你想要更硬的证据，可以对程序进行 profile，检查 trace 中是否出现 TPU operations。

### 简单模拟例子

```python
import jax
import jax.numpy as jnp

@jax.jit
def step(x):
    return x @ x + 1

x = jnp.ones((4096, 4096))
y = step(x)
y.block_until_ready()

print("backend:", jax.default_backend())
print("device:", y.device)
print("sharding:", y.sharding)
print("addressable shard devices:", [s.device for s in y.addressable_shards])
```

如果输出里的 `device` 或 `addressable_shards` 对应的是 `TpuDevice(...)`，并且 `block_until_ready()` 正常完成，那么这一步计算就可以认为已经真实在 TPU 上执行并完成了。

---

## 4. 问题四：报错 `The TPU is already in use by process with pid ...`

### 问题描述

典型报错：

```text
RuntimeError: Unable to initialize backend 'tpu':
ABORTED: The TPU is already in use by process with pid 22589.
Not attempting to load libtpu.so in this process.
```

### 可能原因

最常见的原因有：

1. 你已经有另一个 Python / JAX 进程在占用这块 TPU
2. 之前开的训练脚本没有退出
3. 你在另一个终端、`tmux`、`screen`、notebook kernel 里已经先启动过 TPU 程序
4. 当前脚本又启动了新的独立进程，新的进程试图再次初始化 TPU runtime

### JAX / TPU 专业知识

这类报错的核心不是“JAX 坏了”，而是：

> **当前这块 TPU 已经被另一个独立进程先初始化并占用了。**

要区分两种情况：

#### 情况 A：两个互不协调的独立脚本抢同一块 TPU

这是最常见的错误用法。  
第一个进程已经把 TPU runtime 接管了，第二个独立进程再来初始化 TPU，就会被拒绝。

#### 情况 B：真正的 multi-process / multi-controller JAX

JAX 确实支持多进程分布式，但它不是“随便两个脚本都能同时连同一块 TPU”。  
多进程场景下需要多个 controller 进程协同运行，并通过 `jax.distributed.initialize()` 建立分布式环境；各进程通常运行同一套脚本，并按一致顺序执行 JAX 操作。  
这是一种 **协调式 distributed execution**，不是“多个不相干脚本随意共享一块 TPU”。

### 简单模拟例子

#### 进程 A

```python
import jax
import jax.numpy as jnp
x = jnp.ones((8192, 8192))
y = x @ x
y.block_until_ready()

input("holding TPU...")
```

#### 进程 B

```python
import jax
import jax.numpy as jnp
x = jnp.ones((4096, 4096))
```

如果进程 A 已经占住 TPU，进程 B 在初始化 TPU backend 时就可能报 “already in use by process ...”。

### 实用检查命令

```bash
ps -fp 22589
sudo lsof -w /dev/vfio/*
tpu-info
```

这几条命令分别用于：

- 看指定 PID 是什么进程
- 看 `/dev/vfio/*` 这种 TPU 设备文件被谁打开了
- 用 `tpu-info` 查看 TPU 芯片与对应 PID

---

## 5. 问题五：为什么有时看起来“代码很快就结束了”，但其实 TPU 可能还没真正算完？

### 问题描述

你可能写了下面这种代码：

```python
y = step(x)
print("done")
```

然后发现程序几乎瞬间打印 `done`。  
这时候很容易误以为：

> “TPU 已经把这一步算完了。”

### 可能原因

真正的原因通常是：

1. JAX 采用 **asynchronous dispatch**
2. Python 主线程只是把任务下发给设备
3. 设备端还在跑，但主线程已经继续执行后面的代码了

### JAX 专业知识

JAX 的 **asynchronous dispatch** 表示：  
Python 可以“跑在 accelerator 前面”，即主机侧代码继续执行，而 accelerator 侧工作仍在排队或执行中。  
因此，如果你想做**准确计时**、**准确判断是否执行完成**、**检查某一步是否真跑在 TPU 上**，通常都要显式调用：

```python
y.block_until_ready()
# 或者
jax.block_until_ready(y)
```

否则你测到的往往只是 **提交任务的时间**，不是 **设备真正执行完成的时间**。

### 简单模拟例子

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

一般来说：

- `dispatch time` 会更短
- `real execution time` 才更接近真实 TPU 计算时间

---

## 6. 问题六：`JAX_PLATFORMS` 是干什么的？为什么有时它会影响报错行为？

### 问题描述

有时 JAX 报错里会提示：

```text
set JAX_PLATFORMS='' to automatically choose an available backend
```

或者你可能自己设置过：

```bash
JAX_PLATFORMS=tpu
JAX_PLATFORMS=cpu,tpu
```

### 可能原因

这是因为 `JAX_PLATFORMS` 控制了：

1. JAX 要初始化哪些 backend
2. 哪个 backend 作为默认 backend
3. 某个 backend 初始化失败时，程序是直接报错还是改用别的 backend

### JAX 专业知识

`JAX_PLATFORMS` 是一个 **逗号分隔** 的 platform 名称列表。

例如：

```bash
JAX_PLATFORMS=cpu,tpu
```

表示初始化 CPU 和 TPU backend，并且默认 backend 是 CPU；如果 TPU 初始化失败，仍然会报异常。  
所以这个变量更像是一种 **强约束配置**，而不是“随便试试哪个能用”。

### 简单模拟例子

#### 强制只用 TPU

```bash
JAX_PLATFORMS=tpu python test.py
```

如果 TPU 初始化失败，程序会直接报错。

#### 让 JAX 自动选可用 backend

```bash
JAX_PLATFORMS='' python test.py
```

这通常更适合临时调试，因为它允许 JAX 自动选择当前可用 backend。

---

## 7. 问题七：怎么做“最小化、最可靠”的 TPU 判断？

### 推荐判断顺序

我更推荐按下面这套顺序检查：

### 第一步：确认 JAX 是否认到 TPU

```python
import jax

print("backend:", jax.default_backend())
print("devices:", jax.devices())
assert jax.devices()[0].platform == "tpu"
```

### 第二步：确认具体输出数组是否落在 TPU 上

```python
import jax
import jax.numpy as jnp

@jax.jit
def step(x):
    return x @ x

x = jnp.ones((4096, 4096))
y = step(x)
y.block_until_ready()

print("device:", y.device)
print("sharding:", y.sharding)
print("addressable shard devices:", [s.device for s in y.addressable_shards])
```

### 第三步：需要铁证时，做 profiler

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

JAX profiler 可以采集 CPU/GPU/TPU activity。

---

## 8. 一句话总结

### 可以把常见现象压缩成下面四句话

1. **`jax.devices()` 里有 `TpuDevice`**  
   说明 **JAX backend 已经接到 TPU**。

2. **输出数组的 `device / sharding / addressable_shards` 指向 TPU**  
   说明 **这一步结果确实落在 TPU 设备上**。

3. **不加 `block_until_ready()`，很多时候只代表任务已经提交，不代表 TPU 已经算完**  
   这是 **asynchronous dispatch** 的典型表现。

4. **`TPU is already in use by process ...`**  
   说明 **另一独立进程已经先占住了 TPU runtime**；这和真正受协调的 multi-process JAX 不是一回事。