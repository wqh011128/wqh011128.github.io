---
layout: post
title: "MoE 优化的探索"
date: 2026-05-10
categories:
  - blog
---

# **MoE 优化的探索：从 MiniMax-01、Comet、FlashMoE 到 DeepSeek-V4**

**Link:**

- MiniMax-01: [arXiv PDF](https://arxiv.org/pdf/2501.08313)
- Comet: [arXiv](https://arxiv.org/abs/2502.19811)
- FlashMoE: [Project Page](https://flash-moe.github.io/) / [arXiv](https://arxiv.org/abs/2506.04667)
- DeepSeek-V4: [Technical Report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf)

[toc]

------

## **Main Idea**

MoE 的核心收益来自 sparse computation：每个 token 只激活少数 experts，因此模型总参数可以很大，但单 token 计算量仍然可控。

问题是，MoE 在分布式训练和推理里会引入很重的通信。一个 token 被 router 分到某个 expert 后，那个 expert 可能在另一张 GPU 上，于是系统必须先把 token 发过去，再把 expert 的输出发回来。这个过程通常叫：

```text
Dispatch -> Expert Compute -> Combine
```

当模型规模变大时，瓶颈不再只是 expert 的矩阵乘，而是 **通信和计算之间的空转**。如果所有 GPU 都先等 `Dispatch All-to-All` 完成，再开始 expert GEMM，算完之后又一起等 `Combine All-to-All`，那么 GPU 会在通信阶段闲着，网络也会在计算阶段闲着。

MoE 优化的主线就是：

> 不要让通信和计算串行排队，而是把 token、expert、tile 或 kernel 切成更细粒度，让“正在算的一部分”和“正在传的一部分”重叠起来。

------

## **1. MoE 层到底在执行什么**

以常见的 Transformer MoE FFN 为例，一个 MoE 层可以分成四类操作。

| 阶段 | 做什么 | 主要瓶颈 |
|---|---|---|
| Router | 为每个 token 选择 top-k experts | 负载均衡、路由开销 |
| Dispatch | 把 token 发到 expert 所在设备 | all-to-all 通信 |
| Expert FFN | 每个 expert 对收到的 token 做 MLP | GEMM 计算 |
| Combine | 把 expert 输出发回原位置并加权合并 | all-to-all 通信 |

Expert FFN 本身通常又包含两个大 Linear：

```text
Linear-1: X_e -> hidden expansion
Activation: SiLU / SwiGLU
Linear-2: hidden expansion -> output
```

如果写成 SwiGLU 风格：

$$
\begin{aligned}
g_e &= X_e W_{e,\mathrm{gate}} \\
u_e &= X_e W_{e,\mathrm{up}} \\
z_e &= \mathrm{SiLU}(g_e) \odot u_e \\
Y_e &= z_e W_{e,\mathrm{down}}
\end{aligned}
$$

其中 $X_e$ 是分配给 expert $e$ 的 token 表示。`Linear-1` 可以理解为 $W_{e,\mathrm{gate}}$ 和 $W_{e,\mathrm{up}}$ 这一侧的 GEMM，`Linear-2` 是 $W_{e,\mathrm{down}}$ 的 GEMM。

这里的 **GEMM** 是 General Matrix Multiplication，也就是通用矩阵乘。GPU 对 GEMM 极度优化，所以 expert compute 的核心就是如何让这些 GEMM 一直吃满算力。

------

## **2. MiniMax-01：token group / process group 级别的 overlap**

MiniMax-01 的 MoE 优化重点不是把 expert 内部拆成 `Linear-1 / Linear-2` 的流水线，而是围绕 **Expert Parallelism (EP)**、**Expert Tensor Parallelism (ETP)** 和 **Expert Data Parallelism (EDP)** 做更合理的通信计算重叠。

### 2.1 Token-grouping-based overlap

MiniMax-01 先把 tokens 切成多个 group。每个 group 都要经历：

```text
a2a-dispatch -> expert compute -> a2a-combine
```

如果完全串行，时间线是：

```text
group 0: dispatch -> compute -> combine
group 1: dispatch -> compute -> combine
group 2: dispatch -> compute -> combine
```

MiniMax-01 希望改成：

```text
group 0: dispatch -> compute -> combine
group 1:          dispatch -> compute -> combine
group 2:                   dispatch -> compute -> combine
```

这样，某个 group 在做 expert compute 时，另一个 group 可以做 dispatch 或 combine。

这是一种 **token group 粒度** 的 overlap。它没有改变 expert FFN 的数学结构，也没有把单个 expert 的 GEMM 拆成 wave。它只是把 token batch 切小，让通信和计算不要在整层级别完全串行。

### 2.2 为什么还要 ETP / EDP

MiniMax-01 进一步指出，仅靠 EP 不一定够。当 expert 参数太大时，可以用 **Expert Tensor Parallelism (ETP)** 把单个 expert 的参数也切到多个设备上。

这时 MoE 层的流程会变成：

```text
a2a-dispatch -> allgather -> expert compute -> reduce-scatter -> a2a-combine
```

这里的 `allgather` 和 `reduce-scatter` 不是 DeepSeek Figure 5 里的 `Linear-1 / Linear-2`。它们来自 tensor parallelism：为了让多个设备共同计算一个 expert，需要先收集输入或中间张量，再把结果规约分散回去。

MiniMax-01 还引入 **Expert Data Parallelism (EDP)**，本质上是在 expert 维度做复制，缓解某些 expert 负载过高的问题。

### 2.3 MiniMax-01 的定位

MiniMax-01 更像是训练系统级别的 MoE 并行优化：

- 重点是 EP / ETP / EDP 的组合。
- overlap 粒度主要是 token group 和 process group。
- 它处理的是大规模训练时不同并行策略之间的通信压力。
- 它不是 Comet / FlashMoE / DeepSeek 那种更细的 expert GEMM pipeline 或 kernel-level 调度。

这也是最容易误解的地方：MiniMax-01 的“五段”并不等价于 DeepSeek-V4 的“五段”。

------

## **3. Comet：shared tensor / GEMM tile 级别的 overlap**

Comet 讨论的问题更细。它关注 MoE 中两个 producer-consumer pipeline：

```text
Dispatch communication -> Linear-1 GEMM
Linear-2 GEMM -> top-k reduction / Combine communication
```

Comet 的关键观察是：通信和计算的粒度不一致。

- 通信通常以 token 或 token block 为单位。
- GEMM 通常以 tile 为单位，例如 $128 \times 128$。
- 如果系统必须等整个 dispatch 完成再做 GEMM，就会浪费已经到达的 token。

### 3.1 Shared tensor 是什么

Comet 把通信和 GEMM 之间共享的中间张量叫 **shared tensor**。例如 dispatch 后的 token buffer，会被 `Linear-1` 消费；`Linear-2` 的输出 buffer，又会被 combine 或 top-k reduction 消费。

Comet 的优化可以概括为两步：

```text
1. Decompose shared tensor
2. Reschedule decomposed tensor
```

也就是先把 shared tensor 拆成更小的单元，再重新安排这些单元被通信和 GEMM 消费的顺序。

### 3.2 Linear-1 前的优化

在 `Dispatch -> Linear-1` 这条链路上，Comet 沿 token 维度切 shared tensor。某些 token block 已经到达时，GEMM 就可以先处理这些 token block，不必等待所有远程 token 全部到齐。

直觉时间线是：

```text
dispatch tile 0 done -> Linear-1 consumes tile 0
dispatch tile 1 done -> Linear-1 consumes tile 1
dispatch tile 2 done -> Linear-1 consumes tile 2
```

这比整层 barrier 更细。

### 3.3 Linear-2 后的优化

在 `Linear-2 -> Combine` 这条链路上，Comet 需要解决另一个问题：GEMM 产生的输出如果必须等全部算完，combine 仍然会被拖住。

所以 Comet 会重排 GroupGEMM 的计算顺序，让一部分输出 tile 先被产生、先进入 reduction 或 combine。它的目标不是改变最终结果，而是改变“什么时候产生哪一块结果”。

### 3.4 Thread block specialization

Comet 还做了 thread block specialization：一部分 thread blocks 专门负责计算，一部分 thread blocks 专门负责通信。原因是通信和 GEMM 对 GPU 资源的需求不同，如果同一个 block 什么都做，容易互相干扰。

因此 Comet 会预编译多个 kernel 版本，比如不同数量的 communication blocks，然后运行时根据 profile 选择更合适的配置。

### 3.5 Comet 的定位

Comet 的核心不是“把 MoE 拆成几个阶段”这么简单，而是：

> 找到 MoE 中通信和 GEMM 之间共享的 tensor，拆开它，并重排每一块 tensor 的执行顺序。

它比 MiniMax-01 更细，因为它已经进入 GEMM tile / thread block 级别；但它还没有像 FlashMoE 那样把整个 MoE operator 放进一个 persistent kernel 里。

------

## **4. FlashMoE：persistent kernel + GPU-resident scheduling**

FlashMoE 的目标更激进：把 distributed MoE operator 尽量放进一个 persistent kernel 里执行。

传统 MoE 通常会触发一串 kernel 和 collective：

```text
router kernel
dispatch communication kernel
expert GEMM kernel
activation kernel
expert GEMM kernel
combine communication kernel
```

这会带来两个问题：

- kernel launch 多，CPU 调度和 GPU 同步开销大。
- All-to-All 是 bulk-synchronous collective，容易被慢设备或负载不均拖住。

FlashMoE 的思路是：让 GPU 内部长期驻留一个 kernel，由它自己调度通信和计算任务。

### 4.1 Actor model

FlashMoE 把执行角色抽象成类似 actor 的组件：

| 角色 | 作用 |
|---|---|
| Subscriber | 接收来自其他 GPU 的 tile/task message |
| Scheduler | 根据 ready 状态决定下一步执行什么 |
| Processor | 执行 GEMM、elementwise、combine、tile communication |

这样一来，MoE 不再只是 CPU 发起一串 kernel，而是 GPU kernel 内部自己做调度。

### 4.2 One-sided communication

FlashMoE 强调 one-sided、device-initiated communication。直觉上，它不是等 CPU 或 collective runtime 统一安排通信，而是 GPU kernel 内部的线程主动读写远端 buffer。

这能减少 collective barrier，让通信更接近 fine-grained message passing。

### 4.3 Symmetric layout 和 temporal buffering

如果多个 GPU 同时读写远端 buffer，会出现并发冲突。FlashMoE 通过 symmetric tensor layout 和 temporal buffering 管理这些读写位置，让不同 tile 的数据可以安全地在不同时间段进入 buffer。

另外，它还使用 in-place padding，避免把 padding token 当成真实 token 通过网络传输。

### 4.4 FlashMoE 的定位

FlashMoE 可以理解成：

> 把 MoE 层看成一堆 tile-level tasks，然后在一个 persistent GPU kernel 里做通信、计算和调度。

它比 Comet 更系统级、更激进。Comet 仍然围绕 shared tensor decomposition 和 kernel scheduling 展开；FlashMoE 则试图把 MoE 的执行模型整体改成 GPU-resident task runtime。

------

## **5. DeepSeek-V4：expert-wave 级别的 MoE pipeline**

DeepSeek-V4 的 MoE 优化小节叫 **Fine-Grained Communication-Computation Overlap in Expert Parallelism**。它借鉴了前面这些工作，但落点更工程化：围绕 DeepSeek-V4 自己的 expert parallelism，把 MoE 层拆成可以流水的 expert waves。

### 5.1 DeepSeek 的五段流程

DeepSeek-V4 把 MoE 层执行写成：

```text
Dispatch All-to-All
-> Linear-1 GEMM
-> SwiGLU / FP8 Cast
-> Linear-2 GEMM
-> Combine All-to-All
```

这里的 `Linear-1` 和 `Linear-2` 是 expert FFN 的两个自然 GEMM，不是额外加出来的新层。

### 5.2 什么是 wave

一个 wave 可以理解成一小批 experts。例如某个 MoE 层有很多 experts，不必等所有 experts 的 dispatch 都完成后再一起计算，而是：

```text
wave 0 的 token 到了 -> 先算 wave 0 的 experts
wave 1 的 token 正在传 -> 传完后接着算 wave 1
wave 0 算完后 -> 立刻 combine wave 0 的输出
```

理想化时间线：

```text
t0: dispatch wave 0
t1: compute  wave 0 | dispatch wave 1
t2: combine  wave 0 | compute wave 1 | dispatch wave 2
t3: combine  wave 1 | compute wave 2 | dispatch wave 3
```

也就是说，DeepSeek-V4 的切分粒度是 **expert wave**。它既不是 MiniMax-01 的 token group，也不是 Comet 的 shared tensor tile，也不是 FlashMoE 的完整 persistent task runtime。

### 5.3 为什么它能隐藏通信

如果计算足够重，通信就可以被计算盖住。DeepSeek-V4 给出的判断可以理解成：

$$
\frac{C}{B} \leq \frac{V_{\mathrm{comp}}}{V_{\mathrm{comm}}}
$$

其中：

- $C$：设备峰值计算能力。
- $B$：通信带宽。
- $V_{\mathrm{comp}}$：MoE 层计算量。
- $V_{\mathrm{comm}}$：MoE 层通信量。

如果右边足够大，说明每传一点数据都能对应很多计算，那么通信更容易被计算隐藏。DeepSeek-V4-Pro 里每个 token-expert pair 大约需要 $6hd$ FLOPs，通信量大约是 $3h$ bytes，所以比例大约是：

$$
\frac{V_{\mathrm{comp}}}{V_{\mathrm{comm}}} \approx 2d
$$

当 $d=3072$ 时，就是约 $6144$ FLOPs/Byte。这说明在合适硬件条件下，MoE expert 计算可以覆盖很大一部分通信。

### 5.4 DeepSeek 的定位

DeepSeek-V4 不是在论文里提出一个通用 MoE runtime 系统，而是在自己的大模型系统里落地了一个适合 large-scale EP 的 expert-wave pipeline。

它的重点是：

- 把 expert computation 切成 waves。
- 让 dispatch、expert GEMM、combine 同时推进。
- 尽量让 all-to-all 通信隐藏在 GEMM 计算后面。
- 结合 FP8/FP4、DeepGEMM 等底层优化服务高吞吐训练和推理。

------

## **6. 四篇论文的粒度对比**

最重要的是不要把所有“overlap”都理解成同一种切法。

| 工作 | 切分粒度 | 主要对象 | 解决的问题 | 一句话 |
|---|---|---|---|---|
| MiniMax-01 | token group / process group | EP、ETP、EDP 通信 | 大规模训练中的 a2a、allgather、reduce-scatter 开销 | group-level overlap |
| Comet | shared tensor / GEMM tile / thread block | Dispatch-L1、L2-Combine 的 producer-consumer 链路 | 通信粒度和 GEMM tile 粒度不匹配 | tile-level dependency-aware overlap |
| FlashMoE | tile task / actor / persistent kernel | 整个 distributed MoE operator | collective barrier 和 kernel launch 开销 | GPU-resident task scheduling |
| DeepSeek-V4 | expert wave | Dispatch、Linear-1、Activation、Linear-2、Combine | expert parallelism 下通信阻塞计算 | expert-wave pipeline |

再把它们放到一条“越来越细”的轴上：

```text
MiniMax-01
  token group / process group
    ->
Comet
  shared tensor / GEMM tile / thread block
    ->
DeepSeek-V4
  expert wave pipeline for production EP
    ->
FlashMoE
  persistent kernel + GPU-resident task runtime
```

这条线不是严格的优劣排序，而是抽象层级不同。

------

## **7. MiniMax 和 DeepSeek 的“五段”为什么不是一回事**

这是最容易混淆的点。

MiniMax-01 在 ETP 场景下的流程是：

```text
a2a-dispatch -> allgather -> expert compute -> reduce-scatter -> a2a-combine
```

这里的 `allgather` 和 `reduce-scatter` 是 tensor parallelism 引入的通信操作。

DeepSeek-V4 Figure 5 的流程是：

```text
dispatch -> Linear-1 -> activation -> Linear-2 -> combine
```

这里的 `Linear-1` 和 `Linear-2` 是 expert FFN 内部的两个 GEMM。

所以二者的“五段”不是同一种拆分：

- MiniMax 拆的是 **并行通信路径**。
- DeepSeek 拆的是 **expert FFN 执行路径**。
- Comet 拆的是 **通信和 GEMM 之间的 shared tensor**。
- FlashMoE 拆的是 **整个 MoE operator 的 tile tasks**。

如果用一句话区分：

```text
MiniMax 关心怎么组织并行组；
Comet 关心怎么重排共享张量；
FlashMoE 关心怎么把 MoE 变成 GPU 常驻任务系统；
DeepSeek 关心怎么把 experts 分 wave 后流水执行。
```

------

## **8. 一个统一的理解框架**

MoE 优化可以按三层看。

### 8.1 并行策略层

这一层决定 expert 放在哪些 GPU 上，token 怎么发过去。

典型问题：

- expert 数量如何映射到 GPU？
- top-k 后 token 分布不均怎么办？
- EP、ETP、EDP 怎么组合？
- all-to-all 和 allgather/reduce-scatter 怎么重叠？

MiniMax-01 主要在这一层。

### 8.2 Kernel / GEMM 调度层

这一层决定矩阵乘和通信怎么交错。

典型问题：

- token 到了一部分，GEMM 能不能先算？
- `Linear-2` 算出一部分，combine 能不能先发？
- 哪些 thread block 做通信，哪些做计算？
- group GEMM 的 tile 顺序怎么排？

Comet 和 DeepSeek-V4 主要在这一层，只是切分对象不同。

### 8.3 Runtime 层

这一层决定整个 MoE operator 是否还依赖外部 collective 和一串 kernel launch。

典型问题：

- 能不能一个 persistent kernel 管完整个 MoE？
- GPU 能不能自己发起远端读写？
- task ready 之后能不能马上调度？
- 如何避免 barrier 和 straggler？

FlashMoE 主要在这一层。

------

## **9. 总结**

MoE 优化不是一个单点技巧，而是一组围绕“通信、计算、调度”展开的系统工程。

MiniMax-01 说明，当模型达到 456B 参数、长上下文训练又引入复杂并行时，MoE 的关键是把 EP、ETP、EDP 组织好，并通过 token group 让通信和计算重叠。

Comet 说明，MoE 的低效来自通信和 GEMM 粒度不一致。只要把 shared tensor 拆开，并重新安排 tile 的消费顺序，就能减少等待。

FlashMoE 说明，传统 collective 和多 kernel launch 本身就是瓶颈。把 MoE 做成 persistent kernel，并让 GPU 自己调度通信和计算，可以进一步压低系统开销。

DeepSeek-V4 则说明，在真实超大规模 MoE 模型里，expert-wave pipeline 是一种实用折中：它不必重写整个 runtime，但能把 dispatch、expert GEMM 和 combine 更细地流水起来。

所以我现在会这样记：

```text
MiniMax-01: group-level overlap
Comet: tile-level dependency-aware overlap
FlashMoE: persistent-kernel task scheduling
DeepSeek-V4: expert-wave pipeline
```

这四篇放在一起看，真正讲的是同一个问题的不同层级：**MoE 让参数变稀疏，但通信让系统变复杂；优化的本质，是让每一块数据到达后尽快被计算，每一块结果产生后尽快被发送。**
