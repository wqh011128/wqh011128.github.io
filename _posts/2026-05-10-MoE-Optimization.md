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

## **1. MoE 基本流程**

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

在MiniMax-01的总结中解读了EP，EP overlap，以及MiniMax的MoE创新。其优化重点不是把 expert 内部拆成 `Linear-1 / Linear-2` 的流水线，而是围绕 **Expert Parallelism (EP)**、**Expert Tensor Parallelism (ETP)** 和 **Expert Data Parallelism (EDP)** 做更合理的通信计算重叠。

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

这是一种 **token group 粒度** 的 overlap。它没有改变 expert FFN 的数学结构，也没有把单个 expert 的 GEMM 拆成 wave（像DeepSeek v4那样）。它只是把 token batch 切小，让通信和计算不要在整层级别完全串行。

### 2.2 为什么还要 ETP / EDP

MiniMax-01 进一步指出，仅靠 EP 不一定够。当 expert 参数太大时，可以用 **Expert Tensor Parallelism (ETP)** 把单个 expert 的参数也切到多个设备上。

这时 MoE 层的流程会变成：

```text
a2a-dispatch -> allgather -> expert compute -> reduce-scatter -> a2a-combine
```

这里的 `allgather` 和 `reduce-scatter` 来自 tensor parallelism：为了让多个设备共同计算一个 expert，需要先收集输入或中间张量，再把结果规约分散回去。

MiniMax-01 还引入 **Expert Data Parallelism (EDP)**，本质上是在 expert 维度做复制，缓解某些 expert 负载过高的问题。

### 2.3 MiniMax-01 的定位

MiniMax-01 更像是训练系统级别的 MoE 并行优化：

- 重点是 EP / ETP / EDP 的组合。
- overlap 粒度主要是 token group 和 process group。
- 它处理的是大规模训练时不同并行策略之间的通信压力。
- 它不是 Comet / FlashMoE / DeepSeek 那种更细的 expert GEMM pipeline 或 kernel-level 调度。

------

## **3. Comet：fine-grained computation-communication overlapping**

这一节开始单独精读 **Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts**。这篇文章可以看成是在回答一个很具体的问题：

> 传统 EP overlap 已经把 token batch 切成 chunk 了，为什么 MoE 层里还是有明显 GPU idle time？如果继续往下切，应该切什么，怎么切，谁先算，谁先传？

Comet 的答案不是简单地“再把 chunk 切小一点”。它真正做的是：找到 MoE layer 中通信算子和计算算子之间共享的 buffer，也就是 **shared tensor**，然后根据 consumer 的依赖关系决定沿哪个维度切分，并重新安排 GroupGEMM 的 tile 执行顺序。

### 3.1 传统 EP overlap 为什么还是 coarse-grained

Comet 在 Introduction 里先分析了传统做法。一个分布式 MoE 层通常可以抽象成：

```text
Receive / Dispatch
-> Expert computation
-> Send / Combine
```

如果不 overlap，就是先收完所有 token，再做 expert GEMM，最后再发回结果。传统 EP overlap 会把 expert computation kernel 切成几个 chunk，让某个 chunk 的计算和另一个 chunk 的通信同时发生。

![Comet Figure 1: MoE execution analysis](https://arxiv.org/html/2502.19811/x1.png)

Figure 1(b) 想表达的是：把输入拆成 chunk 后，确实可以让一部分通信和一部分计算重叠，但这种 overlap 仍然是 **coarse-grained**。原因有三个。

第一，chunk 仍然必须作为一个整体 ready。即使 chunk 已经比完整 batch 小，一个 chunk 内部仍然可能有很多 token；只要这个 chunk 需要的 token 还没全部到齐，expert computation 就不能启动。

第二，chunk 变小会伤害 GEMM 效率。论文里提到，原本完整 expert computation 的时间是 $t$，切成两个 chunk 后可能变成 $t_1+t_2>t$。这不是数学计算量变多了，而是小 GEMM 更难吃满 tensor core，还会带来更多调度和访存开销。

第三，MoE 是动态的。这里的动态不是模型结构变了，而是 router 每次会把不同 token 分给不同 experts。某个 step 里 Expert0 可能收到很多 token，Expert1 很少；下个 step 又可能反过来。于是每个 expert 的输入形状、通信量、计算量都在运行时变化，导致“通信 chunk”和“计算 chunk”的时间很难稳定对齐。

所以传统 EP overlap 的核心问题是：它把通信和计算装进不同 kernel / stream 里，让它们粗粒度并行，但对 GPU thread blocks、GEMM tile 顺序、remote I/O 这些底层资源缺少精细控制。

### 3.2 MoE Structure：论文里的 forward 过程

Comet 的 Figure 2 给了一个普通 MoE layer 的执行例子：两个 GPU，总共四个 experts，GPU0 放 Expert0/Expert1，GPU1 放 Expert2/Expert3。每个 token 被 router 分到 $k$ 个 experts。图里 Token A 被路由到 Expert0、Expert1、Expert3。

![Comet Figure 2: MoE layer across two GPUs](https://arxiv.org/html/2502.19811/x2.png)

论文中几个重要符号可以这样读：

| 符号 | 含义 |
|---|---|
| $E$ | expert 总数 |
| $k$ | 每个 token 被路由到的 expert 数量，也就是 top-k |
| $TP$ | tensor parallel size |
| $EP$ | expert parallel size |
| $TP \times EP$ | 总并行 world size |
| $M$ | GEMM 中的 row 维度，在 MoE 里通常对应 token / token-expert rows |
| $K$ | GEMM reduction 维度，例如输入 hidden 或 FFN intermediate 维 |
| $N$ | GEMM output column 维度，例如输出 hidden 或 FFN intermediate 维 |
| $T_M$ | GEMM tile 在 $M$ 维的大小 |
| $T_N$ | GEMM tile 在 $N$ 维的大小 |

MoE 的每个 expert FFN 有两层 GEMM：

```text
layer0: first expert GEMM, usually up/gate projection
activation: SiLU / SwiGLU
layer1: second expert GEMM, usually down projection
```

论文把 MoE 的执行过程分成两类 pipeline。

**communication-computation pipeline:** 对应 MoE layer0。

```text
Dispatch communication -> layer0 GroupGEMM
```

这里通信是 producer，layer0 GroupGEMM 是 consumer。shared tensor 是 dispatch 之后、即将被 layer0 GEMM 消费的 expert input buffer。

**computation-communication pipeline:** 对应 MoE layer1。

```text
layer1 GroupGEMM -> top-k routed reduction / combine communication
```

这里 layer1 GroupGEMM 是 producer，reduction/combine 是 consumer。shared tensor 是 layer1 GEMM 产生、即将被合并和通信的 expert output buffer。

### 3.3 TP 是什么，为什么 MoE 会用

**Tensor Parallelism (TP)** 是把同一个线性层的权重切到多个 GPU 上。它和 EP 的区别是：

```text
EP: 不同 experts 放在不同 GPU 上，每个 expert 权重通常是完整的。
TP: 同一个 expert / linear 的权重沿 hidden dimension 切开，多张 GPU 一起算一个矩阵乘。
```

比如一个线性层：

$$
Y = XW,\quad W\in\mathbb{R}^{K\times N}
$$

如果按 $N$ 维做 column parallel，GPU0 负责 $W[:,0:N/2]$，GPU1 负责 $W[:,N/2:N]$。如果按 $K$ 维做 row parallel，多个 GPU 分别计算部分乘积，再做 reduce。

MoE 会用 TP，是因为单个 expert 的 FFN 也可能很大。只用 EP 时，每个 expert 权重完整放在某张 GPU 上；如果 expert 太大，或者希望提升单 expert GEMM 的吞吐，就需要 TP 继续切 expert 内部权重。实际大模型里常常是 **EP + TP 混合并行**。

### 3.4 Granularity mismatch：token-level communication vs tile-level computation

Comet 的第一个关键观察是 **granularity mismatch between computation and communication**。

在 MoE 里，通信的基本单位通常是 token。Router 决定某个 token 要去哪个 expert，于是系统把这个 token 发到 expert 所在 GPU。

但高性能 GEMM 的基本单位不是单 token，而是 tile。论文里 Figure 2 的紫色块就是一个 computation tile，例如 $128\times128$。这意味着一个 expert 的某个 GEMM tile 可能需要 128 个 token rows，而这些 token 由 router 决定，可能随机分布在多个 GPU 上。

这就产生了依赖：

```text
一个 GEMM tile 需要的 token rows 没有全部 ready
-> 这个 tile 不能开始计算
```

所以问题不是“tile 大小不一样导致完成时间不一样”，而是：

> 每个 tile 依赖的 token 来源不同，ready 时间不同；coarse-grained dispatch 会让 tile 等整个 chunk 或整个 expert input buffer。

Comet 因此提出 fine-grained communication：每个 computation tile 通过 **Unified Virtual Address (UVA)** 直接读/写它需要的数据。

UVA 的实际作用是提供统一虚拟地址空间。在支持 GPU peer access 的情况下，GPU kernel 可以拿到远端 GPU buffer 的地址，并发起细粒度 remote load/store。它不是让 tile “自己有智能”，而是让 kernel 里的 communication thread blocks 可以根据路由 metadata，把某个 tile 需要的 remote token rows 拉到本地，或者把某个输出 tile 写回目标位置。

但 fine-grained remote I/O 很慢。如果把远程读写塞进 GEMM compute thread block，会破坏 tensor core pipeline。Comet 后面才需要 thread block specialization：通信 block 专门做 remote I/O，计算 block 保持高效 GEMM。

### 3.5 Design overview：shared tensor 是桥

Comet 的 Figure 3 是整篇文章的设计总览。

![Comet Figure 3: Design overview](https://arxiv.org/html/2502.19811/x3.png)

Comet 有两个核心设计：

| 机制 | 解决什么 |
|---|---|
| Shared tensor based dependency resolving | 分析 producer/consumer 之间的真实数据依赖，决定 shared tensor 沿哪个维度切，并重排 tile 顺序 |
| Adaptive workload assignment | 在 fused kernel 内动态分配 thread blocks 给通信和计算，减少 pipeline bubble |

这里的 **shared tensor** 可以简单理解成：producer 和 consumer 共用的那块中间 buffer。

对 layer0：

```text
producer: dispatch communication
shared tensor: expert input X_e
consumer: layer0 GroupGEMM
```

对 layer1：

```text
producer: layer1 GroupGEMM
shared tensor: expert output Y_e
consumer: top-k routed reduction + combine communication
```

shared tensor 重要，是因为 overlap 只有在 producer 和 consumer 能处理 shared tensor 的不同独立部分时才成立。如果 consumer 必须等完整 tensor，overlap 就退化成普通串行。

### 3.6 3.1.1：How to decompose the shared tensor

Figure 4 把 layer0 和 layer1 都建模成 producer-consumer 关系。

![Comet Figure 4: Producer-consumer modeling](https://arxiv.org/html/2502.19811/x4.png)

Comet 的原则是：

> 沿 consumer 视角下相互独立的维度切 shared tensor。

#### 3.6.1 Layer0 为什么沿 $M$ 切

Layer0 的 shared tensor 是 layer0 GEMM 的输入矩阵：

$$
X_e \in \mathbb{R}^{M_e\times K}
$$

其中 $M_e$ 是 expert $e$ 收到的 token rows 数量，$K$ 是 token embedding / hidden dimension。

Layer0 的 consumer 是 GEMM：

$$
H_e = X_e W_{e,0}
$$

对 GEMM 来说，不同 token rows 之间相互独立。也就是说，先算 $X_e[M_0,:]$ 和后算 $X_e[M_1,:]$ 不会改变结果。因此 layer0 可以沿 $M$ 维切：

```text
X_e[M0, :] -> layer0 GroupGEMM
X_e[M1, :] -> layer0 GroupGEMM
X_e[M2, :] -> layer0 GroupGEMM
```

但不能沿 $K$ 维随便切，因为 GEMM 对 $K$ 维做 reduction。算一个输出元素需要完整的 $K$ 维乘加：

$$
H_{e,i,n}=\sum_{k}X_{e,i,k}W_{e,0,k,n}
$$

如果切 $K$，不同分块之间还要额外做 partial sum reduction，consumer 不能直接独立消费。

#### 3.6.2 Layer1 为什么沿 $N$ 切

Layer1 的 shared tensor 是 layer1 GEMM 的输出：

$$
Y_e \in \mathbb{R}^{M_e\times N}
$$

Layer1 后面的 consumer 不是普通逐元素操作，而是 **top-k routed reduction + combine**。注意，这里的 top-k reduction 不是重新选择 top-k；top-k 在 router 阶段已经完成了。这里的意思是：对同一个原始 token 的多个 expert 输出按 router weight 做加权合并。

例如 top-2：

$$
O_t = w_{t,e_1}Y_{t,e_1}+w_{t,e_2}Y_{t,e_2}
$$

如果沿 $M$ 切，可能把同一个 token 的两个 expert 输出拆到不同块：

```text
M tile 0: token A from Expert0
M tile 1: token A from Expert3
```

这时 consumer 处理 `M tile 0` 时拿不到 token A 的完整 top-k routed outputs，因此 $M$ 维存在 interdependency。

但 $N$ 维是 output feature column。不同 feature columns 的 weighted reduction 相互独立：

$$
O_{t,n}=w_{t,e_1}Y_{t,e_1,n}+w_{t,e_2}Y_{t,e_2,n}
$$

所以 layer1 可以沿 $N$ 切：

```text
Y[:, N0] -> reduction + combine
Y[:, N1] -> reduction + combine
Y[:, N2] -> reduction + combine
```

这就是论文中“layer0 沿 $M$ 分解，layer1 沿 $N$ 分解”的根本原因。

### 3.7 3.1.2：How to reschedule the decomposed shared tensor

只知道沿哪个维度切还不够。Comet 还要决定切完之后怎么排执行顺序。论文给了两个原则：

1. sub-tensors 要尽量对齐原始 GEMM tile granularity，否则 GEMM 效率会下降。
2. 优先执行 producer 已经产出、consumer 可以立即使用的部分，让 consumer 尽早启动。

#### 3.7.1 Layer0：按 $M$ 切后，先算 local-token tiles

Layer0 是：

```text
Dispatch -> layer0 GroupGEMM
```

Figure 5 画的是 Rank0 上有三个 experts，每个 expert 都需要 local data 和 remote data。

![Comet Figure 5: Decompose and reschedule layer0 shared tensor](https://arxiv.org/html/2502.19811/x5.png)

Comet 会先按 source rank 对 token 排序。直觉上：

```text
local tokens | remote rank 1 tokens | remote rank 2 tokens | ...
```

然后 GroupGEMM 的 tile compute sequence 会优先从 local tokens 所在 tile 开始。这样本地 tile 可以马上计算，同时远程 token 还在通过 communication blocks 传输。

时间线可以理解成：

```text
t0: compute local-token tiles
    communicate remote-rank-1 token rows

t1: compute remote-rank-1 tiles
    communicate remote-rank-2 token rows

t2: compute remote-rank-2 tiles
```

这里的 tile 通常类似 $T_M\times T_N$，例如论文前文举的 $128\times128$。但要注意，这个 tile 是 **某个 expert GEMM 内部的 tile**，不是把不同 experts 的 token 混在一起乘同一个权重。

#### 3.7.2 关键疑问：不同 experts 权重不同，$128\times128$ tile 怎么算

这是理解 GroupGEMM 的关键。

GroupGEMM 不是把所有 experts 的 token 拼成一个大矩阵，然后乘同一个权重。它是一组独立 GEMM 的调度：

$$
H_e = X_e W_{e,0},\quad e\in\mathcal{E}_{\mathrm{local}}
$$

也就是说，Rank0 上如果有 Expert0、Expert1、Expert2，那么 GroupGEMM 实际上在执行：

```text
Expert0: X_0 @ W_0
Expert1: X_1 @ W_1
Expert2: X_2 @ W_2
```

每个 computation tile 都带着 expert id。属于 Expert0 的 tile 只会用 $W_0$，属于 Expert1 的 tile 只会用 $W_1$。高性能 grouped GEMM kernel 会通过 metadata / pointer arrays / offsets 找到每个 expert 对应的 input pointer、weight pointer 和 output pointer。

所以 Figure 5 中的 $M$ 切分不是：

```text
把不同 expert 的 128 行混成一个 tile，用同一个 W 去乘
```

而是：

```text
在每个 expert 自己的 X_e[M_e, K] 里按 M tile 切；
GroupGEMM 只是把多个 expert 的 tile 放到同一个 kernel 里统一调度。
```

如果某个 expert 收到的 token 不足一个完整 $T_M$，实现上可以用 partial tile、padding 或 grouped GEMM 的 ragged shape metadata 处理。数学上仍然是每个 expert 使用自己的权重。

这也解释了为什么 layer0 按 $M$ 切完不需要“还原成原始 token 顺序”再进入 layer1。Layer0 输出仍然按 expert 分组保存：

$$
H_e[M_e, K']
$$

中间 activation 是逐元素的，不需要跨 expert 或跨 token 重新排列。Layer1 继续对同一个 expert 的 $H_e$ 做：

$$
Y_e = H_e W_{e,1}
$$

真正需要恢复到原始 token 顺序，是 layer1 结束后的 routed reduction / combine 阶段。

#### 3.7.3 Layer1：按 $N$ 切后，column-wise 执行 GroupGEMM

Layer1 是：

```text
layer1 GroupGEMM -> reduction + combine communication
```

Figure 6 说明了 Comet 如何重排 layer1 的 GroupGEMM。

![Comet Figure 6: Rescheduled compute sequence for layer1](https://arxiv.org/html/2502.19811/x6.png)

如果不重排，GroupGEMM 可能按 expert 顺序执行：

```text
Expert0: N0 -> N1 -> N2 -> N3
Expert1: N0 -> N1 -> N2 -> N3
Expert2: N0 -> N1 -> N2 -> N3
```

这样 consumer 很难提前开始，因为它想处理某个 column block 时，需要相关 experts 的同一段 columns 都已经产生。

Comet 改成 column-wise：

```text
N0 group:
  Expert0:N0 -> Expert1:N0 -> Expert2:N0

N1 group:
  Expert0:N1 -> Expert1:N1 -> Expert2:N1

N2 group:
  Expert0:N2 -> Expert1:N2 -> Expert2:N2
```

注意，这里的 `N0` 不是“对 128 列做 top-k selection”。Top-k selection 已经在 router 完成。这里做的是：对已经确定的 top-k experts，在 `N0` 这一段 output features 上做 weighted reduction 和 combine。

如果 $N_0$ 表示 columns $0:T_N$，那么 consumer 可以先做：

$$
O_{t,N_0}=\sum_{e\in\mathrm{TopK}(t)}w_{t,e}Y_{t,e,N_0}
$$

同时 layer1 GroupGEMM 继续计算 $Y[:,N_1]$。这样就形成：

```text
compute Y[:, N0]
-> reduce/combine Y[:, N0]

while

compute Y[:, N1]
```

所以 Figure 6 的核心不是“列维度上重新选 top-k”，而是：按 column block 提前产出可被 consumer 完整处理的一段 output features。

### 3.8 3.2：Adaptive Workload Assignment

经过 3.1 的 dependency resolving 后，Comet 已经知道哪些数据可以先算、哪些数据可以先传。但还有一个问题：fine-grained remote I/O 很慢，GEMM 又很吃 tensor core。谁来做通信，谁来做计算，分配多少 GPU 资源，不能拍脑袋。

Figure 7 展示的是 Comet 在 Hopper 上的 fused kernel 设计。

![Comet Figure 7: Thread block specialized kernel](https://arxiv.org/html/2502.19811/x7.png)

#### 3.8.1 Thread block specialization

最直接的融合方式叫 vertical fusion：每个 thread block 既做 GEMM，也在 prologue / epilogue 里做通信 I/O。

问题是 remote I/O 延迟远高于本地显存访问。如果把 remote read/write 插进 GEMM thread block，可能会阻塞后续 tensor core 计算，尤其 Hopper 上 GEMM 通常利用 TMA 建立异步 compute pipeline，长延迟 remote I/O 会破坏这个 pipeline。

Comet 因此把 thread blocks 隔离成两类：

```text
compute thread blocks: 负责 GEMM，尽量复用默认 CUTLASS GEMM 实现
communication thread blocks: 负责 remote I/O、top-k routed reduction、local/remote writeback
```

这样做的代价是会多一些 global memory 读写，但收益是通信不会污染 GEMM 的关键路径，而且系统可以精确控制多少 blocks 做通信、多少 blocks 做计算。

#### 3.8.2 Adaptive thread block assignment

论文 3.2.2 解决的是：通信 block 和计算 block 到底分多少？

假设一个 fused kernel 总共有 $N_{\mathrm{TB}}$ 个 thread blocks，其中：

```text
N_{\mathrm{comp}}: compute blocks
N_{\mathrm{comm}}: communication blocks
N_{\mathrm{TB}} = N_{\mathrm{comp}} + N_{\mathrm{comm}}
```

如果 $N_{\mathrm{comm}}$ 太少，远程 I/O 跟不上，GEMM 算完后会等通信。
如果 $N_{\mathrm{comm}}$ 太多，GEMM blocks 变少，计算吞吐下降。
最佳分界点和输入 token length、TP/EP 配置、expert shape、硬件带宽都有关。

Figure 8 说明不同配置下最优 $N_{\mathrm{comm}}$ 不同。

![Comet Figure 8: Adaptive thread block assignment](https://arxiv.org/html/2502.19811/x8.png)

论文给的例子是：当输入 token length 从 4096 变到 16384 时，最优通信 block 数会变化；当 TP 从 8 调到 4 时，最优分配点也会明显变化。

所以 Comet 的做法不是运行时在线搜索，而是：

```text
1. 预编译多个 kernel，每个 kernel 使用不同的 compute/communication block division point。
2. 部署前 profile 不同模型配置和输入形状，记录最优配置 metadata。
3. 运行时根据 metadata 选择合适 kernel。
```

这就是 adaptive workload assignment。它的目标不是改变数学计算，而是让 fine-grained pipeline 的通信段和计算段时间尽量对齐，减少 pipeline bubbles。

### 3.9 把我们讨论过的几个易错点放在一起

**Tile 不是 token chunk。** token chunk 是 EP overlap 的 coarse 粒度；tile 是 GEMM kernel 的计算分块，例如 $128\times128$。一个 tile 通常包含多个 token rows 和一段 output columns。

**Tile 不是 expert 的最大容量。** 一个 expert 实际收到多少 token 由 router 决定，记作 $M_e$。GEMM tile 是在 $M_e\times K$ 或 $M_e\times N$ 这个矩阵内部继续切出来的计算块。

**Layer0 的 $M$ 切分发生在 layer0 计算前。** 它切的是 dispatch 后的 expert input $X_e$，目的是 token rows 到一块、算一块。

**Layer1 的 $N$ 切分发生在 layer1 计算过程中。** 它不是先完整算出 $Y_e$ 再切，而是调整 GroupGEMM 顺序，直接先产出 $Y[:,N_0]$，让 reduction/combine 提前消费。

**Layer0 到 layer1 中间不需要还原成原始 token 顺序。** layer0 输出仍然按 expert 分组，activation 和 layer1 都可以在 expert-local layout 里继续做。只有最终 combine 时才需要根据 routing metadata 回到原 token 位置。

**UVA 不等于免费远程访问。** UVA 只是让 kernel 可以用统一地址访问远端 GPU buffer；真正的通信仍然有高延迟，所以 Comet 才要用 communication thread blocks 隔离远程 I/O。

### 3.10 Comet 的定位

Comet 的核心不是“把 MoE 拆成几个阶段”，而是：

> 找到 MoE 中通信和 GEMM 之间共享的 tensor，分析 consumer 在哪个维度上可以独立消费，再按这个维度切分并重排 GroupGEMM tile 顺序，最后用 adaptive thread block assignment 让通信和计算更稳定地重叠。

它比 MiniMax-01 更细，因为它已经进入 GEMM tile / thread block 级别；但它还没有像 FlashMoE 那样把整个 MoE operator 改造成一个 persistent GPU runtime。Comet 更像是在现有 MoE/GEMM 执行栈上，通过 dependency resolving 和 fused kernel scheduling，把 coarse EP overlap 推到 tile-level overlap。

------

## **4. FlashMoE：persistent kernel + GPU-resident scheduling**

如果说 Comet 是“把 MoE 中通信和 GEMM 之间的 shared tensor 拆到 tile 级，并重排 producer-consumer 顺序”，那 FlashMoE 再往前走了一步：

> 只优化某两段 pipeline 还不够。只要 MoE 仍然依赖 host-managed scheduling、bulk-synchronous collectives 和大量短 kernel launch，就仍然会有系统级 idle gap。FlashMoE 想把整个 distributed MoE operator 放进一个 GPU-resident persistent kernel 里。

这篇文章适合在读完 Comet 后继续看，因为它们都在讲 fine-grained overlap，但抽象层级不同：

```text
Comet:
  找 shared tensor -> 按 consumer 依赖切分 -> 重排 GEMM tile -> 专门 thread blocks 做通信/计算

FlashMoE:
  把 MoE operator 变成 GPU 常驻 runtime
  用 actor、task、symmetric tensor layout 管理 dispatch / expert compute / combine
```

### 4.1 从 Comet 视角看 FlashMoE 的动机

FlashMoE 论文的 Introduction 和 Motivation 里强调三个传统 MoE 系统问题。

![FlashMoE overview](https://flash-moe.github.io/static/images/intro-fig.png)

第一是 **synchronous communication**。传统 MoE 常用 `AllToAll` / `AllGather`。这些 collective 是 bulk-synchronous 的：参与通信的 GPU 都要进入同一个集体操作，慢的 GPU 会拖住快的 GPU。MoE router 又是动态的，每个 step 分到各 expert 的 token 数不同，所以 straggler 很常见。

第二是 **kernel launch overhead**。一个 MoE forward 可能包含：

```text
router kernel
dispatch communication kernel
expert GEMM kernel
activation kernel
expert GEMM kernel
combine communication kernel
```

如果每个阶段都要 host 端发起 kernel，并在 kernel 之间交接，短 kernel 和同步点会堆出很明显的 CUDA API / launch gap。FlashMoE 论文里强调，它的目标是一个 persistent kernel，而不是一串短命 kernel。

第三是 **task locality 没有被充分利用**。Comet 已经告诉我们：数据到达和 GEMM tile ready 是更细的粒度。但如果调度仍然由 CPU 或粗粒度 collective 管，GPU 很难做到“这个 tile ready 了就马上安排一个 block 去算/传”。

所以 FlashMoE 的目标不是再提出一种新的 router，也不是改变 MoE 数学，而是把 MoE 的执行形态从：

```text
host launches many kernels + collective barriers
```

变成：

```text
one persistent GPU kernel + device-side tasks + one-sided communication
```

### 4.2 MoE 数学没有变：变的是执行系统

FlashMoE 仍然处理普通的 MoE FFN。对每个 token $x_i$，gate 选择 top-k experts，并给出 combine weights。一个简化的 MoE 输出可以写成：

$$
y_i = \sum_{j=1}^{k} w_{i,e_j} \cdot E_{e_j}(x_i)
$$

其中 $E_{e_j}$ 是第 $j$ 个被选中的 expert，$w_{i,e_j}$ 是 gate 给出的权重。

每个 expert 本身还是普通 FFN：

$$
E_e(x)=W_{e,2}\,\sigma(W_{e,1}x+b_{e,1})+b_{e,2}
$$

所以要记住：

> FlashMoE 改的是 MoE operator 的 runtime，不是 MoE 的函数形式。

这和 Comet 一样：Comet 的 shared tensor decomposition 也不改变数学结果，只改变数据到达、GEMM tile 和 consumer operator 的执行顺序。

### 4.3 FlashMoE 的核心架构：single persistent kernel

项目页里的架构图很适合作为第一张地图。

![FlashMoE architecture](https://flash-moe.github.io/static/images/architecture.jpg)

FlashMoE 的 persistent kernel 在每张 GPU 上长期运行。它内部不是所有 thread blocks 都做同一件事，而是用 actor model 把 blocks / warps 分成几类角色：

| 角色 | 作用 |
|---|---|
| Processor | 执行真正的计算和通信任务，例如 GEMM、element-wise、combine、tile transfer |
| Subscriber | 接收 peer GPU 发来的 tile packets，并解码成 task descriptors |
| Scheduler | 维护 ready queue，把已经 ready 的 task 分配给 Processor |

论文实现里，大部分 thread blocks 是 Processor。最后一个 block 被当作类似 “OS block” 的管理块，其中三条 warps 做 Subscriber，一条 warp 做 Scheduler。这个细节不太显眼，但很重要：FlashMoE 并不是让所有 blocks 都参与调度，而是用很少的 GPU 资源做管理，把大部分 SM 留给计算任务。

这和 Comet 的 thread block specialization 有继承关系：

```text
Comet:
  compute blocks + communication blocks

FlashMoE:
  Processor blocks + Subscriber/Scheduler management warps
```

Comet 的重点是隔离 remote I/O 和 GEMM；FlashMoE 的重点是进一步把 task 的产生、通知、调度、执行都放进 GPU kernel 内部。

### 4.4 Task abstraction：tile 是 runtime 的基本工作单元

Comet 里我们已经反复说过，tile 不是 token chunk，而是 GEMM / tensor 的静态分块。FlashMoE 沿用这个视角，但把 tile 进一步封装成 runtime task。

论文把 task descriptor 理解成一组 metadata + operator。一个 task 至少要告诉 Processor：

```text
我要处理哪块 tile
这个 tile 属于哪个 expert / 哪张 GPU / 哪个通信阶段
要执行的是 FFN、combine，还是 tile transfer
输入地址和输出地址在哪里
依赖是否已经 ready
```

用一个很简化的伪代码表达：

```python
while persistent_kernel_is_running:
    task = scheduler.pop_ready_task()

    if task.type == "dispatch_tile":
        processor.transfer_tile(task)

    elif task.type == "ffn_tile":
        processor.gemm_and_activation(task)

    elif task.type == "combine_tile":
        processor.weighted_accumulate_and_writeback(task)
```

这里的关键不是伪代码本身，而是 **ready task** 这个概念。传统 MoE 是阶段式：

```text
所有 dispatch 完成 -> 所有 expert GEMM 开始 -> 所有 combine 开始
```

FlashMoE 是任务式：

```text
某个 dispatch tile 到了 -> 生成 FFN task
某个 FFN output tile 完成 -> 生成 combine task
某个 remote packet 到了 -> Subscriber 解码成 task
```

这就把 Comet 的“tile ready 就尽早消费”推广成一个 GPU 内部的任务系统。

### 4.5 Tile dimensions：为什么不是越大越好

FlashMoE 论文里有一个很值得记的工程细节：它选择的 tile size 是 $(128,64)$。

这和我们讨论 Comet 时常说的 $128\times128$ 不矛盾。不同 kernel、不同 operator、不同寄存器压力下，最优 tile shape 会变。FlashMoE 给出的直觉是：

- tile 太小：每个 task 的计算量太少，GPU 利用率低，调度开销相对变大。
- tile width 太大：每个线程需要保存更多寄存器，容易 register spill 到 local memory。
- tile height 太大：单线程工作量变重，可能降低并行度。
- thread block 太大：同时驻留的 blocks 变少，SM occupancy 下降，`__syncthreads()` 等 block 内同步成本也更高。

所以 tile size 不是“expert 能处理的最大 token 数”，而是 kernel/hardware co-design 的结果。这个点和 Comet 的 granularity mismatch 可以接上：tile 是 GPU 计算和调度的单位，不是 MoE 逻辑层的 token chunk。

### 4.6 One-sided communication：UVA、NVSHMEM 和 device-initiated DMA

FlashMoE 和 Comet 都想摆脱纯 collective 的粗粒度等待，但 FlashMoE 更强调 **one-sided, device-initiated communication**。

传统 `AllToAll` 是：

```text
所有 GPU 进入 collective
runtime 统一搬数据
所有参与方等待完成
```

FlashMoE 使用 NVSHMEM 建立跨 GPU 的 global address space，并在可用时利用 UVA 做 DMA / RDMA 风格的数据搬运。直觉上，GPU kernel 里的 Processor 或 communication task 可以直接把某个 tile 写到目标 GPU 的某个地址，而不是等 host 发起一个大 collective。

但这里有两个容易误解的点。

第一，one-sided 不等于没有通信成本。远端读写仍然有延迟和带宽限制，只是它不再要求所有 GPU 同步进入同一个 collective。

第二，one-sided 需要非常小心的内存布局。如果两个 GPU 同时往目标 GPU 的同一块 buffer 写，就会发生 write-write conflict。FlashMoE 的 symmetric tensor layout 就是为了解决这个问题。

### 4.7 Symmetric Tensor Layout：为什么需要多出来的 temporal buffers

FlashMoE 的 Figure 7 讲的是 symmetric tensor layout。它是全篇一个不太容易一眼看懂、但非常关键的设计。

论文里可以把这个 layout 粗略读成：

$$
L \sim [P, r, 2, s, n, C_{\mathrm{up}}, h]
$$

这里不是逐字符复刻论文公式，而是帮助理解每个维度的含义：

| 符号 | 含义 |
|---|---|
| $P$ | expert-parallel world size |
| $r$ | communication rounds，例如 dispatch 和 combine |
| $2$ | 每个 communication round 有 outgoing / incoming 两个方向 |
| $s$ | staging buffers 数量 |
| $n$ | 每张 GPU 上 local experts 数 |
| $C_{\mathrm{up}}$ | upscaled expert capacity |
| $h$ | token hidden dimension |

为什么要这么复杂？因为 FlashMoE 要支持 fully non-blocking one-sided writes。不同 GPU 的 task 会同时向 symmetric layout 中写 tile。如果只用一个普通 buffer，就很难避免冲突；要么加锁，要么同步，要么冒险覆盖。

FlashMoE 的思路是加 temporal dimensions：

```text
不同通信轮次用不同 slots
outgoing 和 incoming 分开
不同 staging buffer 分开
不同 peer/local expert/capacity slot 分开
```

这样一个 remote write 的目标地址由 source GPU、target GPU、round、direction、expert slot、capacity slot 等 metadata 唯一决定。论文还给了 theorem：这个布局是 write-write conflict-free。

这个设计和 Comet 的 shared tensor 思路有微妙区别：

```text
Comet shared tensor:
  关注 producer 和 consumer 怎么共享一块中间 tensor，并按依赖切分。

FlashMoE symmetric tensor:
  关注跨 GPU one-sided writes 怎样在不加同步的情况下安全落位。
```

换句话说，Comet 主要解决“什么时候可以算”；FlashMoE 还要解决“数据可以安全写到哪里”。

### 4.8 In-place padding：为什么 padding 也会浪费通信

MoE 里常见一个 capacity 概念：每个 expert 最多接收多少 token。为了让 buffer shape 固定，很多实现会把 expert input padding 到 capacity。

例如某个 expert capacity 是 128，但这次只收到 37 个真实 token。传统实现可能会构造一个 128 行的 buffer：

```text
37 real tokens + 91 null tokens
```

如果这些 null tokens 也被跨 GPU 传输，就浪费网络带宽。FlashMoE 的 payload efficiency 指的就是避免发送这种无意义 payload。

FlashMoE 的 in-place padding 思路是：

```text
只把真实 token 通过网络发送过去
padding 在本地 symmetric tensor buffer 里完成
```

这样既满足后续 Processor 按固定 tile shape / aligned capacity 读取的需求，又避免把 null token 当作真实数据走网络。

这个点有点“不显眼但很实用”：MoE 的动态路由会导致每个 expert 的真实 token 数远小于或不等于 capacity，如果系统总是按 capacity 传输，那么稀疏计算节省下来的部分收益会被无效通信吃掉。

### 4.9 FlashMoE 的执行流程：从一个 token tile 的视角看

可以用一个 tile 的生命周期来理解 FlashMoE。

```text
1. Gate / routing 产生 token -> expert 的映射。

2. 对某个目标 expert，Processor 生成 dispatch tile task。

3. 如果 expert 在远端 GPU：
   Processor 通过 NVSHMEM / UVA 把 tile 写入目标 GPU 的 symmetric tensor slot。
   同时发送 signal，通知目标 GPU 有 tile 到达。

4. 目标 GPU 的 Subscriber 收到 packet / signal，
   解码成 task descriptor。

5. Scheduler 把 ready task 分给空闲 Processor。

6. Processor 执行 expert FFN tile：
   Linear-1 -> activation -> Linear-2。

7. 输出 tile ready 后，生成 combine task。

8. combine task 根据 routing weight 做 weighted accumulation，
   如果原 token 在远端，再通过 one-sided write 写回。
```

这条链路的核心是：每个 tile 的 arrival、compute、combine 都可以独立推进，而不是被整个 MoE layer 的阶段边界卡住。

### 4.10 和 Comet 的关系：不是替代，而是更底层的 runtime 化

如果只读过 Comet，很容易把 FlashMoE 理解成“另一个更激进的 Comet”。这个说法有一半对，一半不够准确。

相同点：

- 都认为 MoE 的瓶颈来自通信和计算的粒度不匹配。
- 都把粒度降到 tile/task 级别。
- 都不满足于传统 EP overlap 的 token chunk 级 pipeline。
- 都需要把通信和 GEMM 资源隔离开，避免 remote I/O 破坏计算。

不同点：

| 维度 | Comet | FlashMoE |
|---|---|---|
| 核心抽象 | shared tensor dependency resolving | GPU-resident actor/task runtime |
| 主要对象 | 两条 producer-consumer pipeline | 整个 distributed MoE operator |
| 通信方式 | 更细粒度 P2P / kernel 协调 | one-sided device-initiated DMA/RDMA |
| 调度位置 | fused kernel 内的 block assignment | persistent kernel 内 Scheduler 分配 tasks |
| 内存布局重点 | shared tensor 的切分方向和执行顺序 | symmetric tensor layout 保证 non-blocking writes |
| 解决的额外问题 | granularity mismatch | kernel launch、collective barrier、payload inefficiency |

所以我会这样理解：

```text
Comet:
  把 MoE 的通信-计算 overlap 做到 tile-aware。

FlashMoE:
  把 MoE 的执行环境改成 tile-task runtime。
```

### 4.11 读 FlashMoE 时值得记录的“不太显眼”的细节

**第一，single kernel 不等于把所有代码机械粘在一个 kernel 里。** 真正关键是 persistent kernel 内部有 task abstraction、actor role 和 ready queue。否则只是把阶段写进一个大 kernel，仍然可能内部空转。

**第二，Subscriber / Scheduler 是成本，不是免费午餐。** FlashMoE 特意只用最后一个 block 做 OS block，就是为了把管理成本压低。调度能力太弱会供不上 tasks；调度资源太多又会抢走计算 blocks。

**第三，tile size 是硬件约束下的折中。** 论文选择 $(128,64)$，背后是 register pressure、shared memory、SM occupancy、block synchronization 的平衡，不是理论上越大越好。

**第四，symmetric tensor layout 是正确性设计，不只是性能优化。** 如果没有这个布局，one-sided writes 可能需要同步或锁；一旦同步变多，FlashMoE 的核心优势就会被抵消。

**第五，payload efficiency 是 MoE 特有问题。** 因为 router 动态分配 token，capacity padding 很常见。FlashMoE 的 in-place padding 让网络只传真实 token，不传 null token。

**第六，实验结果要看清边界。** 论文的 evaluation 主要测单个 MoE layer forward，硬件是 8 张 H100；FlashMoE 用 FP32，而很多 baseline 用 FP16，论文认为这反而让 FlashMoE 更吃亏。但这也意味着，读结果时要把精度、forward-only、单层 MoE operator 和完整训练系统区分开。

### 4.12 FlashMoE 的定位

FlashMoE 可以理解成：

> 把 distributed MoE layer 从“host 发起的一串 kernels + collectives”改造成“GPU 内部常驻的 tile-task runtime”。

它比 Comet 更系统级、更激进。Comet 解决的是 shared tensor 的依赖切分和 tile 重排；FlashMoE 解决的是整个 MoE operator 如何在 GPU 内部自行调度、通信、计算和写回。

这也是为什么 DeepSeek-V4 那种 expert-wave pipeline 读起来会更像工程折中：它吸收了 fine-grained overlap 的思想，但没有把论文重点放在一个完整 persistent MoE runtime 上。而 FlashMoE 的主张更明确：**要突破 MoE 系统瓶颈，不能只看 GEMM，也不能只看通信，必须把 kernel launch、调度、远程写、内存布局和 padding 一起设计。**

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
