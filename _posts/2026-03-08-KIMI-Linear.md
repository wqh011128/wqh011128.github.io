---
layout: post
title: "KIMI LINEAR、Delta Network总结"
date: 2026-03-08
categories:
  - blog
---

# KIMI LINEAR:  AN EXPRESSIVE, EFFICIENT ATTENTION ARCHITECTURE

**Link:** [MoonshotAI/Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)



------

## **Main Contributions**

- Kimi Delta Attention（KDA）：一种线性注意力机制，通过改进的循环内存管理和硬件效率来 细化门控 delta 规则。
-  Kimi Linear 架构：一种混合设计，采用 3:1 的 KDA 与全局注意力比例，在超越全注意力质量的 同时减少内存占用。
-  性能验证：通过 1.4T 令牌训练运行，Kimi Linear 在短/长上下文和 RL 风格评估中超 越full attention。



## **Knowledge Base** 

------

KIMI-Linear主要融合了以下几个网络的设计：

- MAMBA2
- Delta Network
- Gated Delta Network

因此，在介绍KIMI-Linear之前，首先需要解释清楚这些网络的**架构设计**以及**并行化实现**。

------

### **无损记录 vs** 有损压缩

注意力QK^TV有两个关键的优化思路（其实是不同的角度）：

1. 计算方式
2. Cache



计算方式在 [Minimax-01]({% post_url 2026-03-07-MiniMax-01 %}) 梳理中解释过，可以分为左乘和右乘两个大类。



本文重点要讲的是Cache，在存储KV键值对时，也可以分为如下两种类别。



- **无损KVCache：****Softmax Attention Only ! - 左乘法**
  - 历史的每一个Token都完整保存，即所有KV全都保存完好，可以无损检索；
- **有损压缩：****Lightning Attention, Linear Attention...(delete Softmax !) - 右乘法**
  - 维护一个固定大小的矩阵状态 $S$（和 $K^TV$ 保持相同的维度）；
  - 每次进来一个新的 token，就用 $S_{\text{new}} = S_{\text{old}} + K^TV$ 来更新；
  - 计算attention时直接用Q乘以S；
  - 因为S的维度是固定的，每次新的信息都是直接加在旧信息上（存在"记忆碰撞"），导致检索时精确度不如KVCache（无损失）；



**后续介绍的MAMBA-2、Delta Network 以及 Gated Delta Network都是基于****有损压缩****形式去探讨的。**

### MAMBA2：Scalar-Valued Data-Dependent Decay

关键点：给旧状态 $S_{t-1}$ 添加 decay，这里的 decay 用标量 $\alpha_t$ 来表示。

只需要记住这一点即可，下面截图是其模型关键部分的形式化描述：



![img](/img/mamba2.png)



### Delta Network 

------

Link：[2412.06464v1](https://arxiv.org/pdf/2412.06464v1)

Author's explain of deltanet: [sustcsonglin.github.io/blog/2024/deltanet-1](https://sustcsonglin.github.io/blog/2024/deltanet-1/)

------



**Delta Network 是理解 KIMI-Linear 最关键的一步。**



后者可以粗略的认为是Delta Network系列的扩展，因此，需要首先详细解释Delta Network的架构以及计算优化。



本节将依次梳理：

- Delta Network 解决有损压缩问题的关键思路是什么？
- Delta Network 转换成并行化计算之错误的尝试-Parallel Scan 
- 回顾Linear Attention的并行化计算-Chunkwise Parallel
- Delta Network如何使用Chunkwise Parallel？





**Delta Network 解决有损压缩问题的关键思路是什么？**



引入一个例子：

令状态矩阵 $S = \sum_i v_i k_i^T$，并且所有的向量都是单位向量，

这时，如果信息没有损失，从S中查询k_j应当得到v_j,但实际上并非如此，如下图

（来源于 [sustcsonglin.github.io/blog/2024/deltanet-1](https://sustcsonglin.github.io/blog/2024/deltanet-1/)）：

![img](/img/delta_problem.png)

为了消除 error，需要让 $k_i^T k_j = 0,\ i \ne j$。也就是说，token $i$ 和 token $j$ 必须是正交的。



但是，feature 的维度只有d维，token数量大于d时，由于rank最大只能是d，无法满足正交条件。

从直观上来看，要求不同token之间的表示是正交的，这是很难的。



**所以，问题在于，如何尽可能的消除error？**



Delta network给出的方案是：**错误修正(error-correction principle)**



解决方案总结如下：

- 架构调整：思路是修改状态矩阵，**将目前的状态修改为期望得到的状态**。
- $S_{t-1}k_t$ 相当于 $k_t$ 检索出的 “old value”，积累着过去的记忆和检索 error。
- $v_t$ 表示理想的检索值。
- 用加权的方式重组value（即要准确检索，也不能完全丢失记忆）。



具体来说，Delta Network的公式如下所示：

![img](/img/delta_paper.png) 





**Delta Network 转换成并行化计算之错误的尝试-Parallel Scan** 



主要讨论以下两个问题：

- **Parallel Scan 是什么？**
- **为什么用在Delta Network上不行？**



已知，Delta Network公式如下：



![img](/img/parallel_scan.png)



可以做如下定义：

定义：$c_t = [M_t, X_t]$，因此，$S_t$ 可以通过 $c_t$ 的确定而确定。

将迭代公式看作一步更新对应的函数：$f_t(S_{t-1}) = S_{t-1} M_t + X_t$

假设初始状态为 $S$，继续定义区间 $[0\!:\!1]$ 的“复合变换” $f_{0:1}(S) = f_1\!\bigl(f_0(S)\bigr)$

将总变换写回 $SM+X$ 的形式 $f_{0:1}(S) = S M_{0:1} + X_{0:1}$，如下：

$$
\begin{aligned}
f_1\!\bigl(f_0(S)\bigr) &= f_1(SM_0 + X_0) \\
&= (SM_0 + X_0)M_1 + X_1 \\
&= S(M_0M_1) + (X_0M_1 + X_1).
\end{aligned}
$$

因而得到区间 $[0\!:\!1]$ 的合成对 $c_{[0:1]} = [\,M_0M_1,\; X_0M_1 + X_1\,]$



我们可以继续迭代下去，例如获得复合变换c_{[i:j]}。



这表明，S_t可以直接由下标为i的S_i(i给直接求出来。



再来举一个例子，

对于下图四个块，

- S0是直接可获得的
- S1最快的办法是0->1复合
- S2最快的办法是0-1复合，再用（0->1）->2复合
- S3最快的办法是2->3复合，再用(0->1)->(2->3) 复合



![img](/img/parallel_scan1.png)



但是，为什么这种方法实际不可用？



原因在于将M连乘会导致项数爆炸。





**回顾Linear Attention的并行化计算-Chunkwise Parallel**



------

首先以向量的视角建立基本递推：

设 $q_t, k_t, v_t \in \mathbb{R}^{d\times 1}$（列向量），

状态 $S_t \in \mathbb{R}^{d\times d}$，

输出 $o_t \in \mathbb{R}^{d\times 1}$，

递推为： 

$$
S_t = S_{t-1} + v_t k_t^\top,\qquad o_t = S_t q_t.
$$

------

**分块（chunk）**与堆叠方式

块长度为 $C$，块索引为 $i$，块内位置 $r = 1,\dots,C$

在列向量约定下，把 token 按列堆叠：

$$
\begin{aligned}
Q_{[i]} &= [q_{[i]}^1,\dots,q_{[i]}^C] \in \mathbb{R}^{d\times C} \\
K_{[i]} &= [k_{[i]}^1,\dots,k_{[i]}^C] \in \mathbb{R}^{d\times C} \\
V_{[i]} &= [v_{[i]}^1,\dots,v_{[i]}^C] \in \mathbb{R}^{d\times C} \\
O_{[i]} &= [o_{[i]}^1,\dots,o_{[i]}^C] \in \mathbb{R}^{d\times C}
\end{aligned}
$$



块起点（checkpoint）状态：

$$
S_{[i]} := S_{iC}\in\mathbb{R}^{d\times d}.
$$

块中间第r个位置状态的表述为：

$$
S_{[i]}^r := S_{[i]} + \sum_{t=1}^{r}v_{[i]}^t(k_{[i]}^t)^T \in \mathbb{R}^{d\times d}.
$$

![img](/img/chunk1.png)

------

计算出第r个位置状态后就可以计算块内第  r 个 token 的输出：

$$
o_{[i]}^r = S_{[i]}^r q_{[i]}^r = S_{[i]}q_{[i]}^r + \sum_{t=1}^{r}v_{[i]}^t(k_{[i]}^t)^T q_{[i]}^r.
$$

------

我们的目标是并行化计算，也就是能否一次性计算出一块块的所有输出，

也就是能否一次性计算出 $O_{[i]}$？

$O_{[i]}$ 等于 $o_{[i]}^r$ 堆叠成矩阵：

$$
O_{[i]} = S_{[i]}Q_{[i]} + \sum_{r=1}^{C}\sum_{t=1}^{r}v_{[i]}^t(k_{[i]}^t)^T q_{[i]}^r
$$

![img](/img/chunk2.png)

所以，右侧计算可以直接用矩阵运算乘以一个**上三角矩阵**来解决

$$
\boxed{ O_{[i]} = S_{[i]}Q_{[i]} \;+\; V_{[i]}\Bigl((K_{[i]}^\top Q_{[i]})\odot M\Bigr) } \qquad (O_{[i]}\in\mathbb{R}^{d\times C})
$$

因此，可以一次性计算第i个块的所有输出，只要获得了S_{[i]}。

$S_{[i]}$ 也可以一次性计算完毕，

$$
S_{[i]} = S_{[i-1]} + \sum_{t=1}^{C}v_{[i-1]}^t(k_{[i-1]}^t)^T = S_{[i-1]} + V_{[i-1]}(K_{[i-1]})^T
$$

------

**转换成行向量表示形式？（最终输出（C,d）维度）**



大部分实现都是把 token 放在行上，于是定义：

$$
\tilde Q_{[i]} := Q_{[i]}^\top\in\mathbb{R}^{C\times d},\quad
\tilde K_{[i]} := K_{[i]}^\top\in\mathbb{R}^{C\times d},\quad
\tilde V_{[i]} := V_{[i]}^\top\in\mathbb{R}^{C\times d},\quad
\tilde O_{[i]} := O_{[i]}^\top\in\mathbb{R}^{C\times d}.
$$



把列向量公式整体转置

最终得到行向量布局下：

$$
\boxed{ \tilde O_{[i]} = \tilde Q_{[i]} S_{[i]}^\top \;+\; \bigl((\tilde Q_{[i]}\tilde K_{[i]}^\top)\odot \tilde M\bigr)\tilde V_{[i]} } \qquad (\tilde O_{[i]}\in\mathbb{R}^{C\times d})
$$

------



**Delta Network如何使用Chunkwise Parallel？**



Delta Network因为有c_t =[M_t, X_t]的存在，导致想并行化时会遇到很多个 d×d 矩阵相乘。

如果当作稠密矩阵乘，代价高、还要存很多中间 d×d 状态（I/O）。



**WY表示法**

![img](/img/wy.png)

**下面这一步非常关键**：它把 DeltaNet 的“矩阵状态”也变成了**一堆外积的和**，从而让 chunkwise 的整体形状开始像线性注意力那样能用矩阵乘实现，

![img](/img/wy2.png)

![img](/img/wy3.png)

![img](/img/wy4.png)



**UT Transform**

虽然 $O$、$S$ 更新都写成 matmul 了，但 $\mathbf{W}_{[i]}, \mathbf{U}_{[i]}$ 仍是递推定义的，串行依赖还在。原文明确点名这是关键瓶颈，并提出 UT transform。

![img](/img/ut.png)



### Gated Delta Networks

总结：结合了2.3和2.4节。

给 Delta Network 添加了门控单元 $\alpha_t$ 控制状态衰减。

![img](/img/gateddelta.png)






## Kimi Delta Attention: Improving Delta Rule with Fine-grained Gating

关键：将 Gated Delta Network (GDN) 中的标量门控单元替换成细粒度对角化门控 Diag($\alpha_t$)。



![img](/img/kda.png)

![img](/img/kda1.png)



------

补充解释：

1. **为什么在有了 WY 之后还需要 UT trasnform？**

- 即使有了 WY  representation可以一次性算出一个chunk的state，但公式 (4) 和 (5) 依然是**串行递归**的（有求和）。
- 在训练时存在循环算这几个矩阵，导致效率不足。

1. **UT Transform 到底做了什么？**

- UT Transform (Unit Triangular Transform) 的核心任务是：把公式 (4) 和 (5) 这种“求和递归”通过代数手段改写，变成一个涉及“**单位下三角矩阵求逆**”的操作。

------

最后，chunkwise的计算如下图所示：

![img](/img/kdachunk.png)


## Kimi Linear Model Architecture

- Hybrid architecture：三层KDA + 一层MLA（full global attention）
- No Position Encoding (NoPE) for MLA Layers：KDA被建立为主要的位置感知算子。
  - KDA 继承了 RNN 的特性，拥有衰减机制。在线性递归中，模型天然地知道t时刻是在t-1之后的。
  - MLA充当全局信息提取，在推理阶段可以转换成MQA，且摆脱了RoPE的外推问题。


架构图如下：

![img](/img/kda_arch.png)
