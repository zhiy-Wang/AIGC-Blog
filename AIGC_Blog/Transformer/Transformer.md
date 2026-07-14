# 从 Self-Attention 到 DiT：Transformer 成为核心架构？

> 这篇文章记录我对 Transformer 的第一轮完整理解。重点不是把论文公式重新抄一遍，而是回答三个问题：
>
> 1. Self-Attention 到底在算什么？  
> 2. Cross-Attention 为什么能让文本控制图像或视频？  
> 3. Transformer 为什么会进入 Diffusion，并演化成 DiT？

## 目录

- [1. 为什么需要 Transformer](#1-为什么需要-transformer)
- [2. 从文字到向量](#2-从文字到向量)
- [3. Self-Attention 到底在做什么？](#3-self-attention-到底在做什么)
- [4. 为什么要除以 sqrt(d_k)？](#4-为什么要除以-sqrtd_k)
- [5. Multi-Head Attention 为什么需要多个头？](#5-multi-head-attention-为什么需要多个头)
- [6. Transformer Block 不只有 Attention](#6-transformer-block-不只有-attention)
- [7. Self-Attention 和 Cross-Attention 的区别](#7-self-attention-和-cross-attention-的区别)
- [8. Transformer 如何进入 Diffusion？](#8-transformer-如何进入-diffusion)
- [9. 为什么视频生成尤其需要 Transformer？](#9-为什么视频生成尤其需要-transformer)
- [10. 我的理解总结](#10-我的理解总结)

---

## 1. 为什么需要 Transformer

在 Transformer 出现之前，序列建模通常依赖 RNN
RNN的计算过程可以简单理解为：

```text
x1 → h1 → h2 → h3 → ...
```

后一个位置依赖前一个位置，因此计算天然是串行的。序列很长时，早期信息还需要经过多次传递才能影响后面的位置。这也就导致了长程依赖困难的问题

Transformer 换了一种思路：

> 不再让信息沿着序列一步步传递，而是让每个 token 直接查看其他 token，并决定自己应该从哪里取信息。

这就是 Attention。
假设一句话有 N 个 token，每个 token 用一个 D 维向量表示，那么输入可以写成：

```math
X \in \mathbb{R}^{N \times D}
```

Transformer 的目标不是简单地压缩这些向量，而是不断更新每个 token 的表示，让它逐渐包含上下文信息。

## 2. 从文字到向量

文字通常不能直接进入神经网络，而是经过两步处理

```text
1>.文字 -> token id
2>.token id -> Embeeding
```

例如：

```text
"Transformer is powerful"
```

经过 tokenizer 后，可能得到：

```text
[1012, 3821, 726]
```

这些数字只是词表中的编号，本身没有语义。随后通过 Embedding 层查表，得到每个 token 对应的向量：

```math
X \in \mathbb{R}^{N \times D}
```

这里需要特别注意：

> Embedding 不是概率，也不是把 token “压缩”成一个数字，而是把每个 token 映射成一个可学习的向量。

如果有 10 个 token，隐藏维度是 768，那么输出形状就是：

```text
[10, 768]
```

token 数量没有减少，只是每个 token 都有了一个 768 维表示。

---

## 3. Self-Attention 到底在做什么？

Self-Attention 的核心公式是：

```math
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
```

- **Query**：当前 token 想找什么信息
- **Key**：每个 token 能提供什么索引
- **Value**：每个 token 真正携带的内容

对于输入 X，通过三个不同的线性层得到：

```math
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
```

其中：

```math
Q,K,V \in \mathbb{R}^{N \times d_k}
```

然后计算：

```math
QK^\top \in \mathbb{R}^{N \times N}
```

这个矩阵非常关键。第 i 行第 j 列表示：

> 第 i 个 token 对第 j 个 token 的关注程度。

经过 softmax 后，每一行都会变成一组和为 1 的权重。最后再乘上 V，就得到每个 token 从整个序列聚合到的新信息。

### 一个最小 Self-Attention 实现

```python
import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        # x: [batch_size, num_tokens, dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = q @ k.transpose(-2, -1) / (self.dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        return out
```

---

## 4. 为什么要除以 sqrt(d_k)？

如果向量维度很高，Q 和 K 的点积数值可能变得很大。

当 softmax 的输入绝对值过大时，输出容易变得非常尖锐，例如：

```text
[0.0001, 0.0002, 0.9997]
```

这样会使梯度变得不稳定。

因此使用：

```math
\frac{QK^\top}{\sqrt{d_k}}
```

对分数进行缩放，让训练更稳定。这也是它被称为 **Scaled Dot-Product Attention** 的原因。

---

## 5. Multi-Head Attention 为什么需要多个头？

单个 Attention 只能在**一个**表示空间中计算关系。Multi-Head Attention 会把隐藏维度拆成多个 head，让不同 head 学习不同的关系。

假设：

```text
hidden_dim = 768
num_heads = 12
```

那么每个 head 的维度是：

```text
head_dim = 768 / 12 = 64
```

每个 head 独立计算 Attention，最后把结果拼接起来：

```math
\operatorname{MultiHead}(Q,K,V)
=
\operatorname{Concat}(head_1,\ldots,head_h)W_O
```

它不意味着某个 head 一定负责语法、另一个 head 一定负责位置，而是给模型提供多个并行的关系建模空间。

### Multi-Head Self-Attention 实现

```python
import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim 必须能够被 num_heads 整除")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        # x: [B N D],batch_size,num_tokens,dim
        # batch_size=n可以理解为输入是n句话
        batch_size, num_tokens, _ = x.shape
        # [B, N, D] -> [B, N, 3D]
        qkv = self.qkv_proj(x)
        # [B, N, 3D] -> [B, N, 3, H, Dh]
        # 3表示Q,K,V, H表示head，Dh表示每个头分到的维度
        qkv = qkv.view(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim
        )
        # [B, H, N, Dh]
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.dim)
        out = self.out_proj(out)
        return out
```

---

## 6. Transformer Block 不只有 Attention

一个典型的 Transformer Block 主要由两部分组成：

```text
输入
 ↓
Multi-Head Self-Attention
 ↓
Residual + LayerNorm
 ↓
Feed Forward Network
 ↓
Residual + LayerNorm
 ↓
输出
```

可以把它们理解成：

- Attention：负责 token 之间的信息交换
- FFN：负责对每个 token 的特征做非线性变换
- Residual：保留原始信息，并改善深层网络训练
- LayerNorm：稳定不同层之间的数值分布

一个常见的 Pre-Norm Transformer Block 可以写成：

```python
import torch
import torch.nn as nn
import math


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        # 第一层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.attention = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

这两行是整个 Block 的核心：

```python
x = x + self.attention(self.norm1(x))
x = x + self.ffn(self.norm2(x))
```

第一行完成上下文信息交换，第二行完成特征变换。

---

## 7. Self-Attention 和 Cross-Attention 的区别

Self-Attention 中，Q、K、V 都来自同一个输入：

```math
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
```

Cross-Attention 中，Query 和 Key/Value 来自不同输入。

例如在文本生成视频中：

- 视频 latent 产生 Q
- 文本 embedding 产生 K 和 V

可以写成：

```math
Q=X_{\text{video}}W_Q,\qquad
K=X_{\text{text}}W_K,\qquad
V=X_{\text{text}}W_V
```

它表达的是：

> 每个视频 token 根据自身状态，去文本条件中寻找需要的信息

### Cross-Attention 实现

```python
import torch
import torch.nn as nn
import math


class CrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int):
        super().__init__()

        self.dim = dim

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(context_dim, dim)
        self.v_proj = nn.Linear(context_dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ):
        """
        x:       [B, N_video, D]
        context: [B, N_text, context_dim]
        """
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        # [B, N_video, N_text]
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.dim)

        attention_weights = torch.softmax(scores, dim=-1)

        # [B, N_video, D]
        output = attention_weights @ v

        return self.out_proj(output), attention_weights
```

## 8. Transformer 如何进入 Diffusion？

传统 Diffusion 模型常使用 UNet 作为去噪网络。后来，越来越多模型开始采用 Transformer 作为 backbone，这类架构通常被称为 Diffusion Transformer，也就是 DiT。

一个简化的 DiT 流程是：

```text
图像或视频
 ↓
VAE Encoder
 ↓
Latent
 ↓
Patchify / Tokenize
 ↓
Transformer Blocks
 ↓
预测噪声或速度
 ↓
Scheduler 更新 latent
 ↓
VAE Decoder
 ↓
生成结果
```

Transformer 在这里并不是直接输出最终图像或视频，而是在每一个 diffusion timestep 中预测当前 latent 应该如何更新。

可以把一次去噪过程抽象为：

```python
for timestep in timesteps:
    prediction = transformer(
        noisy_latent,
        timestep,
        text_condition,
    )
    noisy_latent = scheduler.step(
        prediction,
        timestep,
        noisy_latent,
    )
```

因此，DiT 中存在两个不同层级的循环：

1. 外层 diffusion loop：不断更新 latent
2. 内层 Transformer：在每个 timestep 内对 token 做多层 Attention 和 FFN 计算

如果模型有 30 个 Transformer Block，那么每一个 diffusion timestep 都需要依次经过这 30 个 Block。

---

## 9. 为什么视频生成尤其需要 Transformer？

视频不仅包含空间关系，还包含时间关系。

一段视频 latent 可以被拆成大量时空 token：

```math
N = T \times H \times W
```

其中：

- T：时间维度
- H,W：空间维度的height和width

模型需要同时处理：

- 同一帧内部不同区域的空间关系
- 不同帧之间的运动和主体一致性
- 文本对整个视频内容的控制

Attention 天然适合建模 token 之间的远距离关系，因此很适合处理时空依赖。

但视频 token 数量通常非常大，而标准 Attention 的复杂度近似为：

```math
O(N^2)
```

因此实际视频模型还需要继续解决：

- 如何降低 Attention 计算量
- 如何分块生成长视频
- 如何利用 KV Cache 保存历史信息
- 如何保持多段视频中的主体和场景一致性

这也是 Long Video Generation 中不断出现 AR、KV Cache、Frame Sink、Causal Attention 等设计的原因。

---

## 10. 我的理解总结

经过这次整理，我对 Transformer 的理解可以概括为：

1. **Embedding** 把 token ID 映射成向量，但不会自动减少 token 数量。
2. **Self-Attention** 让同一序列中的 token 彼此交换信息。
3. **Multi-Head Attention** 让模型在多个表示空间中并行建模关系。
4. **FFN** 对每个 token 做独立的非线性特征变换。
5. **Cross-Attention** 让一种模态从另一种模态中读取条件信息。
6. **DiT** 使用 Transformer 作为 Diffusion 的去噪 backbone。
7. 视频生成需要同时建模空间、时间和文本条件，因此 Transformer 很适合这一任务。

Transformer 本身并不神秘。它的核心可以压缩成一句话：

> Attention 负责信息交换，FFN 负责特征变换，Residual 和 LayerNorm 保证深层网络能够稳定训练。

下一篇我准备继续整理：

> Diffusion 到底如何从随机噪声一步步生成图像或视频，以及 DiT 在每个 denoising timestep 内究竟做了什么。

---
