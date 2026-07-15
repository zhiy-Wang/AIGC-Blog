# AIGC Learning Notes

> A personal knowledge base for understanding modern generative AI models, including Transformer, Diffusion Models, and Long Video Generation.

这个仓库记录我在学习和研究 AIGC 过程中的理解、源码分析以及实验记录。

目标不是简单整理论文公式，而是尝试回答：

- 一个模型为什么这样设计？
- 核心模块具体如何工作？
- 论文思想如何对应到真实代码实现？

---

## 📚 Contents

### Transformer

从 Self-Attention 出发，理解 Transformer 的核心结构。

包括：

- Tokenization & Embedding
- Positional Encoding
- Self-Attention
- Multi-Head Attention
- Transformer Block
- Cross-Attention
- Transformer in Diffusion (DiT)


---

### Diffusion Models

学习扩散模型从噪声到生成的过程。

计划包含：

- Forward Diffusion
- Reverse Denoising
- Noise Prediction
- Scheduler
- Latent Diffusion
- Diffusion Transformer (DiT)

---

### Long Video Generation

关注长视频生成中的关键问题：

- Temporal Consistency
- Autoregressive Generation
- KV Cache
- Causal Attention
- Frame Sink
- Long Video Generation Pipeline

计划分析：

- LongLive
- SkyReels-V2
- Causal Forcing
- Wan Series

---

## 🧠 Learning Philosophy

AIGC 模型越来越复杂，但核心思想仍然可以拆解：

```
Text/Image/Video
        |
        ↓
Representation
        |
        ↓
Transformer / Diffusion
        |
        ↓
Generation
```

希望通过源码、公式和实验，将这些复杂系统拆解成可理解的模块。

---

## 🛠️ Topics

目前主要关注：

- Transformer Architecture
- Diffusion Models
- Vision Transformer
- Diffusion Transformer (DiT)
- Text-to-Video Generation
- Multimodal Models
- Long Video Generation

---

## 📌 Roadmap

- [x] Transformer fundamentals
- [ ] Diffusion Model fundamentals
- [ ] DiT architecture analysis
- [ ] Wan model source code analysis
- [ ] LongLive source code analysis
- [ ] Long video generation experiments

---

This repository is a record of my learning journey and technical exploration.
