# 第11章 Token：AI宇宙的基本粒子

> "Token是AI世界的原子——看不见、摸不着，但一切智能都由它构成。理解Token，就是理解AI的第一性原理。" —— 本书作者

---

> **🗺️ 本章地图**
>
> | 项目 | 内容 |
> |:----|:------|
> | **核心问题** | ① 从字节到Token经历了怎样的转化过程？② BPE算法如何决定模型的"词汇量"和语言偏好？③ 嵌入矩阵的SVD分解揭示了什么语义秘密？④ Prefill vs Decode——推理为什么分两阶段？⑤ Temperature/Top-p/Top-k采样如何影响输出质量？ |
> | **前置知识** | 基础线性代数（矩阵乘法、SVD概念）、基本概率论（熵、softmax） |
> | **阅读时间** | 40-50分钟 |
> | **关键词** | Tokenizer、BPE、嵌入矩阵、SVD、Prefill、Decode、KV-cache、Temperature、Top-p、Top-k、RoPE、ALiBi |

---

## 11.1 分词的艺术：从字节到Token

### 11.1.1 什么是分词？

当你向ChatGPT输入一段文字时，它看到的不是完整的句子，也不是一个个独立的汉字或英文字母，而是一串被称为**Token**的数字ID。

**Token**是语言模型处理文本的最小单位。对于英文，一个Token通常是一个子词（subword），如"unbelievable"可能被分为["un", "believ", "able"]；对于中文，一个Token可能是一个汉字、一个常用词或一个成语。

这个过程被称为**分词（Tokenization）**——它将原始文本转换为模型可以理解的数字序列。

分词的看似简单，实则对模型行为有深远影响：
- **Token数量与计算成本成正比**——更长的token序列意味着更多的计算
- **分词的边界影响语义理解**——"New York"作为一个Token vs 两个Token，语义表征质量不同
- **分词的粒度过细影响效率**——按字符分词会导致极长序列，$O(n^2)$ 的注意力计算不可承受

### 11.1.2 BPE：字节对编码

**BPE（Byte Pair Encoding）** 是当今最主流的分词算法，被GPT系列、Llama系列、DeepSeek等几乎所有主要LLM使用。

**算法步骤**：

```
输入：原始语料（如"low lower low lowest"）
目标：学习一个"子词词表"

第1步：将每个词拆分为字符序列
"low" → l, o, w
"lower" → l, o, w, e, r
"lowest" → l, o, w, e, s, t

第2步：统计所有相邻字节对的频率
(l, o): 3次, (o, w): 3次, (w, e): 2次, (e, r): 1次, (e, s): 1次, (s, t): 1次

第3步：合并最频繁的对 (l, o) → "lo"
"low" → lo, w
"lower" → lo, w, e, r
"lowest" → lo, w, e, s, t

第4步：重复步骤2-3，直到词表达到目标大小
下一次合并：(lo, w) → "low"（频率3）
再下一次：(low, e) → "lowe"（频率2）
...

训练完成后，词表中包含：
[..., "l", "o", "w", "e", "r", "s", "t", "lo", "low", "lowe", "lowes", "lower", "lowest"]
```

训练完成后，当遇到新词时，使用**贪心合并**策略：从左到右扫描，尽可能合并成词表中已有的最长子词。

**BPE为什么优于按词分词？**

| 维度 | 按词分词 | BPE子词分词 |
|:----|:---------|:------------|
| 词表大小 | 数十万到数百万 | **32K-128K（可控）** |
| 未登录词（OOV） | 常见问题 | **几乎不存在**——任何词都可分解为子词 |
| 形态学处理 | 不处理 | 自动学习"un-""-ed""-ing"等构词单元 |
| 对拼写错误的鲁棒性 | 差 | 好（"teh"→模型可能分解为"te"+"h"） |
| 对多语言支持 | 困难（每种语言需独立词表） | **好**（同一词表可覆盖多语言） |

**中文分词的特殊性：**

中文没有天然的词边界（空格），这使得分词比英文更复杂：

```python
# 中文分词示例
"北京大学位于海淀区"
# 粒度1：按字
['北', '京', '大', '学', '位', '于', '海', '淀', '区']
# 粒度2：按词（需要额外分词工具）
['北京大学', '位于', '海淀区']
# 粒度3：BPE自动学习
['北京', '大学', '位于', '海淀', '区']  # 可能在训练中自动学到
```

GPT-4的tokenizer对中文的处理方式是：常用汉字（如"的"、"是"）通常是一个字一个Token，生僻字或成语可能被拆分为多个Token。这意味着中文文本**每个Token的信息密度高于英文**——中文的压缩率（每个Token对应的字符数）约为1.5-2.0，而英文为3.5-4.0。

### 11.1.3 Token与计算成本的直接关系

Token数量直接决定了计算成本，因为Transformer的计算复杂度为 $O(n^2)$（注意力层）：

| 模型 | 词表大小 | 平均Token/中文词 | 平均Token/英文词 |
|:----|:--------:|:----------------:|:----------------:|
| GPT-3.5 | 50,257 | ~1.8 | ~1.3 |
| GPT-4 | ~100,000 | ~1.5 | ~1.2 |
| Llama 3 | 128,000 | ~2.0 | ~1.4 |
| DeepSeek-V2 | 102,400 | ~1.7 | ~1.3 |

这意味着：**同样的内容，中文文本需要的Token数大约是英文的1.3-1.5倍**。这是为什么中文API调用通常比英文贵的原因之一。

---

> **🔬 数学深潜：BPE的贪心合并算法与期望Log-likelihood最大化**

> BPE的贪心合并策略有一个信息论解释：每次合并两个最频繁的字节对，实际上是在**最大化合并后语料库的压缩率**。
>
> 从信息论的角度，我们希望学习一个分词方案，使得：
> $$\mathcal{L} = -\sum_{i} \log P(\text{token}_i)$$
>
> BPE的贪心合并等价于在每个步骤选择使压缩增益最大的合并。假设语料库当前由 $n$ 个token组成，合并頻率为 $f$ 的一对token（每个占用2个位置），我们会用 $f$ 个新token替换 $2f$ 个旧token，语料库长度减少 $f$。合并后，语言模型的困惑度会下降——因为需要预测的"位置"变少了。
>
> **从C++视角看Trie树在Tokenizer中的实现：**
>
> 高效的Tokenizer实现使用**Trie树（前缀树）** 来存储词表：
>
> ```cpp
> struct TrieNode {
>     unordered_map<int, TrieNode*> children; // 下一个Token ID → 子节点
>     int token_id;                           // 如果是完整词，存储Token ID
>     bool is_end;                            // 是否是词尾
> };
>
> // 编码过程：贪心最长匹配
> vector<int> encode(const string& text, TrieNode* root) {
>     vector<int> tokens;
>     int i = 0;
>     while (i < text.length()) {
>         TrieNode* node = root;
>         int last_valid_id = -1;
>         int last_valid_pos = i;
>         // 贪心向前匹配最长的词表项
>         for (int j = i; j < text.length(); j++) {
>             int byte = (unsigned char)text[j];
>             if (!node->children.count(byte)) break;
>             node = node->children[byte];
>             if (node->is_end) {
>                 last_valid_id = node->token_id;
>                 last_valid_pos = j + 1;
>             }
>         }
>         tokens.push_back(last_valid_id);
>         i = last_valid_pos;
>     }
>     return tokens;
> }
> ```
>
> Trie树保证了编码的 $O(L)$ 复杂度（$L$ 为文本长度），且空间效率高——共享公共前缀（如"un"、"unb"、"unbe"共享前缀节点）。

---

## 11.2 嵌入：从离散到连续

### 11.2.1 嵌入矩阵的语义结构

Token的ID是离散的整数（如13245、897），这些数字本身没有任何语义含义。为了让模型能"理解"Token的含义，我们需要将它们映射到一个**连续的向量空间**中。这个映射由**嵌入矩阵（Embedding Matrix）** 完成：

$$ \mathbf{E} \in \mathbb{R}^{V \times d} $$

其中 $V$ 是词表大小，$d$ 是隐藏维度。Token ID为 $i$ 的词，其嵌入向量就是矩阵的第 $i$ 行。

这个嵌入矩阵在训练过程中会被优化，最终形成具有丰富语义结构的向量空间。

### 11.2.2 嵌入矩阵的SVD分解

**SVD（奇异值分解）** 是理解嵌入矩阵结构的强大工具。对于嵌入矩阵 $\mathbf{E}$，我们可以分解为：

$$ \mathbf{E} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T $$

其中 $\mathbf{U} \in \mathbb{R}^{V \times V}$，$\boldsymbol{\Sigma} \in \mathbb{R}^{V \times d}$，$\mathbf{V} \in \mathbb{R}^{d \times d}$。

$\boldsymbol{\Sigma}$ 的对角线上的奇异值 $\sigma_1 \ge \sigma_2 \ge \ldots \ge \sigma_d \ge 0$ 反映了**不同语义维度的重要性**。

---

> **🔬 数学深潜：嵌入矩阵的SVD分解**

> **奇异值谱分析：**
>
> 典型的预训练嵌入矩阵的奇异值谱呈现**长尾分布**——前10%的奇异值占据了总能量的60-70%，而尾部的30%奇异值贡献了细微的语义差异。
>
> ```python
> # 伪代码：奇异值谱分析
> U, S, Vt = svd(embedding_matrix)  # E ∈ R^{V×d}
> total_energy = sum(S[i]^2 for i in range(d))
> cum_energy = [sum(S[:k]^2) / total_energy for k in range(1, d+1)]
> # 结果示例：
> # k=10: 累积能量 35%
> # k=50: 累积能量 68%
> # k=d: 累积能量 100%
> ```
>
> **为什么嵌入维度通常不是2的幂的倍数？**
>
> GPT-4的嵌入维度为12,288，即 $12288 = 2^{12} \times 3$。这个数字不是严格的2的幂，因为:
> 1. 模型的隐藏维度往往被设计为**可以被注意力头数整除**（如12288 / 96 = 128）
> 2. 实际嵌入维度是**工程和性能的权衡**——维度越大，语义容量越大，但计算和存储成本也越高
> 3. 从信息论看，嵌入维度决定了"语义信道"的带宽——更多维度可以编码更丰富的语义差异，但也会引入噪声
>
> **嵌入维度与语义容量的关系：**
>
> $d$ 维向量可以编码的语义关系数量约为 $O(d^2)$（因为语义关系表现为向量对之间的点积和角度）。这解释了为什么从BERT的768维增加到GPT-4的12288维（16倍），模型的能力提升远大于16倍——语义容量的增长是二次的。

---

## 11.3 推理的前世今生：Prefill与Decode

### 11.3.1 推理的两阶段

LLM的推理过程分为两个截然不同的阶段：

**第一阶段：Prefill（预填充）**

当用户输入一个查询（Prompt）时，模型需要一次性处理所有输入Token。这个阶段的特点是：**大量并行计算**。

在Prefill阶段，模型计算所有输入Token的注意力，生成每个位置对应的Key和Value向量，并缓存这些信息供后续使用。

**第二阶段：Decode（解码）**

模型逐Token生成输出。每一步只生成一个Token，然后将新Token的Key和Value添加到缓存中，再预测下一个Token。

这个阶段的特点是：**串行、内存密集**——每一步都受限于KV-cache的访问速度。

---

> **🔬 数学深潜：注意力计算的IO复杂度分析**

> Prefill和Decode的计算特性完全不同：
>
> **Prefill阶段（计算密集）：**
>
> 对于输入长度 $n$ 和隐藏维度 $d$：
> - 计算量：$O(n^2 d)$（注意力）+ $O(nd^2)$（FFN）
> - 内存访问量：$O(nd)$（读取输入）+ $O(n^2)$（写入注意力分数）
> - **瓶颈：计算**（矩阵乘法）
>
> **Decode阶段（内存密集）：**
>
> 对于已生成 $m$ 个Token，生成第 $m+1$ 个Token：
> - 计算量：$O(m d)$（注意力）+ $O(d^2)$（FFN）
> - 内存访问量：$O(md)$（读取KV-cache）+ $O(d)$（读取新token的Q）
> - **瓶颈：内存带宽**（KV-cache读取）
>
> **为什么推理引擎使用CUDA Graph优化？**
>
> CUDA Graph将一系列GPU操作（kernel launch）编译为一个**静态计算图**，减少kernel launch的开销。在Decode阶段，每一步的计算模式几乎相同（只是KV-cache长度不同），这正好是CUDA Graph的理想场景——避免每次生成都重新编译和调度GPU操作，可带来 **1.5-2倍** 的推理加速。
>
> ```cpp
> // 伪代码：KV-cache的实现
> struct KVCache {
>     Tensor<float> keys;   // [num_layers, max_seq_len, num_heads, head_dim]
>     Tensor<float> values; // [num_layers, max_seq_len, num_heads, head_dim]
>     int current_len;      // 当前缓存长度
>
>     // Decode步骤：只计算新token，复用历史缓存
>     Tensor<float> attend_step(Tensor<float> q_new) {
>         // q_new: [num_heads, head_dim]
>         // keys[:current_len]: [current_len, num_heads, head_dim]
// 只需计算新q与所有历史k的注意力
>         Tensor<float> scores = matmul(q_new, keys[:current_len].transpose());
>         Tensor<float> weights = softmax(scores / sqrt(head_dim));
>         Tensor<float> output = matmul(weights, values[:current_len]);
>         return output;
>     }
> };
> ```

---

### 11.3.2 KV-cache：时间换空间的工程智慧

KV-cache是LLM推理中最关键的工程优化之一。它的核心思想是：**缓存已经计算过的Key和Value，避免重复计算**。

在生成长序列时（如10,000个Token），KV-cache的大小约为：

$$ \text{KV-cache大小} = 2 \times L \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{bytes\_per\_param} $$

以Llama 3 70B为例：
- $L = 10,000$（序列长度）
- $n_{\text{layers}} = 80$
- $n_{\text{heads}} = 64$
- $d_{\text{head}} = 128$
- FP16格式：2 bytes

$$ \text{大小} = 2 \times 10000 \times 80 \times 64 \times 128 \times 2 = 2.6 \times 10^{10} \text{ bytes} \approx \textbf{26 GB} $$

这意味着：**仅KV-cache就需要26GB显存**。当并发处理多个请求时，KV-cache的显存需求线性增长——10个并发请求需要260GB，远超单张GPU的容量（H100为80GB）。

---

## 11.4 采样：随机性的艺术

### 11.4.1 Temperature、Top-p、Top-k

LLM的输出不是确定性的。在每个生成步骤，模型输出下一个Token的概率分布：

$$ P(x_t | x_{<t}) = \text{softmax}(z_t / \tau) $$

其中 $z_t$ 是logits（模型最后一层的输出分数），$\tau$ 是**温度（Temperature）** 参数。

**Temperature的作用：**

| $\tau$ 值 | 效果 | 适用场景 |
|:---------:|:----|:---------|
| $\tau \to 0$ | 退化为贪心解码，每次选概率最高的Token | 数学、代码、事实问答 |
| $\tau = 1.0$ | 保持原始分布 | 一般对话 |
| $\tau > 1.0$ | 分布变\"平\"，低概率Token被提升 | 创意写作、头脑风暴 |
| $\tau \to \infty$ | 退化为均匀采样，完全随机 | 少数探索场景 |

---

> **🔬 数学深潜：Softmax温度参数的熵分析**

> 温度参数对模型输出的**信息熵**有直接控制作用。
>
> 定义输出分布 $P_\tau$ 的熵为：
>
> $$H(P_\tau) = -\sum_{i=1}^{V} p_i^{(\tau)} \log p_i^{(\tau)}$$
>
> 其中 $p_i^{(\tau)} = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$。
>
> **为什么 $\tau \to 0$ 退化为贪心解码？**
>
> 当 $\tau \to 0$ 时，最大logit $z_{\max}$ 对应的概率 $p_{\max}^{(\tau)} \to 1$，其他所有概率 $\to 0$。因为：
>
> $$\frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)} = \frac{1}{\sum_j \exp((z_j - z_i)/\tau)}$$
>
> 当 $\tau \to 0$，如果 $z_i < z_{\max}$，则 $(z_i - z_{\max})/\tau \to -\infty$，$\exp((z_i - z_{\max})/\tau) \to 0$，概率趋近于0。
>
> **为什么 $\tau \to \infty$ 退化为均匀采样？**
>
> 当 $\tau \to \infty$ 时，$\exp(z_i / \tau) \to \exp(0) = 1$，所有Token的概率趋近于 $1/V$——均匀分布。
>
> **熵随温度的变化曲线：**
>
> - $\tau = 0$：$H = 0$（确定性，无信息熵）
> - $\tau = 0.5$：$H \approx 0.3-0.5$（高度集中）
> - $\tau = 1.0$：$H \approx 1.0-2.0$（典型对话分布）
> - $\tau = 2.0$：$H \approx 3.0-5.0$（较分散）
> - $\tau \to \infty$：$H \to \log V$（最大熵）
>
> 对于 $V=128,000$，$\log V \approx 11.8$——这是理论最大熵，但实际中几乎不会用到。

**Top-p（Nucleus Sampling）：**

只从累积概率达到 $p$ 的最小Token集合中采样。例如，$p=0.9$ 意味着只从"覆盖总概率90%的最可能Token"中采样。

**Top-k：**

只从概率最高的 $k$ 个Token中采样。例如，$k=50$ 意味着每一步只从Top-50的Token中选择。

**什么时候用哪个？**

- **Temperature**：控制输出的\"创造性\"（低→更确定，高→更多样）
- **Top-p**：自适应地去除\"长尾\"低概率Token（避免生成稀奇怪异的输出）
- **Top-k**：硬性地限制候选Token数量（适合有明显正确答案的场景）

实践中，**Temperature + Top-p** 是最常用的组合。

---

## 11.5 上下文窗口的极限

### 11.5.1 为什么有上下文限制？

LLM的上下文窗口受到两个根本性的限制：

**1. 注意力计算的 $O(n^2)$ 复杂度：**

当序列长度 $n$ 翻倍时，注意力计算量 $QK^T$ 翻4倍：

$$ \text{计算量} = 2 \times n^2 \times d_{\text{head}} \times n_{\text{heads}} $$

对于 $n=128K$ 的序列，注意力分数矩阵 $\in \mathbb{R}^{128K \times 128K}$，每个元素2字节（FP16），存储需要：

$$ (128 \times 10^3)^2 \times 2 \text{ bytes} = 3.28 \times 10^{10} \text{ bytes} \approx \textbf{32 GB} $$

仅一个注意力分数矩阵就需要32GB——这还不算模型参数和其他中间结果。

**2. 位置编码的外推能力：**

如果模型只在 $n_{\text{train}}$ 长度上训练过位置编码（如RoPE），它在 $n > n_{\text{train}}$ 时面临未见过的位置——位置编码的外推性成为瓶颈。

---

> **🔬 数学深潜：注意力矩阵的内存消耗与RoPE的外推能力**

> **注意力分数矩阵的内存消耗：**
>
> 标准注意力 $S = QK^T / \sqrt{d_k}$ 中，$S \in \mathbb{R}^{n \times n}$。对于不同的 $n$：
>
> | n | 注意力分数矩阵大小 | 典型显存占比 |
> |:--:|:----------------:|:----------:|
> | 4K | 32 MB | ~0.04% |
> | 32K | 2 GB | ~2.5% |
> | 128K | 32 GB | ~40% |
> | 1M | 2 TB | 远超单GPU |
>
> FlashAttention通过不显式计算完整的 $S$ 矩阵，将HBM访问从 $O(n^2)$ 降低到 $O(n)$。这也是为什么长上下文推理必须依赖FlashAttention。
>
> **RoPE的外推能力：**
>
> RoPE的核心性质是注意力分数只依赖于相对位置 $m-n$。但问题是：如果训练时最大相对位置是 $n_{\text{train}}$，模型从没见过相对位置 $> n_{\text{train}}$ 的旋转角度组合。
>
> 一个直观的类比：想象你只见过0-100岁的人，让你预估150岁人的状态——你可能能大致推断，但准确性下降。
>
> **ALiBi为什么比RoPE在长上下文上更稳定？**
>
> ALiBi直接使用线性偏置项 $-m \cdot |i-j|$ 来编码位置，其中 $m$ 是head-specific的固定斜率。因为偏置项与训练序列长度完全无关，ALiBi在没有任何额外训练的情况下就能外推到训练时2-4倍的序列长度。而RoPE需要YaRN或位置插值等微调技术才能实现外推。
>
> 但RoPE的优势在于：它通过旋转操作编码了相对位置的结构化信息，在训练长度内的表现通常优于ALiBi。这又是一个\"训练性能 vs 外推能力\"的权衡。

---

## 11.6 Token的未来

### 11.6.1 多模态Token

当前的主流趋势是将**所有模态都编码为Token**：

- **文本Token**：BPE子词
- **图像Token**：ViT将图像切分为Patch Token
- **音频Token**：将音频信号编码为Token序列
- **视频Token**：时空Patch Token
- **代码Token**：BPE直接处理代码文本
- **工具调用Token**：特殊的\"Tool Call\" / \"Tool Response\" Token

### 11.6.2 Token压缩

Token数量直接决定成本，因此Token压缩成为一个活跃的研究方向：
- **多Token预测（Multi-Token Prediction）**：Meta最新研究显示，让模型同时预测未来多个Token可以提升训练效率
- **语义Token**：用更高语义粒度的Token代替子词Token，压缩序列长度
- **混合粒度Token**：根据上下文动态选择Token粒度

---

## 本章小结

Token是理解LLM的\"第一性原理\"，它在看似简单的表面上掩盖了深刻的技术细节：

- **BPE分词**通过贪心合并算法学习子词词表，Trie树实现保证了高效的 $O(L)$ 编码
- **嵌入矩阵**的SVD分解揭示了语义容量的结构——前10%的奇异值占总能量的60-70%
- **Prefill vs Decode**两阶段的推理特性完全不同——前者计算密集，后者内存密集
- **KV-cache**在长序列下需要数十GB显存，是推理成本的核心
- **Temperature、Top-p、Top-k**控制着输出的创造性，温度参数的熵分析清晰地展示了其作用机理
- **注意力矩阵**在128K上下文下需要32GB存储，FlashAttention是长上下文的关键技术

从\"Token\"这个最小单位出发，我们能更深入地理解LLM的每个设计决策——这正是\"第一性原理\"思维的力量。

---

> **本章参考文献**：
> - Sennrich et al. \"Neural Machine Translation of Rare Words with Subword Units.\" ACL 2016. (BPE)
> - Kudo & Richardson. \"SentencePiece: A simple and language independent subword tokenizer.\" 2018.
> - Devlin et al. \"BERT: Pre-training of Deep Bidirectional Transformers.\" NAACL 2019. (WordPiece)
> - Vaswani et al. \"Attention Is All You Need.\" NeurIPS 2017.
> - Holtzman et al. \"The Curious Case of Neural Text Degeneration.\" ICLR 2020. (Top-p sampling)
> - Press et al. \"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation.\" ICLR 2022. (ALiBi)
> - Su et al. \"RoFormer: Enhanced Transformer with Rotary Position Embedding.\" 2021. (RoPE)
> - Dao et al. \"FlashAttention: Fast and Memory-Efficient Exact Attention.\" NeurIPS 2022.
