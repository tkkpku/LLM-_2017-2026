# 第13章 知识与记忆——RAG、搜索与知识库的工程实践

> "知识不是记忆——知道去哪里找，比记住更重要。RAG就是这个'知道去哪里找'的系统化方法。" —— 本书作者

---

> **🗺️ 本章地图**
>
> | 项目 | 内容 |
> |:----|:------|
> | **核心问题** | ① 嵌入模型如何将文本转化为可搜索的向量？② 分块策略如何影响检索质量？③ HNSW如何实现对数级别的近似最近邻搜索？④ 混合检索为什么比纯向量搜索更强大？⑤ 高级RAG模式（HyDE、Agentic RAG）的工作原理是什么？ |
> | **前置知识** | 基本的线性代数（向量、余弦相似度）、第2章关于FlashAttention的知识 |
> | **阅读时间** | 40-50分钟 |
> | **关键词** | 嵌入模型、InfoNCE、分块策略、HNSW、IVF、BM25、混合检索、Cross-encoder、RAG、HyDE |

---

## 13.1 嵌入模型的选型

### 13.1.1 什么是嵌入模型？

**嵌入模型（Embedding Model）** 是一种将文本（或图像、音频）映射到固定维度向量空间的模型。与LLM不同，嵌入模型的核心目标不是生成文本，而是**语义编码**——将语义相似的输入映射到向量空间中相近的位置。

常用的嵌入模型：
- **OpenAI text-embedding-3**（1536维，2024年发布）
- **BGE（BAAI General Embedding）**（多种尺寸，开源）
- **E5 / E5-mistral**（微软，支持多语言）
- **Cohere Embed**（多语言支持好）
- **Jina Embeddings**（长文本优化）

### 13.1.2 对比学习与InfoNCE损失

嵌入模型一般通过**对比学习（Contrastive Learning）** 来训练。核心思想是：让语义相似的文本对（正样本）在向量空间中靠近，语义不同的文本对（负样本）远离。

---

> **🔬 数学深潜：对比学习损失函数的InfoNCE推导**

> InfoNCE损失是嵌入模型训练的核心：
>
> $$ \mathcal{L} = -\log \frac{\exp(\text{sim}(q, k^+) / \tau)}{\sum_{i=1}^{N} \exp(\text{sim}(q, k_i) / \tau)} $$
>
> 其中 $\text{sim}(q, k)$ 是余弦相似度，$\tau$ 是温度参数。
>
> **InfoNCE是互信息的下界——详细推导：**
>
> 定义随机变量 $X$（查询）和 $Y$（键），我们希望最大化互信息 $I(X; Y)$：
>
> $$I(X; Y) = H(X) - H(X|Y) = \mathbb{E}_{p(x,y)} \left[ \log \frac{p(x|y)}{p(x)} \right]$$
>
> 由于我们不知道真实的 $p(x|y)$，用一个参数化的模型 $f_\theta(x, y)$（即我们的嵌入模型）来近似：
>
> $$p(x|y) \approx \frac{f_\theta(x, y)}{\sum_{x' \in \mathcal{X}} f_\theta(x', y)}$$
>
> 代入并取对数，InfoNCE损失正是这个近似下界的负对数——最小化InfoNCE等价于最大化互信息的下界。
>
> **温度参数 $\tau$ 的作用：**
>
> $\tau$ 控制了对负样本的惩罚力度。当 $\tau$ 较小时，注意力集中在一个小区域——对困难负样本（与查询较相似的负样本）的区分要求更高。当 $\tau$ 较大时，对所有负样本的惩罚趋于平均。
>
> **维度选择的数学——高维诅咒：**
>
> 高维空间中，所有向量对的点积趋于0（集中现象），导致嵌入模型难以区分\"相似\"和\"不相似\"。而维度太低，语义容量不足。最优维度通常在768-1536之间——这是\"语义容量\"和\"可区分性\"的权衡。

---

## 13.2 分块策略

### 13.2.1 为什么需要分块？

LLM的上下文窗口有限（即使扩展到128K），且检索系统的效率随文档增大而下降。因此，需要将长文档**分块（Chunking）** 为较小的片段。

### 13.2.2 分块策略对比

| 策略 | 方法 | 优点 | 缺点 | 适用场景 |
|:----|:-----|:----|:----|:---------|
| 固定大小分块 | 按固定Token数切分 | 简单、可预测 | 可能切断语义连续的句子 | 简单文档 |
| 递归字符分块 | 按段落/句子递归切分 | 语义完整 | 块大小不一 | 一般文档 |
| 语义分块 | 使用嵌入模型检测语义边界 | 语义完整性最好 | 计算成本高 | 高质量知识库 |
| 基于文档结构 | 按章节、标题切分 | 自然语义单元 | 依赖文档格式 | 结构化文档（论文、手册） |

---

> **💻 C++视角：字符串分割的算法复杂度**

> **固定大小分块（$O(n)$）：**
>
> ```cpp
> // 按固定Token数切分
> vector<string> fixed_chunk(const vector<int>& tokens, int chunk_size) {
>     vector<string> chunks;
>     for (int i = 0; i < tokens.size(); i += chunk_size) {
>         int end = min(i + chunk_size, (int)tokens.size());
>         chunks.push_back(decode(tokens.begin() + i, tokens.begin() + end));
>     }
>     return chunks;
> }
> // 时间复杂度：O(n)，空间复杂度：O(n)
> ```
>
> **语义分块（$O(n \log n)$）：**
>
> 语义分块需要计算相邻句子之间的嵌入相似度，然后使用\"变化点检测\"算法寻找语义边界。这涉及到：
> - 计算 $m$ 个句子的嵌入：$O(m \cdot d)$（$d$ 为嵌入维度）
> - 在句子间检测相似度下降点：通常使用二分搜索+动态规划，$O(m \log m)$
> - 综上：$O(n + m \log m)$，其中 $n$ 为字符数，$m$ 为句子数

---

## 13.3 向量检索的算法

### 13.3.1 从精确搜索到近似搜索

在检索少量文档（<10万）时，可以使用**暴力精确搜索**——计算查询向量与所有文档向量的余弦相似度，取Top-K。

但当文档数量达到百万甚至十亿级别时，暴力搜索的 $O(Nd)$ 时间不可接受。这时需要**近似最近邻搜索（ANN, Approximate Nearest Neighbor）**。

### 13.3.2 IVF：倒排文件索引

**IVF（Inverted File Index）** 是最直观的ANN方法：
1. 对文档向量进行聚类（通常使用K-means），得到 $k$ 个聚类中心
2. 每个文档向量被分配到最近的聚类中心
3. 检索时，只搜索与查询最近的几个聚类中心内的文档

### 13.3.3 HNSW：分层可导航小世界图

---

> **🔬 数学深潜：HNSW的图论基础**

> **HNSW（Hierarchical Navigable Small World）** 是当前最主流的ANN算法，被FAISS、Milvus、Pinecone等向量数据库广泛使用。
>
> **核心思想：多层级图结构**
>
> HNSW构建了多个层级的图，底层包含所有向量点，顶层只包含少量\"枢纽\"点：
>
> ```
> Level 2: ● ── ● ── ●         （~1%的向量，粗粒度导航）
>            ↓    ↓    ↓
> Level 1: ● ── ● ── ● ── ●    （~10%的向量，中粒度）
>            ↓    ↓    ↓    ↓
> Level 0: ● ── ● ── ● ── ● ── ● （100%的向量，细粒度搜索）
> ```
>
> **搜索过程：**
>
> 从顶层开始，在每一层找到与查询最近的节点，然后下降一层继续搜索。这类似于二分搜索——先快速定位到大致区域，再逐步细化。
>
> **复杂度分析：**
>
> HNSW的搜索复杂度为 $O(\log N)$——与二分查找处于同一量级。在1亿个向量中搜索，只需要大约20-30次距离计算。
>
> **为什么分层导航有效？** 这是一个\"尺度的层次\"问题：在顶层看全局，在底层看细节。与人类寻找目标的方式如出一辙——先确定\"在哪个城市\"，再确定\"在哪条街\"，最后找到\"具体门牌号\"。

---

## 13.4 混合检索与重排序

### 13.4.1 词汇搜索的复兴：BM25

纯向量搜索有一个理论上的弱点：它依赖嵌入模型的语义理解能力，在精确匹配（如产品编号、人名）上不如传统的词汇搜索。

**BM25** 是词汇搜索的最强基线：

---

> **🔬 数学深潜：BM25的数学原理**

> BM25是TF-IDF的改进版本。对于查询中的词 $q$ 和文档 $d$：
>
> $$\text{BM25}(q, d) = \sum_{i=1}^{n} \frac{\text{TF}(q_i, d) \cdot (k_1 + 1)}{\text{TF}(q_i, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})} \cdot \log \frac{N - \text{DF}(q_i) + 0.5}{\text{DF}(q_i) + 0.5}$$
>
> **BM25对TF-IDF的三个改进：**
>
> 1. **词频饱和**：$k_1$ 参数（通常1.2-2.0）控制了词频的边际收益。在TF-IDF中，一个词出现10次的得分是出现1次的10倍。在BM25中，词频的贡献被\"压缩\"——10次可能是1次的2.5倍。这符合直觉：搜索\"机器学习\"，一篇文章中出现10次不意味着相关度是出现1次的10倍。
>
> 2. **文档长度归一化**：$b$ 参数（通常0.75）控制了对长文档的惩罚。长文档更可能\"恰好\"包含查询中的词，因此需要一个归一化因子。
>
> 3. **IDF平滑**：$\log \frac{N - \text{DF} + 0.5}{\text{DF} + 0.5}$ 避免了TF-IDF中DF=0导致的除以0问题。

---

### 13.4.2 混合检索：向量+词汇的最佳组合

**混合检索（Hybrid Search）** 将向量搜索的语义理解能力与BM25的精确匹配能力结合：

$$ \text{Score}_{\text{hybrid}}(q, d) = \alpha \cdot \text{Score}_{\text{vector}}(q, d) + (1 - \alpha) \cdot \text{Score}_{\text{BM25}}(q, d) $$

其中 $\alpha$ 是混合权重，通常取0.3-0.7。

### 13.4.3 重排序（Re-ranking）

初次检索（向量搜索或混合搜索）返回Top-50或Top-100候选文档。然后使用**Cross-encoder**对候选文档进行精确重排序——计算查询和每个文档的匹配分数。

**Bi-encoder（嵌入模型） vs Cross-encoder（重排序模型）：**

| 模型类型 | 架构 | 速度 | 精度 | 适用阶段 |
|:--------|:----|:----:|:----:|:--------|
| Bi-encoder | 查询和文档独立编码 | **极快**（可预计算） | 中等 | **第一阶段检索** |
| Cross-encoder | 查询和文档联合编码 | 慢（需实时计算） | **高** | **第二阶段重排序** |

---

## 13.5 RAG的进阶模式

### 13.5.1 HyDE：假设性文档嵌入

**HyDE（Hypothetical Document Embeddings）** 的核心思想是：让LLM根据查询生成一个\"假设性的理想文档片段\"，然后用这个假设文档的嵌入去检索。

为什么有效？有些查询（如\"AI在医疗领域的应用挑战\"）与最佳匹配文档的文本形式差异较大。但LLM生成的假设文档更接近真实文档的文本风格，在嵌入空间中的位置更准确。

### 13.5.2 Agentic RAG

**Agentic RAG** 将RAG与Agent结合，让LLM自主决定：
1. **是否需要检索？** 有些查询不需要外部知识
2. **如何改写查询？** 原始查询可能不适合检索
3. **需要检索几次？** 可能需要多轮检索来收集不同信息
4. **检索结果够了吗？** 自主判断回答是否充分

---

> **📖 进阶阅读：决策树——如何为你的场景选择RAG架构**

> ```
> 你的数据量有多大？
> ├── <1万文档 → 暴力搜索即可（无需ANN）
> └── >1万文档 → 
>     你需要精确关键词匹配吗？
>     ├── 是 → 混合检索（向量+BM25）
>     └── 否 → 纯向量检索（HNSW）
>          
> 你的查询有多复杂？
> ├── 简单事实问答（"2024年GDP是多少"）→ 基本RAG
> └── 复杂多跳推理（"哪些因素导致了2024年的GDP变化"）
>     你需要检索质量高吗？
>     ├── 是 → 混合检索 + Cross-encoder重排序
>     └── 还需要多步检索？
>          └── 是 → Agentic RAG（多跳推理）
> ```

---

## 本章小结

RAG是从\"让模型知道\"到\"让模型能找到\"的关键技术：

- **嵌入模型**通过InfoNCE损失最大化互信息下界，将语义转换为向量空间中的距离
- **分块策略**从 $O(n)$ 的固定分块到 $O(n \log n)$ 的语义分块，需根据场景权衡
- **HNSW**利用分层小世界图实现对 $O(\log N)$ 量级的近似最近邻搜索
- **BM25**通过对词频和文档长度的数学建模，在精确匹配上仍然不可替代
- **混合检索**将向量和词汇搜索结合，Cross-encoder重排序进一步提升精度
- **HyDE**和**Agentic RAG**代表了RAG的进阶方向——让AI更智能地决定如何获取信息

---

> **本章参考文献**：
> - Reimers & Gurevych. \"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.\" EMNLP 2019.
> - Karpukhin et al. \"Dense Passage Retrieval for Open-Domain Question Answering.\" EMNLP 2020. (DPR)
> - Malkov & Yashunin. \"Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.\" TPAMI 2018. (HNSW)
> - Robertson & Zaragoza. \"The Probabilistic Relevance Framework: BM25 and Beyond.\" Foundations and Trends in IR 2009.
> - Lewis et al. \"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.\" NeurIPS 2020.
> - Gao et al. \"Precise Zero-Shot Dense Retrieval without Relevance Labels.\" ACL 2023. (HyDE)
> - Izacard et al. \"Atlas: Few-shot Learning with Retrieval Augmented Language Models.\" 2022.
