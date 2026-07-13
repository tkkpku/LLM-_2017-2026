# 第12章 工具的延伸——Function Calling、MCP与Agent

> "语言是思想的延伸，工具是行动的延伸。当LLM学会使用工具，它从'思考者'变成了'行动者'。" —— 本书作者

---

> **🗺️ 本章地图**
>
> | 项目 | 内容 |
> |:----|:------|
> | **核心问题** | ① LLM如何学会调用工具？Function Calling的底层原理是什么？② ReAct循环如何让Agent自主完成多步任务？③ MCP协议的设计哲学是什么？④ Agent常见的失败模式有哪些？如何设计容错机制？⑤ 什么时候需要多Agent编排？ |
> | **前置知识** | 基本的LLM推理原理、JSON数据结构、HTTP/RPC协议基础 |
> | **阅读时间** | 45-55分钟 |
> | **关键词** | Function Calling、ReAct、MCP、Agent、Grammar-constrained decoding、POMDP、状态机 |

---

## 12.1 Function Calling的底层原理

### 12.1.1 什么是Function Calling？

**Function Calling（函数调用）** 是LLM与外部世界交互的桥梁。它允许模型在生成过程中，不仅输出文本，还可以输出一个结构化的"工具调用请求"——告诉调用者："请帮我执行这个函数，把结果给我"。

一个典型的Function Calling流程：

```
用户：明天北京的天气怎么样？
模型内部推理：我需要调用 get_weather(location="北京", date="明天")
模型输出：<function_call> {"name": "get_weather", "arguments": {"location": "北京", "date": "2026-07-14"}}
系统执行函数 → 结果：{"temperature": 32, "condition": "晴", "humidity": 45%}
模型继续生成：明天北京天气晴朗，最高温度32°C...
```

### 12.1.2 函数调用的底层机制

从底层看，Function Calling本质上是一种**格式约束下的自回归生成**。

训练时，模型在数据中看到这样的格式：

```
用户：明天北京天气如何？
助手：让我查一下天气。<tool_call> get_weather(location="北京", date="2026-07-14") </tool_call>
系统：{"temperature": 32, "condition": "晴", "humidity": "45%"}
助手：明天北京天气晴朗，最高温度32°C...
```

模型学会了：在某些情况下（需要外部信息时），不直接输出答案，而是输出一个特殊的"工具调用"格式。

推理时，通过**Grammar-constrained decoding（语法约束解码）** 来确保生成的JSON格式正确——当模型预测到需要输出工具调用时，token预测被限制在符合JSON Schema的token范围内。

---

> **🔬 数学深潜：Grammar-constrained Decoding**

> Grammar-constrained decoding的核心思想是**用有限状态自动机（FSA）约束token预测的每一步**。
>
> 假设工具接受一个JSON对象 `{"location": string, "date": string}`。我们需要构建一个FSA，其状态转换图如下：
>
> ```
> 状态0: 初始 → 遇到 "{" 进入状态1
> 状态1: 刚输入 "{" → 只能输入 "location" 或 "date" 
> 状态2: 输入 "location" → 只能输入 ":"
> 状态3: 输入 ":" → 只能输入字符串
> 状态4: 输入字符串值 → 只能输入 "," 或 "}"
> ...（以此类推）
> ```
>
> 在每一步解码中，模型计算所有token的概率分布 $P(\text{token} | \text{context})$，然后**将不符合当前FSA状态的token概率设为0**（mask），只从合法token中采样：
>
> $$P_{\text{masked}}(\text{token}_i) = \begin{cases} P(\text{token}_i) & \text{if FSA允许token}_i \\ 0 & \text{otherwise} \end{cases}$$
>
> **工具描述的语义嵌入：** 工具的name和description通过嵌入层映射到向量空间中，作为Prefix的一部分参与注意力计算。这就是为什么工具名和描述影响调用成功率——更好的描述能让模型更准确地理解工具的用途。研究表明，在工具名中包含动词（如"search_web" vs "web"）可以提升调用准确率约15%。

---

### 12.1.3 JSON Schema的作用

Function Calling的核心是**JSON Schema**——一个描述函数参数格式的标准：

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "获取指定地点和时间的天气信息",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "城市名，如'北京'、'上海'"
        },
        "date": {
          "type": "string",
          "description": "日期，格式YYYY-MM-DD"
        }
      },
      "required": ["location", "date"]
    }
  }
}
```

JSON Schema不仅描述了参数的格式，还通过`description`字段为模型提供了**语义提示**——帮助模型理解每个参数的含义。

---

## 12.2 ReAct循环详解

### 12.2.1 什么是ReAct？

**ReAct（Reasoning + Acting）** 是由Yao等人（2023）提出的Agent框架，将**推理**和**行动**交替进行。与Function Calling的一次调用不同，ReAct允许Agent进行多轮推理-行动循环。

典型的ReAct循环：

```
Thought: 用户想知道明天北京的天气。我需要先查询天气信息。
Action: get_weather(location="北京", date="2026-07-14")
Observation: {"temperature": 32, "condition": "晴", "humidity": "45%"}
Thought: 天气信息已获取。明天北京晴天，温度32°C。用户可能还想知道是否需要带伞——湿度45%不算高，不需要。
Action: 无（直接回答）
Final: 明天北京天气晴朗，最高32°C，湿度适中，适合外出活动。
```

### 12.2.2 ReAct的形式化

---

> **🔬 数学深潜：ReAct的严格形式化**

> ReAct循环可以建模为一个**部分可观测马尔可夫决策过程（POMDP）**：
>
> - **状态 $s_t$**：环境（如代码库、数据库、互联网）的真实状态
> - **观测 $o_t$**：Agent获取到的信息（如函数返回结果）
> - **行动 $a_t$**：Agent选择执行的工具调用
> - **策略 $\pi(a_t | h_t)$**：LLM根据历史 $h_t = (o_1, a_1, ..., o_t)$ 决定下一步行动
>
> ReAct的每个步骤的条件概率：
>
> $$P(\text{action}_t | \text{thought}_t, \text{history}) = \text{LLM}(\text{[history]} + \text{[thought}_t\text{]} + \text{[action}_t\text{]})$$
>
> 其中history包含所有之前的Thought-Action-Observation三元组。
>
> **Agent循环中上下文窗口的管理：**
>
> 当Agent执行了20步ReAct循环后，history的长度可能达到10,000+ Token。由于上下文窗口有限，需要一种**截断策略**：
>
> 1. **滑动窗口**：只保留最近 $K$ 步的history（简单但可能丢失关键上下文）
> 2. **摘要压缩**：将早期history总结为一段摘要（信息损失但保留关键信息）
> 3. **分层记忆**：长期记忆（压缩摘要）+ 短期记忆（完整最近几步）

---

## 12.3 MCP协议架构

### 12.3.1 什么是MCP？

**MCP（Model Context Protocol）** 是由Anthropic于2024年底提出的开放协议，旨在标准化LLM与外部工具和数据源的交互方式。

MCP的三层架构：

```
┌─────────────────────────────────────┐
│              LLM应用                  │  ← MCP Client（如Chatbox、Claude Desktop）
├─────────────────────────────────────┤
│           MCP协议（JSON-RPC）          │
├──────────┬──────────────┬────────────┤
│ 文件系统   │  数据库      │  Web搜索    │  ← MCP Server
│ MCP Server│ MCP Server  │ MCP Server  │
└──────────┴──────────────┴────────────┘
```

**MCP与Function Calling的区别：**

| 维度 | Function Calling | MCP |
|:----|:---------------|:-----|
| **协议类型** | API调用约定 | **开放协议**（类似HTTP对于Web） |
| **工具发现** | 静态注册（代码中定义） | **动态发现**（Server列举可用工具） |
| **工具范围** | 单个工具集合 | **任意数量Server**，每个Server提供多工具 |
| **安全性** | 调用方自行管理 | **标准化权限模型**（资源授权） |
| **传输层** | HTTP | JSON-RPC（支持stdio/HTTP/WebSocket） |
| **生态互操作性** | 低（每个API独立） | **高**（任何MCP Client可用任何MCP Server） |

### 12.3.2 MCP的设计哲学

---

> **📖 进阶阅读：MCP协议的设计哲学**

> **为什么选择JSON-RPC而非gRPC？**
>
> 1. **人类可读性**：JSON直观易懂，便于开发者调试。gRPC使用Protocol Buffers二进制格式，调试需要额外工具。
> 2. **动态类型**：LLM本身就是处理文本的，JSON的灵活性更适合LLM生态。
> 3. **降低门槛**：任何会Python/JavaScript的开发者都能在10分钟内写一个MCP Server。
> 4. **代价**：JSON序列化/反序列化的性能开销约比Protobuf多5-10倍——但这对于一次LLM调用（毫秒到秒级）来说可以忽略。
>
> **为什么工具发现是动态的而不是静态注册？**
>
> 静态注册意味着在编译时就确定所有可用的工具。这在传统API设计中是合理的，但对于LLM应用，动态发现有独特优势：
>
> - **热更新**：可以在不重启LLM应用的情况下，添加新的MCP Server/工具
> - **按需加载**：用户可以根据当前任务，动态连接不同的MCP Server
> - **权限细化**：每个MCP Server可以独立管理自己的授权策略

### 12.3.3 MCP工具调用的安全性模型

工具调用带来了新的安全风险。MCP的安全性模型设计了三层防护：

**第一层：权限最小化**

每个MCP Server声明自己需要的权限（只读文件系统？只能访问特定目录？）。LLM应用在连接Server时，拒绝超越权限的请求。

**第二层：沙箱隔离**

MCP Server运行在独立的进程/容器中，无法访问宿主系统的敏感资源。即使Server代码被恶意利用，影响也局限在沙箱内。

**第三层：调用审计**

所有工具调用都被记录日志，包括：调用者、参数、时间戳、返回结果摘要。便于事后审计和异常检测。

---

## 12.4 错误与恢复

### 12.4.1 Agent的失败模式

Agent在自主执行任务时，会面临多种失败模式。以下是Agent失败模式的分类学树：

```
Agent失败模式
├── 推理层错误
│   ├── 任务理解偏差：Agent误解了用户的真实目标
│   ├── 计划过度复杂：Agent生成10步计划，但前3步就能完成
│   └── 循环推理：Agent反复思考同一个问题，不采取行动
├── 工具调用错误
│   ├── 工具错选：应该用搜索，却调用了计算器
│   ├── 参数错误：调用搜索时传了空字符串
│   ├── 参数幻觉：虚构了不存在的参数
│   └── 格式错误：生成的JSON格式不正确
├── 执行层错误
│   ├── 工具超时：外部API 5秒无响应
│   ├── 工具返回错误：数据库查询失败
│   └── 数据不一致：搜索返回的结果前后矛盾
├── 上下文管理错误
│   ├── 上下文溢出：历史记录太长，超出模型限制
│   ├── 关键信息丢失：截断历史时删除了后续步骤依赖的信息
│   └── 历史污染：错误行动的后果污染了后续判断
└── 安全相关错误
    ├── Prompt注入：工具返回的内容包含恶意指令
    ├── 权限越级：Agent尝试执行未授权的操作
    └── 数据泄露：Agent将敏感信息写入日志
```

---

> **💻 C++视角：如何用状态机管理Agent生命周期**

> Agent的生命周期可以用一个有限状态机（FSM）来精确建模：
>
> ```cpp
> enum class AgentState {
>     IDLE,          // 等待用户输入
>     THINKING,      // LLM正在生成Thought
>     WAITING_TOOL,  // 等待工具执行结果
>     OBSERVING,     // 处理工具返回结果
>     COMPLETED,     // 任务完成
>     ERROR,         // 发生不可恢复错误
>     MAX_RETRIES    // 超过最大重试次数
> };
>
> class AgentFSM {
>     AgentState state;
>     int retry_count = 0;
>     const int MAX_RETRY = 3;
>
>     // 状态转换函数
>     AgentState transition(AgentState current, Event event) {
>         switch (current) {
>             case IDLE:
>                 if (event == Event::NEW_TASK) return THINKING;
>                 break;
>             case THINKING:
>                 if (event == Event::TOOL_NEEDED) return WAITING_TOOL;
>                 if (event == Event::ANSWER_READY) return COMPLETED;
>                 if (event == Event::ERROR) return handle_error();
>                 break;
>             case WAITING_TOOL:
>                 if (event == Event::TOOL_COMPLETE) return OBSERVING;
>                 if (event == Event::TOOL_TIMEOUT) return handle_timeout();
>                 break;
>             case OBSERVING:
>                 if (event == Event::MORE_STEPS_NEEDED) return THINKING;
>                 if (event == Event::TASK_DONE) return COMPLETED;
>                 break;
>             // ...
>         }
>     }
>
>     AgentState handle_error() {
>         retry_count++;
>         if (retry_count >= MAX_RETRY) return MAX_RETRIES;
>         return THINKING;  // 重试
>     }
> };
> ```
>
> 这种设计使得Agent的行为可预测、可调试、可测试——每个状态转换都有明确的触发条件和处理逻辑。

---

## 12.5 多Agent编排

### 12.5.1 单Agent vs 多Agent

| 维度 | 单Agent | 多Agent |
|:----|:--------|:--------|
| **复杂度** | 低 | 高（需要协调） |
| **并行度** | 串行 | 可并行 |
| **错误隔离** | 单点故障 | 一个Agent出错不影响其他 |
| **灵活性** | 有限（单个模型的能力边界） | 高（不同专业Agent协同） |
| **通信开销** | 无 | 高（Agent间消息传递） |

### 12.5.2 多Agent的编排模式

**模式一：主管-员工（Orchestrator-Worker）**

一个\"主管Agent\"负责任务分解和分配，多个\"员工Agent\"各自执行子任务。

**模式二：辩论式（Debate）**

多个Agent扮演不同角色（正方、反方），通过辩论来探索问题的不同维度。

**模式三：流水线式（Pipeline）**

Agent A的输出作为Agent B的输入，形成处理流水线。

---

> **📖 进阶阅读：多Agent通信的熵分析**

> 多Agent系统的效率与Agent间通信的**熵**密切相关。
>
> 定义消息的熵为：
> $$H(M) = -\sum_{m} P(m) \log P(m)$$
>
> 当消息高度可预测时（Agent A总是说\"OK\"），熵很低——信息量很少。
> 当消息高度不可预测时（Agent A给出复杂分析），熵很高——信息量丰富。
>
> **多Agent系统的设计原则：**
>
> 1. **减少高熵通信**：不要让Agent频繁交换长消息。短消息（状态码、分值）比长篇分析更高效。
> 2. **信息蒸馏**：让Agent在发送消息前先\"蒸馏\"——提取关键信息，而不是发送原始上下文。
> 3. **异步通信**：Agent A可以继续工作，而不是阻塞等待Agent B的响应。
>
> **决策树：什么时候需要多Agent？**
>
> ```
> 任务需要同时涉及多个专业知识领域？
>   ├── 否 → 单Agent即可
>   └── 是 → 
>       任务可以被清晰分解为独立子任务？
>       ├── 是 → 流水线式多Agent ✓
>       └── 否 →
>            任务需要对比不同视角的意见？
>            ├── 是 → 辩论式多Agent ✓
>            └── 否 → 主管-员工模式 ✓
> ```

---

## 本章小结

Function Calling、MCP和Agent代表了LLM从\"聊天机器人\"到\"数字员工\"的进化：

- **Function Calling**通过Grammar-constrained decoding确保工具调用的格式正确性
- **ReAct循环**将Agent建模为POMDP，thought→action→observation的三元组构成了自主决策的基本单元
- **MCP协议**通过动态工具发现、JSON-RPC传输和三层安全模型，为工具调用提供了标准化基础设施
- **Agent失败模式**的10+种分类告诉我们：Agent设计不仅是\"让AI更聪明\"，更是\"让AI更可靠\"
- **多Agent编排**在合适场景下可以显著提升任务处理能力

当我们能可靠地让LLM使用工具时，它就从一个\"会说话的模型\"变成了一个\"能办事的助手\"——这是通向真正实用AI的关键一步。

---

> **本章参考文献**：
> - Yao et al. \"ReAct: Synergizing Reasoning and Acting in Language Models.\" ICLR 2023.
> - Schick et al. \"Toolformer: Language Models Can Teach Themselves to Use Tools.\" NeurIPS 2023.
> - Qin et al. \"Tool Learning with Foundation Models.\" arXiv 2023.
> - Anthropic. \"The Model Context Protocol.\" 2024.
> - LangChain. \"LangChain: Building Applications with LLMs through Composability.\" 2022.
> - Wang et al. \"A Survey on Large Language Model based Autonomous Agents.\" 2023.
> - Park et al. \"Generative Agents: Interactive Simulacra of Human Behavior.\" UIST 2023.
> - Xi et al. \"The Rise and Potential of Large Language Model Based Agents.\" 2023.
