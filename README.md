# llm-interview

本仓库收录 AI / LLM 工程面试题与参考答案。这是 cluster1900 的独立项目，每天新增 AI/LLM 相关题目。题号形如 `RAG-012`：前缀为章节代码，后三位为章内序号；中英文版本共用同一套题号，新题追加在章末，已发布题号不再改动。

AI / LLM engineering interview questions with reference answers. This is cluster1900's own project, maintained independently. New AI/LLM-related questions are added daily. Question IDs look like `RAG-012`: a chapter code plus a three-digit sequence within the chapter. Chinese and English share the same IDs; new questions are appended to the end of a chapter and published IDs never change.

每个问题都按五层组织：

- **答**：直接回答面试问题，给出定义、机制、工程做法或设计方案。
- **深入展开**：提示面试时应继续补充的工程细节，例如数据流、指标、成本、边界条件、风险和验证方式。
- **后续追问**：列出面试官可能继续追问的方向，帮助把单点问题扩展成系统化讨论。
- **扩展问题**：把原问题扩展成几个更接近真实面试的追问，覆盖概念边界、生产约束、故障模式和验证方式。
- **推理回答**：给出可公开表达的推理路径，说明如何从目标、约束、风险、指标和工程边界推导出答案。

Each question is organized in five layers:

- **Answer**: Directly answer the interview question with a definition, mechanism, engineering practice, or design.
- **Deep dive**: Engineering details to add in an interview, such as data flow, metrics, cost, edge cases, risks, and how to verify.
- **Follow-ups**: Directions an interviewer may pursue next, so a single point can grow into a systematic discussion.
- **Extended questions**: Stretch the original question toward a real interview, covering concept boundaries, production constraints, failure modes, and verification.
- **Reasoning**: A publicly expressible reasoning path from goals, constraints, risks, metrics, and engineering boundaries to the answer.

## 目录 / Contents

| # | 代码 Code | 中文 | English | 题数 Count |
|---:|---|---|---|---:|
| 01 | `ARCH` | [模型原理：Transformer 与语言模型](zh/01-model-internals.md) | [Model Internals: Transformers and Language Models](en/01-model-internals.md) | 45 |
| 02 | `GEN` | [生成、解码与长上下文](zh/02-decoding-context.md) | [Decoding, Context and Long Context](en/02-decoding-context.md) | 28 |
| 03 | `TRAIN` | [预训练、微调与分布式训练](zh/03-training-adaptation.md) | [Pre-training, Fine-tuning and Distributed Training](en/03-training-adaptation.md) | 32 |
| 04 | `ALIGN` | [对齐、后训练与推理模型](zh/04-alignment-post-training.md) | [Alignment, Post-training and Reasoning Models](en/04-alignment-post-training.md) | 24 |
| 05 | `PROMPT` | [提示与上下文工程](zh/05-prompt-context-engineering.md) | [Prompting and Context Engineering](en/05-prompt-context-engineering.md) | 30 |
| 06 | `EMB` | [Embedding 与向量检索](zh/06-embeddings-vector-search.md) | [Embeddings and Vector Search](en/06-embeddings-vector-search.md) | 28 |
| 07 | `RAG` | [检索增强生成](zh/07-rag.md) | [Retrieval-Augmented Generation](en/07-rag.md) | 42 |
| 08 | `AGENT` | [Agent、工具调用与 MCP](zh/08-agents-tools-mcp.md) | [Agents, Tool Use and MCP](en/08-agents-tools-mcp.md) | 57 |
| 09 | `MM` | [多模态与语音](zh/09-multimodal-voice.md) | [Multimodal AI and Voice](en/09-multimodal-voice.md) | 35 |
| 10 | `SERVE` | [推理优化与模型服务](zh/10-inference-serving.md) | [Inference Optimization and Serving](en/10-inference-serving.md) | 61 |
| 11 | `OPS` | [LLMOps 与生产可靠性](zh/11-llmops-reliability.md) | [LLMOps and Production Reliability](en/11-llmops-reliability.md) | 40 |
| 12 | `EVAL` | [评估与测试](zh/12-evaluation.md) | [Evaluation and Testing](en/12-evaluation.md) | 41 |
| 13 | `SAFE` | [安全、隐私与治理](zh/13-safety-privacy-governance.md) | [Safety, Privacy and Governance](en/13-safety-privacy-governance.md) | 53 |
| 14 | `DESIGN` | [系统设计案例](zh/14-system-design-cases.md) | [System Design Cases](en/14-system-design-cases.md) | 43 |
| 15 | `CODE` | [手写实现](zh/15-hands-on.md) | [Hands-on Implementation](en/15-hands-on.md) | 32 |
| 16 | `CAREER` | [工程判断与沟通](zh/16-judgment-communication.md) | [Engineering Judgment and Communication](en/16-judgment-communication.md) | 22 |

## 许可证 / License

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。

This project is licensed under the MIT License; see [LICENSE](LICENSE).
