# Intelligent Customer Service Agent - Xiao Zhi

一个用于学习 Agent 工作流的中文智能客服项目。项目以 Python 和 LangGraph 实现了问题分类、情绪判断、条件路由、本地 FAQ 检索、人工转接和终端交互。

项目同时保留两条实现路线：

- 规则版：使用关键词完成分类和情绪判断，适合稳定、免费地运行本地测试。
- 大模型版：仅使用 OpenAI 兼容模型完成问题分类；情绪、路由、FAQ 和回复节点继续复用本地规则。

这是一项学习和作品集项目，不是可直接投入生产的客服系统。

## Features

- 明确的 `CustomerState` 状态结构，保存 `query`、`category`、`sentiment`、`route` 和 `response`。
- 三类问题分类：`technical`、`billing`、`general`。
- 三类情绪判断：`positive`、`neutral`、`negative`。
- 负面情绪优先转人工客服。
- LangGraph 状态图和条件边：人工转接跳过 FAQ，自动路线进入 FAQ 检索。
- 本地 FAQ 知识库：退款时效、客服工作时间和密码重置。
- OpenAI 兼容模型分类器，支持 JSON 输出、分类范围校验及重复 JSON 恢复。
- 模型请求超时、连接失败、服务端错误或非法输出时，自动降级到规则分类，并记录分类来源和错误类型。
- 本地 `unittest` 测试，不在单元测试中发起真实模型请求。

## Architecture

```mermaid
flowchart TD
    Start([START]) --> Categorize[问题分类]
    Categorize --> Sentiment[情绪分析]
    Sentiment --> Route[路由决策]
    Route -->|human_handoff| Response[生成回复]
    Route -->|自动回复路线| FAQ[本地 FAQ 检索]
    FAQ --> Response
    Response --> End([END])
```

规则版将 `categorize_query` 作为分类节点。大模型版将相同位置替换为 `categorize_with_model`：

```text
用户问题
-> OpenAI 兼容模型分类
-> JSON 解析与校验
-> 本地情绪判断、路由、FAQ 和回复

如果模型分类失败，分类节点会改用规则分类器，并继续执行后续本地节点：

```text
模型分类失败
-> 规则分类降级
-> 本地情绪判断、路由、FAQ 和回复
```
```

## Project Structure

```text
src/
  agent.py                Rule nodes, state definition, and manual workflow baseline
  langgraph_agent.py      Rule-based LangGraph workflow
  langgraph_llm_agent.py  LLM-classification LangGraph workflow
  knowledge_base.py       Local FAQ data and keyword-score lookup
  model_config.py         .env loading and model configuration validation
  model_client.py         OpenAI-compatible client construction
  model_probe.py          Minimal live model connection probe
  llm_classifier.py       LLM request, JSON parsing, and classification validation
  cli.py                  Rule-based interactive CLI
tests/
  test_agent.py                 Rule-based workflow tests
  test_knowledge_base.py        FAQ lookup tests
  test_llm_classifier.py        Local LLM classifier tests with mocked API requests
  test_langgraph_llm_agent.py   LLM LangGraph tests with mocked classification
```

## Quick Start

The project was tested with Python 3.13.14 on Windows PowerShell.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run all local tests:

```powershell
python -m unittest discover -s .\tests -v
```

The current local suite contains 17 tests. It does not call a real model API.

Run the rule-based interactive CLI:

```powershell
python -m src.cli
```

## Model Configuration

Copy `.env.example` to `.env`, then set the values provided by an OpenAI-compatible model service:

```dotenv
OPENAI_COMPATIBLE_BASE_URL=https://your-provider.example/v1
OPENAI_COMPATIBLE_API_KEY=your-api-key
OPENAI_COMPATIBLE_MODEL=your-model-name
```

Never commit `.env` or a real API key.

Verify the model connection with one small live request:

```powershell
python -m src.model_probe
```

Run the complete LLM LangGraph flow with one live classification request:

```powershell
python -c "from src.langgraph_llm_agent import run_langgraph_llm_customer_service_agent; print(run_langgraph_llm_customer_service_agent('退款一般多久到账？'))"
```

For this query, the expected route is `billing_reply`, and the local FAQ answer is used as the final response.

## Test Strategy

- Rule and FAQ tests exercise deterministic local business logic.
- LLM parser tests cover valid JSON, repeated identical JSON, invalid JSON, unsupported categories, and contradictory JSON objects.
- LLM coordination tests mock `request_model_classification`, so no API request is made.
- LLM LangGraph tests mock `classify_with_model`, while real sentiment, routing, FAQ, and response nodes still execute.
- LLM LangGraph tests also cover an API timeout and verify that rule-based classification takes over.
- Live probes and live end-to-end commands are intentionally separate from `unittest`, because they depend on network availability, API credentials, quota, and model-service behavior.

## Known Limitations

- The LLM is used only for classification; sentiment analysis and FAQ retrieval remain keyword based.
- The local FAQ knowledge base is a small in-memory list with no source citations or versioning.
- Third-party model services can time out, be rate-limited, or return imperfect compatibility behavior. The classifier validates and safely handles repeated identical JSON, but conflicting results are rejected.
- There is no persistent conversation history, ticket system, database, authentication, rate limiting, observability, or human-agent backend.
- This project is appropriate for learning and portfolio demonstration, not direct production deployment.

## Security and Publishing Notes

- Do not commit `.env`, API keys, private logs, or real customer data.
- Keep `.venv`, `__pycache__`, and test caches out of version control.
- The implementation in this repository is an independent learning implementation. Do not copy code, assets, or documentation from third-party tutorial repositories without complying with their licenses.

## License

This project is released under the [MIT License](LICENSE).
