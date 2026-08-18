# Xiao Zhi 客服 Agent 完整运行流程

本文档基于当前项目代码整理，目标是回答三个问题：

1. 用户从哪里输入问题？
2. 代码会依次跳转到哪些函数和 LangGraph 节点？
3. 不同类型的问题为什么会得到不同的最终回复？

项目目录：

```text
D:\codexTest\customer-service-agent-learning
```

> **阅读和运行前提**
>
> 本文中的相对路径命令都默认在项目根目录执行。PowerShell 中请先运行：
>
> ```powershell
> Set-Location D:\codexTest\customer-service-agent-learning
> ```
>
> 如果在 `C:\Users\Administrator\Desktop` 等其他目录直接执行
> `python -m unittest ...`，Python 可能找不到项目内的 `src` 包，并出现
> `ModuleNotFoundError: No module named 'src'`。这不是 Agent 工作流本身失败，而是当前工作目录不正确。

---

## 1. 先记住整个项目的核心思想

这个项目不是“模型收到问题后直接生成答案”，而是一个分阶段工作流：

```text
用户问题
  ↓
分类与情绪分析
  ↓
路由决策
  ↓
是否需要 FAQ 检索
  ↓
是否允许回复模型
  ↓
最终回复
```

其中：

- 模型优先负责分类、情绪分析和受控语言改写。
- 本地 Python 负责路由、FAQ 约束、检索阈值和降级。
- FAQ 是可信事实来源。
- 负面问题优先转人工。
- 模型失败时保留本地流程。

可以把它理解成：

```text
模型负责“理解和组织语言”
本地代码负责“控制流程和限制风险”
```

---

## 2. 项目中有三条实际运行路径

### 2.1 手写顺序版

入口函数：

```python
src.agent.run_customer_service_agent()
```

调用链：

```text
create_customer_state()
  ↓
categorize_query()
  ↓
analyze_sentiment()
  ↓
choose_route()
  ↓
generate_response()
```

这是一条早期的手写协调路径，主要用于学习和部分规则测试。

当前代码中，这条路径没有调用 `retrieve_faq_answer()`，所以它不代表完整的 FAQ LangGraph 流程。

---

### 2.2 规则版 LangGraph

命令：

```powershell
python -m src.cli
```

入口：

```python
src.cli.main()
```

核心函数：

```python
src.langgraph_agent.run_langgraph_customer_service_agent()
```

完整节点：

```text
START
  ↓
categorize
  ↓
analyze_sentiment
  ↓
choose_route
  ├── human_handoff
  │      ↓
  │  generate_response
  │      ↓
  │     END
  │
  └── 自动回复路线
         ↓
  retrieve_faq_answer
         ↓
  generate_response
         ↓
        END
```

这是完整的本地规则工作流。

---

### 2.3 大模型 LangGraph

命令：

```powershell
python -m src.llm_cli
```

入口：

```python
src.llm_cli.main()
```

核心函数：

```python
src.langgraph_llm_agent.run_langgraph_llm_customer_service_agent()
```

完整节点：

```text
START
  ↓
analyze_query_with_model
  ↓
choose_route
  ├── human_handoff
  │      ↓
  │  generate_controlled_response
  │      ↓
  │     END
  │
  └── 自动回复路线
         ↓
  retrieve_faq_answer
         ↓
  generate_controlled_response
         ↓
        END
```

大模型版的分类和情绪分析合并成一个节点，但后续路由和 FAQ 仍由本地代码控制。

---

## 3. CustomerState：所有节点共享的数据

定义位置：

```text
D:\codexTest\customer-service-agent-learning\src\agent.py
```

状态定义：

```python
class CustomerState(TypedDict, total=False):
    query: str
    category: Literal["technical", "billing", "general"]
    analysis_source: Literal["llm", "rule", "rule_fallback"]
    analysis_error: str
    response_source: Literal["llm", "faq_fallback", "local"]
    response_error: str
    sentiment: Literal["positive", "negative", "neutral"]
    route: Literal[
        "technical_reply",
        "billing_reply",
        "general_reply",
        "human_handoff",
    ]
    faq_id: FaqId
    faq_answer: str
    retrieved_contexts: list[str]
    retrieval_score: float
    response: str
```

### 3.1 `total=False` 的含义

每个字段不需要从第一步就存在。

初始状态：

```python
{
    "query": "退款一般多久到账？"
}
```

分类后：

```python
{
    "query": "退款一般多久到账？",
    "category": "billing",
    "analysis_source": "llm",
}
```

FAQ 未命中时，状态里没有：

```python
faq_id
faq_answer
retrieved_contexts
retrieval_score
```

模型失败时，状态可能增加：

```python
analysis_source = "rule_fallback"
analysis_error = "APITimeoutError"
```

---

## 4. 规则版 LangGraph 的详细调用链

### 4.1 CLI 入口

文件：

```text
D:\codexTest\customer-service-agent-learning\src\cli.py
```

执行：

```powershell
python -m src.cli
```

代码流程：

```python
query = input("请输入您的问题：").strip()
```

如果输入为空，CLI 直接打印提示并结束。

有内容时：

```python
final_state = run_langgraph_customer_service_agent(query)
```

随后只展示：

```python
final_state["category"]
final_state["sentiment"]
final_state["route"]
final_state["response"]
```

内部的 `faq_id`、检索上下文和分数仍然保留在状态中，只是规则 CLI 没有打印出来。

---

### 4.2 创建 LangGraph

文件：

```text
D:\codexTest\customer-service-agent-learning\src\langgraph_agent.py
```

创建状态图：

```python
workflow = StateGraph(CustomerState)
```

注册节点：

```python
workflow.add_node("categorize", categorize_query)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("choose_route", choose_route)
workflow.add_node("generate_response", generate_response)
workflow.add_node("retrieve_faq_answer", retrieve_faq_answer)
```

注册固定边：

```python
workflow.add_edge(START, "categorize")
workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_edge("analyze_sentiment", "choose_route")
workflow.add_edge("retrieve_faq_answer", "generate_response")
workflow.add_edge("generate_response", END)
```

注册条件边：

```python
workflow.add_conditional_edges(
    "choose_route",
    should_retrieve_faq,
    {
        "retrieve_faq_answer": "retrieve_faq_answer",
        "generate_response": "generate_response",
    },
)
```

最后：

```python
app = workflow.compile()
```

`compile()` 把图结构编译成可以调用的工作流对象。

当前文件中 `retrieve_faq_answer -> generate_response` 的边出现了两次。现有测试通过，运行结果未受影响；从维护角度看，后续可以保留一条即可。

---

## 5. 分类节点：`categorize_query`

文件：

```text
D:\codexTest\customer-service-agent-learning\src\agent.py
```

规则优先级：

```text
billing
  ↓
technical
  ↓
general
```

账单关键词：

```python
("付款", "支付", "退款", "账单", "扣款")
```

技术关键词：

```python
("登录", "密码", "报错", "无法打开", "崩溃")
```

示例：

```text
退款一般多久到账？
```

包含 `退款`，得到：

```python
{
    "category": "billing",
    "analysis_source": "rule",
}
```

示例：

```text
软件打开后一直崩溃
```

得到：

```python
{
    "category": "technical",
    "analysis_source": "rule",
}
```

没有账单或技术关键词时：

```python
{
    "category": "general",
    "analysis_source": "rule",
}
```

---

## 6. 情绪节点：`analyze_sentiment`

负面关键词：

```python
(
    "不满意",
    "太差",
    "投诉",
    "生气",
    "失望",
    "垃圾",
    "一直没",
    "根本",
)
```

正面关键词：

```python
("谢谢", "满意", "很好", "不错", "赞")
```

判断顺序：

```text
先负面
再正面
最后 neutral
```

例如：

```text
退款一个月还没到账，太差了！
```

结果：

```python
{
    "sentiment": "negative",
    "analysis_source": "rule",
}
```

例如：

```text
你们的服务很好，谢谢！
```

结果：

```python
{
    "sentiment": "positive",
    "analysis_source": "rule",
}
```

---

## 7. 路由节点：`choose_route`

路由优先级：

```text
negative
  >
technical
  >
billing
  >
general
```

代码逻辑：

```python
if state["sentiment"] == "negative":
    route = "human_handoff"
elif state["category"] == "technical":
    route = "technical_reply"
elif state["category"] == "billing":
    route = "billing_reply"
else:
    route = "general_reply"
```

例如：

```text
软件打开后一直崩溃，太差了！
```

虽然 category 是 `technical`，但 sentiment 是 `negative`，因此最终路线是：

```python
"human_handoff"
```

这体现了：

```text
负面情绪优先于问题类别
```

---

## 8. 条件边：`should_retrieve_faq`

文件：

```text
D:\codexTest\customer-service-agent-learning\src\langgraph_agent.py
```

逻辑：

```python
if state["route"] == "human_handoff":
    return "generate_response"

return "retrieve_faq_answer"
```

如果用户是负面问题：

```text
choose_route
  ↓
human_handoff
  ↓
generate_response
```

如果用户不是负面问题：

```text
choose_route
  ↓
retrieve_faq_answer
  ↓
generate_response
```

因此人工转接路线会跳过 FAQ 检索。

---

## 9. FAQ 检索流程

文件：

```text
D:\codexTest\customer-service-agent-learning\src\knowledge_base.py
```

FAQ 条目包含：

```python
{
    "faq_id": "refund_timing",
    "title": "退款到账时效",
    "content": "退款申请审核通过后，原路退款通常在 3 至 5 个工作日到账。",
    "category": "billing",
    "source": "project_faq",
    "version": "1.0",
    "updated_at": "2026-08-12",
    "required_keywords": ("退款",),
    "intent_keywords": (
        "多久",
        "什么时候",
        "几天",
        "工作日",
        "多长时间",
        "何时",
    ),
    "answer": "退款申请审核通过后，原路退款通常在 3 至 5 个工作日到账。",
}
```

搜索函数：

```python
search_faq_entries(query, top_k=3)
```

匹配规则：

```text
required_keywords 必须全部命中
且
intent_keywords 至少命中一个
```

例如：

```text
退款一般多久到账？
```

命中：

```text
主题词：退款
意图词：多久
```

进入候选。

如果用户输入：

```text
退款已经到账，谢谢客服！
```

虽然命中 `退款`，但没有命中到账时效意图词，因此返回：

```python
[]
```

这样可以避免误用退款时效 FAQ。

---

## 10. FAQ 分数计算

当前本地规则检索使用：

```python
score = 0.5 + 0.5 * (
    matched_intent_count / len(entry["intent_keywords"])
)
```

解释：

- 主题关键词已经满足，所以固定得到 `0.5`；
- 意图关键词覆盖率贡献剩余 `0.5`；
- 分数只用于当前候选之间的排序，不代表向量模型相似度。

退款时效有 6 个意图关键词。

查询：

```text
退款一般多久到账？
```

命中 1 个意图词：

```text
0.5 + 0.5 × 1/6
= 0.5833333333333334
```

密码重置问题：

```text
我忘记密码了，怎么重置？
```

命中：

```text
忘记
重置
```

密码 FAQ 一共有 4 个意图词，所以：

```text
0.5 + 0.5 × 2/4
= 0.75
```

---

## 11. 检索阈值：`MIN_RETRIEVAL_SCORE`

位置：

```text
D:\codexTest\customer-service-agent-learning\src\agent.py
```

当前阈值：

```python
MIN_RETRIEVAL_SCORE = 0.55
```

检索节点的实际顺序：

```text
search_faq_entries(query, top_k=1)
  ↓
读取第一名候选
  ↓
读取 score
  ↓
score < 0.55？
```

分数低于阈值：

```python
return {}
```

后续状态不增加：

```python
faq_id
faq_answer
retrieved_contexts
retrieval_score
```

分数达到阈值：

```python
return {
    "faq_id": entry["faq_id"],
    "faq_answer": entry["answer"],
    "retrieved_contexts": [entry["content"]],
    "retrieval_score": retrieval_score,
}
```

---

## 12. 规则版回复节点：`generate_response`

文件：

```text
D:\codexTest\customer-service-agent-learning\src\agent.py
```

第一层判断：

```python
if route == "human_handoff":
    base_response = "您的问题已转交人工客服，请稍候。"
```

人工路线不读取 FAQ。

自动路线先看是否有 FAQ：

```python
faq_answer = state.get("faq_answer")
```

有 FAQ：

```python
base_response = faq_answer
```

没有 FAQ：

```python
fallback_responses = {
    "technical_reply": "抱歉给您带来不便。请尝试重新登录或重启应用。",
    "billing_reply": "您的账单问题已收到，请提供订单号以便进一步核实。",
    "general_reply": "客服工作时间为每日 9:00 至 18:00。",
}
```

如果分析阶段发生规则降级：

```python
if state.get("analysis_source") == "rule_fallback":
```

最终回复前增加：

```text
系统当前繁忙，已使用备用方式继续处理您的问题。
```

注意：这是面向用户的友好提示，不直接展示底层异常名称。

---

## 13. 大模型版分类流程

文件：

```text
D:\codexTest\customer-service-agent-learning\src\langgraph_llm_agent.py
```

入口节点：

```python
analyze_query_with_model(state)
```

内部跳转：

```text
analyze_query_with_model()
  ↓
analyze_with_model()
  ↓
request_model_analysis()
  ↓
load_model_config()
  ↓
create_openai_client()
  ↓
client.chat.completions.create()
  ↓
parse_model_analysis()
  ↓
返回 category + sentiment
```

模型被要求返回：

```json
{
  "category": "billing",
  "sentiment": "neutral"
}
```

本地解析器会验证：

- JSON 是否合法；
- 顶层是否为对象；
- category 是否属于 `technical`、`billing`、`general`；
- sentiment 是否属于 `positive`、`neutral`、`negative`；
- 重复 JSON 是否完全一致；
- 多个 JSON 是否互相矛盾。

模型成功时：

```python
{
    "category": model_update["category"],
    "sentiment": model_update["sentiment"],
    "analysis_source": "llm",
}
```

---

## 14. 大模型分类失败时的降级流程

捕获的错误类型：

```python
APITimeoutError
APIConnectionError
APIStatusError
ValueError
```

失败后：

```text
模型分析失败
  ↓
categorize_query()
  ↓
analyze_sentiment()
  ↓
analysis_source = rule_fallback
  ↓
继续走本地路由
```

返回：

```python
{
    "category": "technical",
    "sentiment": "neutral",
    "analysis_source": "rule_fallback",
    "analysis_error": "APITimeoutError",
}
```

之后仍然进入：

```text
choose_route
  ↓
FAQ 或本地回复
```

分析降级不会让整个工作流直接中断。

---

## 15. 大模型版回复条件

函数：

```python
generate_controlled_response(state)
```

只有以下三个条件全部满足，才调用回复模型：

```python
can_use_model = (
    state["route"] != "human_handoff"
    and state.get("analysis_source") == "llm"
    and state.get("faq_answer") is not None
)
```

也就是：

```text
不是人工转接
且分类分析来自 LLM
且 FAQ 已命中
```

调用：

```python
generate_reply_with_model(
    state["query"],
    state["faq_answer"],
)
```

发送给回复模型的核心数据：

```json
{
  "query": "退款一般多久到账？",
  "verified_answer": "退款申请审核通过后，原路退款通常在 3 至 5 个工作日到账。"
}
```

`verified_answer` 来自本地知识库，回复模型负责组织表达，不负责创造新的业务事实。

---

## 16. 回复模型失败时

回复模型可能出现：

- 超时；
- 网络错误；
- 服务端错误；
- JSON 非法；
- response 字段为空；
- 回复超过 300 个字符；
- 多个回复对象互相矛盾。

失败后：

```text
generate_reply_with_model()
  ↓
异常
  ↓
generate_response(state)
  ↓
使用 FAQ 原文
  ↓
response_source = faq_fallback
```

状态示例：

```python
{
    "analysis_source": "llm",
    "response_source": "faq_fallback",
    "response_error": "APITimeoutError",
    "response": "退款申请审核通过后，原路退款通常在 3 至 5 个工作日到账。",
}
```

---

## 17. 不同用户输入的完整流程

### 17.1 退款时效，中性

输入：

```text
退款一般多久到账？
```

规则版：

```text
categorize_query
  → billing
analyze_sentiment
  → neutral
choose_route
  → billing_reply
retrieve_faq_answer
  → refund_timing
  → score = 0.5833
generate_response
  → FAQ 原文
```

大模型版：

```text
analyze_query_with_model
  → category = billing
  → sentiment = neutral
  → analysis_source = llm
choose_route
  → billing_reply
retrieve_faq_answer
  → 命中 FAQ
generate_controlled_response
  → 调用回复模型
  → response_source = llm
```

---

### 17.2 技术问题，无 FAQ

输入：

```text
软件打开后一直崩溃
```

状态：

```text
category = technical
sentiment = neutral
route = technical_reply
```

由于没有命中密码重置 FAQ：

```text
faq_answer 不存在
```

最终：

```text
抱歉给您带来不便。请尝试重新登录或重启应用。
```

大模型版中，即使分类来源是 `llm`，回复模型仍不调用，因为 FAQ 不存在。

---

### 17.3 技术问题，负面情绪

输入：

```text
软件打开后一直崩溃，太差了！
```

状态：

```text
category = technical
sentiment = negative
route = human_handoff
```

流程：

```text
choose_route
  → human_handoff
  → 跳过 FAQ
  → 跳过回复模型
  → 本地转人工
```

最终：

```text
您的问题已转交人工客服，请稍候。
```

---

### 17.4 退款逾期，负面情绪

输入：

```text
退款一个月还没到账，太差了！
```

状态：

```text
category = billing
sentiment = negative
route = human_handoff
```

虽然文本有“退款”，但负面路线优先，FAQ 节点不会执行。

---

### 17.5 客服时间，中性

输入：

```text
你们的客服工作时间是什么时候？
```

状态：

```text
category = general
sentiment = neutral
route = general_reply
faq_id = service_hours
```

规则版直接使用 FAQ 原文。

大模型版在分析成功且满足条件时调用受控回复模型。

---

### 17.6 密码重置

输入：

```text
我忘记密码了，怎么重置？
```

状态：

```text
category = technical
sentiment = neutral
route = technical_reply
faq_id = password_reset
```

FAQ 回复：

```text
请在登录页选择“忘记密码”，按提示完成密码重置。
```

---

### 17.7 正面服务反馈

输入：

```text
你们的服务很好，谢谢！
```

状态：

```text
category = general
sentiment = positive
route = general_reply
```

如果没有命中 FAQ，就使用本地 `general_reply` 模板。

正面情绪本身不代表一定调用回复模型。

---

### 17.8 分类模型超时

输入：

```text
软件打开后一直崩溃
```

假设模型分析超时：

```text
analyze_with_model
  → APITimeoutError
  → categorize_query
  → analyze_sentiment
  → analysis_source = rule_fallback
  → choose_route
  → technical_reply
  → generate_controlled_response
  → 本地回复
```

用户可见回复前增加：

```text
系统当前繁忙，已使用备用方式继续处理您的问题。
```

---

### 17.9 回复模型超时

输入：

```text
退款一般多久到账？
```

假设分类成功、FAQ 命中，但回复模型超时：

```text
analysis_source = llm
response_source = faq_fallback
response_error = APITimeoutError
```

最终回复使用可信 FAQ 原文。

---

## 18. 观测字段如何显示

文件：

```text
D:\codexTest\customer-service-agent-learning\src\observability.py
```

函数：

```python
format_observability(state)
```

正常结果：

```text
分析来源：llm
回复来源：llm
```

分析降级：

```text
分析来源：rule_fallback
回复来源：local
分析错误：APITimeoutError
```

回复降级：

```text
分析来源：llm
回复来源：faq_fallback
回复错误：APITimeoutError
```

这些字段主要服务开发者和评估系统，最终用户重点看到 `response`。

---

## 19. 模型配置和真实 API 调用位置

配置文件：

```text
D:\codexTest\customer-service-agent-learning\src\model_config.py
```

读取：

```python
load_dotenv()
```

需要：

```text
OPENAI_COMPATIBLE_BASE_URL
OPENAI_COMPATIBLE_API_KEY
OPENAI_COMPATIBLE_MODEL
```

客户端创建位置：

```text
D:\codexTest\customer-service-agent-learning\src\model_client.py
```

```python
OpenAI(
    api_key=config.api_key,
    base_url=config.base_url,
    timeout=30.0,
    max_retries=0,
)
```

真正发起网络请求的位置：

```python
client.chat.completions.create(...)
```

分类请求在：

```text
D:\codexTest\customer-service-agent-learning\src\llm_classifier.py
```

回复请求在：

```text
D:\codexTest\customer-service-agent-learning\src\llm_responder.py
```

`create_openai_client()` 只创建客户端对象；真正产生 API 请求的是 `chat.completions.create()`。

---

## 20. 评估运行流程

### 20.1 规则基线

命令：

```powershell
python -m src.rule_baseline_runner
```

调用链：

```text
rule_baseline_runner.main()
  ↓
evaluate_all_cases(
    agent_runner=run_langgraph_customer_service_agent
)
  ↓
validate_evaluation_cases()
  ↓
evaluate_case()
  ↓
规则版 LangGraph
  ↓
比较预期 category/sentiment/route/FAQ/faq_id
  ↓
build_summary()
  ↓
save_evaluation_report()
```

规则基线不调用模型 API。

---

### 20.2 大模型候选评估

命令：

```powershell
python -m src.llm_candidate_runner --limit 50
```

调用链：

```text
llm_candidate_runner.main()
  ↓
select_evaluation_cases(50)
  ↓
validate_evaluation_cases()
  ↓
run_llm_candidate_evaluation()
  ↓
evaluate_case()
  ↓
run_langgraph_llm_customer_service_agent()
  ↓
真实模型工作流
  ↓
save_evaluation_report()
```

`--limit 3` 用于小样本试运行，`--limit 50` 才是完整候选评估。

---

### 20.3 保存报告比较

命令：

```powershell
python -m src.compare_saved_reports `
  --baseline-dir .\reports\baselines\<baseline_run_id> `
  --candidate-dir .\reports\candidates\<candidate_run_id>
```

调用链：

```text
compare_report_directories()
  ↓
load_evaluation_results()
  ↓
读取 baseline results.json
  ↓
读取 candidate results.json
  ↓
build_comparison()
  ↓
检查样本数量、name 和顺序
  ↓
计算整体、分类、情绪、路由、FAQ 指标
  ↓
save_comparison_report()
```

这个比较过程只读取历史 JSON，不重新运行 Agent。

---

## 21. 最容易混淆的几个区别

### 21.1 `analysis_source` 和 `response_source`

```text
analysis_source：分类和情绪从哪里来
response_source：最终回复从哪里来
```

可能出现：

```text
analysis_source = llm
response_source = local
```

例如：

```text
模型成功识别为 technical
没有 FAQ
使用本地技术模板
```

也可能出现：

```text
analysis_source = llm
response_source = faq_fallback
```

表示分类成功，但回复模型失败，最终使用 FAQ 原文。

---

### 21.2 `faq_id` 和 `retrieval_score`

```text
faq_id：命中了哪条知识
retrieval_score：这条知识与问题的匹配程度
```

例如：

```python
{
    "faq_id": "password_reset",
    "retrieval_score": 0.75,
}
```

`faq_id` 是身份，`retrieval_score` 是质量。

---

### 21.3 `faq_answer` 和 `retrieved_contexts`

```text
faq_answer：当前兼容旧回复流程的答案字符串
retrieved_contexts：面向 RAG 评估保存的上下文列表
```

当前两者内容通常相同，但职责不同：

- `faq_answer` 用于旧的回复调用；
- `retrieved_contexts` 用于后续 Context Precision、Context Recall 和 Faithfulness 评估。

---

## 22. 建议的调试顺序

遇到问题时，不要先看最终回复，按下面顺序检查：

```text
1. query 是否正确
2. category 是否正确
3. sentiment 是否正确
4. route 是否正确
5. 是否执行 FAQ
6. faq_id 是否正确
7. retrieval_score 是否达到 0.55
8. response_source 来自哪里
9. response_error 是否存在
10. 最终 response 是否符合预期
```

查看完整状态：

```powershell
python -c "from src.langgraph_llm_agent import run_langgraph_llm_customer_service_agent; print(run_langgraph_llm_customer_service_agent('退款一般多久到账？'))"
```

查看规则版完整状态：

```powershell
python -c "from src.langgraph_agent import run_langgraph_customer_service_agent; print(run_langgraph_customer_service_agent('退款一般多久到账？'))"
```

---

## 23. 当前项目的准确定位

当前项目已经具备：

- Python Agent 状态设计；
- LangGraph 条件路由；
- 规则版与大模型版双路径；
- 结构化模型输出解析；
- 分类降级；
- 回复降级；
- 本地 FAQ 检索；
- FAQ 负例控制；
- 检索分数和阈值；
- 结构化评估报告；
- 基线与候选方案比较；
- 58 项本地测试（当前最近一次完整运行结果为 `Ran 58 tests ... OK`）。

当前仍处于轻量本地 RAG 过渡阶段：

- FAQ 仍保存在 Python 内存结构中；
- 当前分数是可解释关键词分数；
- 尚未接入向量数据库；
- 尚未执行 RAGAS；
- 尚未加入 FastAPI、Docker 和浏览器界面。

因此项目目前可以准确描述为：

```text
带有规则降级、受控回复和可追踪评估的 LangGraph 智能客服 Agent，
正在从关键词 FAQ 过渡到可评估的本地 RAG 检索层。
```
