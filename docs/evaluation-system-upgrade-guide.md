# 客服 Agent 评估系统升级学习手册

> 本文档记录本轮对项目评估系统的完整升级。
>
> 目标不是只告诉你“改了哪些文件”，而是让你能够理解：
>
> ```text
> Agent 返回什么
> -> 评估器如何读取
> -> EvaluationResult 如何保存
> -> build_summary() 如何统计
> -> results.json 和 summary.json 如何生成
> -> 终端为什么能显示这些指标
> -> 测试如何证明功能没有破坏
> ```

项目根目录：

```text
D:\codexTest\customer-service-agent-learning
```

---

## 1. 本轮升级目标

本轮把原来主要关注“准确率”的评估系统，升级为同时关注：

```text
业务正确性
检索证据
运行效率
模型调用次数
失败类型
Token 使用
成本估算
运行元数据
```

升级前，评估结果主要包含：

```text
分类是否正确
情绪是否正确
路由是否正确
FAQ 是否正确
FAQ ID 是否正确
分析来源
回复来源
```

升级后，每条评估结果还会包含：

```text
retrieved_contexts
retrieval_score
latency_ms
analysis_model_calls
response_model_calls
model_call_count
analysis_error
response_error
input_tokens
output_tokens
estimated_cost_usd
```

整体汇总还会包含：

```text
平均延迟
P95 延迟
模型调用总次数
平均每条模型调用次数
Token 总量
估算成本
超时次数
解析失败次数
失败类型分布
```

---

## 2. 本轮实际修改的文件

本轮修改或新增了以下文件：

```text
src/evaluation_runner.py
src/rule_baseline_runner.py
src/llm_candidate_runner.py
src/agent.py
src/langgraph_llm_agent.py
src/llm_classifier.py
src/llm_responder.py
src/model_usage.py
tests/test_evaluation_runner.py
tests/test_model_usage.py
.env.example
README.md
```

没有修改以下范围：

```text
src/evaluation_cases.py
src/knowledge_base.py
src/langgraph_agent.py
```

原因是本轮的重点是：

```text
完善评估数据记录和运行指标
```

而不是重新设计评估集或 FAQ 检索算法。

---

## 3. 先理解评估系统的完整调用链

以规则基线为例：

```text
python -m src.rule_baseline_runner
    |
    v
run_rule_baseline_evaluation()
    |
    v
evaluate_all_cases(agent_runner=run_langgraph_customer_service_agent)
    |
    v
evaluate_case(case, agent_runner)
    |
    v
run_langgraph_customer_service_agent(query)
    |
    v
返回 CustomerState
    |
    v
evaluate_case() 转换为 EvaluationResult
    |
    v
build_summary(results)
    |
    v
save_evaluation_report()
    |
    +--> results.json
    |
    +--> summary.json
```

大模型候选评估的调用链只有 Agent 入口不同：

```text
python -m src.llm_candidate_runner --limit 50
    |
    v
select_evaluation_cases(50)
    |
    v
run_langgraph_llm_customer_service_agent(query)
    |
    v
evaluate_case()
    |
    v
build_summary()
    |
    v
save_evaluation_report()
```

这说明：

```text
规则基线和大模型候选使用同一个评估器
```

因此两套方案可以公平比较。

---

## 4. `EvaluationResult`：一条样本的完整记录

文件：

```text
D:\codexTest\customer-service-agent-learning\src\evaluation_runner.py
```

核心类型：

```python
class EvaluationResult(TypedDict):
    name: str
    query: str
    passed: bool

    category_ok: bool
    sentiment_ok: bool
    route_ok: bool
    faq_ok: bool
    faq_id_ok: bool

    actual_category: str
    expected_category: str
    actual_sentiment: str
    expected_sentiment: str
    actual_route: str
    expected_route: str

    actual_faq_in_state: bool
    expected_faq_in_state: bool
    actual_faq_id: FaqId | None
    expected_faq_id: FaqId | None

    complexity: Complexity
    tags: list[str]
    analysis_source: str
    response_source: str
```

这些字段主要回答：

```text
业务结果对不对？
```

本轮新增的检索证据字段：

```python
retrieved_contexts: list[str]
retrieval_score: float | None
```

它们回答：

```text
Agent 实际找到了什么？
这个检索结果的分数是多少？
```

本轮新增的运行指标字段：

```python
latency_ms: float
analysis_model_calls: int
response_model_calls: int
model_call_count: int
```

它们回答：

```text
处理一条问题花了多长时间？
分类阶段调用了几次模型？
回复阶段调用了几次模型？
总共调用了几次模型？
```

本轮新增的错误字段：

```python
analysis_error: str | None
response_error: str | None
```

它们回答：

```text
分类阶段是否失败？
回复阶段是否失败？
失败的异常类型是什么？
```

本轮新增的 Token 和成本字段：

```python
input_tokens: int | None
output_tokens: int | None
estimated_cost_usd: float | None
```

它们回答：

```text
模型输入了多少 Token？
模型输出了多少 Token？
这条样本估算花费多少钱？
```

---

## 5. 为什么使用 `NotRequired`

代码中使用了：

```python
from typing import NotRequired
```

例如：

```python
class EvaluationResult(TypedDict):
    retrieved_contexts: NotRequired[list[str]]
    retrieval_score: NotRequired[float | None]
```

这里有两个原因。

### 5.1 兼容旧测试夹具

部分旧测试只关心：

```text
分类是否正确
汇总是否正确
比较是否正确
```

它们手动创建的结果可能没有：

```python
retrieved_contexts
retrieval_score
```

如果把新字段全部声明成强制字段，那么所有旧测试夹具都需要同步补字段。

### 5.2 兼容旧报告

项目中已经存在一些旧的 `results.json`。

旧报告没有本轮新增字段，但仍然应该可以被：

```python
load_evaluation_results()
```

读取并参与比较。

因此：

```text
新运行结果：会产生新字段
旧测试夹具：可以暂时不写新字段
旧报告：仍然可以读取
```

注意：

```text
NotRequired 只影响类型提示
不会自动生成字段
不会自动校验运行时字典
```

真正负责填写字段的是：

```text
evaluate_case()
```

---

## 6. `evaluate_case()` 的新执行顺序

函数位置：

```text
src/evaluation_runner.py::evaluate_case
```

完整顺序如下：

```text
1. 确认 Agent 函数
2. 开始高精度计时
3. 调用 Agent
4. 读取最终 CustomerState
5. 计算单条耗时
6. 判断分类、情绪、路由和 FAQ 是否正确
7. 复制 FAQ 检索证据
8. 推断模型调用次数
9. 复制错误类型
10. 复制 Token 和成本字段
11. 返回 EvaluationResult
```

### 6.1 测量耗时

代码使用：

```python
from time import perf_counter
```

开始时：

```python
started_at = perf_counter()
```

Agent 返回后：

```python
latency_ms = round(
    (perf_counter() - started_at) * 1000,
    3,
)
```

`perf_counter()` 适合测量运行时间，因为它是高精度计时器。

乘以 `1000` 是为了把秒转换成毫秒：

```text
秒 × 1000 = 毫秒
```

例如：

```text
0.0012 秒
= 1.2 毫秒
```

### 6.2 复制检索上下文

代码：

```python
"retrieved_contexts": final_state.get(
    "retrieved_contexts",
    [],
),
```

如果 FAQ 命中：

```python
final_state["retrieved_contexts"]
```

会被复制到评估结果。

如果 FAQ 未命中：

```python
final_state.get("retrieved_contexts", [])
```

返回空列表。

### 6.3 复制检索分数

代码：

```python
"retrieval_score": final_state.get(
    "retrieval_score",
),
```

FAQ 命中时：

```python
0.5833333333333334
```

FAQ 未命中时：

```python
None
```

保存到 JSON 后：

```python
None
```

会变成：

```json
null
```

### 6.4 复制错误类型

分类错误：

```python
"analysis_error": final_state.get("analysis_error")
```

回复错误：

```python
"response_error": final_state.get("response_error")
```

例如：

```python
{
    "analysis_source": "rule_fallback",
    "analysis_error": "APITimeoutError",
    "response_source": "local",
    "response_error": None,
}
```

这表示：

```text
分类模型尝试过，但超时
规则分类接管
回复阶段没有再调用模型
```

---

## 7. 模型调用次数如何计算

当前工作流每个阶段最多调用一次模型：

```text
分类阶段最多一次
回复阶段最多一次
```

所以可以根据来源字段推断调用次数。

### 7.1 分类阶段

```python
analysis_source == "llm"
```

表示分类模型成功调用一次：

```text
analysis_model_calls = 1
```

```python
analysis_source == "rule_fallback"
```

表示分类模型先尝试，但失败后规则接管：

```text
analysis_model_calls = 1
```

```python
analysis_source == "rule"
```

表示规则基线根本没有尝试模型：

```text
analysis_model_calls = 0
```

### 7.2 回复阶段

```python
response_source == "llm"
```

表示回复模型成功调用一次：

```text
response_model_calls = 1
```

```python
response_source == "faq_fallback"
```

表示回复模型先尝试，但失败后使用 FAQ 回退：

```text
response_model_calls = 1
```

```python
response_source == "local"
```

表示使用本地回复：

```text
response_model_calls = 0
```

### 7.3 示例

中性 FAQ 问题：

```python
analysis_source = "llm"
response_source = "llm"
```

调用次数：

```text
分类模型：1
回复模型：1
总次数：2
```

负面问题：

```python
analysis_source = "llm"
response_source = "local"
```

调用次数：

```text
分类模型：1
回复模型：0
总次数：1
```

规则基线：

```python
analysis_source = "rule"
response_source = "local"
```

调用次数：

```text
分类模型：0
回复模型：0
总次数：0
```

---

## 8. `build_summary()` 新增的运行指标

函数位置：

```text
src/evaluation_runner.py::build_summary
```

### 8.1 平均延迟

计算方式：

```text
所有样本延迟之和 / 样本数量
```

代码逻辑：

```python
sum(latency_values) / len(latency_values)
```

例如：

```text
10ms、20ms、40ms
```

平均延迟：

```text
(10 + 20 + 40) / 3
= 23.333ms
```

### 8.2 P95 延迟

P95 的含义是：

> 95% 的样本延迟不超过这个值。

当前使用最近秩方法：

```python
rank = math.ceil(len(values) * 0.95) - 1
```

然后：

```python
ordered = sorted(values)
```

选择对应位置。

例如：

```text
10ms、20ms、40ms
```

排序后：

```text
[10, 20, 40]
```

P95 结果为：

```text
40ms
```

注意：

```text
P95 不是最大值的同义词
```

它用于观察长尾延迟。

### 8.3 模型调用总数

```python
model_call_total = sum(
    result["model_call_count"]
    for result in results
)
```

它可以帮助你回答：

```text
一个问题平均需要几次模型调用？
```

这比只统计“是否使用模型”更具体。

### 8.4 Token 总量

如果模型服务返回 usage：

```python
input_tokens
output_tokens
```

评估器会求和。

如果服务没有返回 usage：

```python
input_tokens_total = None
output_tokens_total = None
```

这里使用 `None`，因为：

```text
没有观测到 Token
```

不等于：

```text
真实使用了 0 Token
```

### 8.5 失败类型统计

所有分析和回复阶段错误会合并统计：

```python
failure_type_counts = {
    "ValueError": 2,
    "APITimeoutError": 1,
}
```

同时分别统计：

```text
timeout_count
parse_failure_count
```

当前将以下类型视为超时：

```python
APITimeoutError
TimeoutError
```

当前将以下类型视为解析失败：

```python
ValueError
```

这套统计是确定性规则，不调用评估模型。

---

## 9. `model_usage.py`：读取模型 usage

新增文件：

```text
D:\codexTest\customer-service-agent-learning\src\model_usage.py
```

这个文件的职责是：

```text
从兼容 OpenAI 的响应对象中提取 usage
保留原始模型文本
计算可选成本
```

### 9.1 为什么不直接改变 request 函数返回字典？

原来的接口是：

```python
request_model_analysis(query) -> str
```

很多测试和解析函数都依赖这个接口。

如果直接改成：

```python
request_model_analysis(query) -> dict
```

会破坏：

```text
JSON 解析器
重复 JSON 处理
现有测试
模型原始文本边界
```

因此本轮使用：

```python
UsageText(str)
```

它本质上仍然是字符串：

```python
isinstance(content, str)
```

结果为：

```text
True
```

但它还可以附带：

```python
content.usage
```

这样：

```text
原有解析接口不变
新增 usage 信息
```

### 9.2 `UsageText` 的数据结构

```python
class UsageText(str):
    usage: ModelUsage
```

创建过程：

```python
instance = str.__new__(cls, value)
instance.usage = usage or {}
```

因此它同时满足：

```text
像字符串一样被 JSON parser 使用
像对象一样读取 usage
```

### 9.3 提取标准字段

兼容 OpenAI 的响应通常类似：

```python
response.usage.prompt_tokens
response.usage.completion_tokens
```

本项目映射为：

```python
input_tokens
output_tokens
total_tokens
```

### 9.4 成本估算

成本配置是可选的：

```dotenv
MODEL_INPUT_COST_PER_1K=
MODEL_OUTPUT_COST_PER_1K=
```

例如：

```dotenv
MODEL_INPUT_COST_PER_1K=0.01
MODEL_OUTPUT_COST_PER_1K=0.02
```

如果某条请求使用：

```text
输入 100 Token
输出 20 Token
```

成本为：

```text
100 / 1000 × 0.01
+ 20 / 1000 × 0.02
= 0.0014 美元
```

如果没有配置价格：

```python
estimated_cost_usd = None
```

这比随意假设模型价格更加准确。

---

## 10. 分类和回复模块如何传递 usage

分类模块：

```text
request_model_analysis()
    ↓
UsageText
    ↓
analyze_with_model()
    ↓
ModelAnalysis
    ↓
analyze_query_with_model()
    ↓
CustomerState
```

回复模块：

```text
request_model_reply()
    ↓
UsageText
    ↓
generate_reply_with_model()
    ↓
ModelReply
    ↓
generate_controlled_response()
    ↓
CustomerState
```

工作流使用了辅助函数：

```python
_copy_usage_fields()
```

它负责复制：

```python
input_tokens
output_tokens
estimated_cost_usd
```

这样分类阶段和回复阶段可以共享同一个状态结构。

---

## 11. 报告元数据

`save_evaluation_report()` 新增参数：

```python
metadata: dict[str, object] | None = None
```

规则基线传入：

```python
{
    "runner": "rule_baseline",
    "mode": "offline",
    "model_name": "rule_based",
}
```

大模型候选传入：

```python
{
    "runner": "llm_candidate",
    "mode": "llm",
    "model_name": config.model,
}
```

保存到：

```json
{
  "run_id": "20260818-092823",
  "created_at": "2026-08-18T09:28:23.355617+00:00",
  "sample_count": 50,
  "metadata": {
    "runner": "rule_baseline",
    "mode": "offline",
    "model_name": "rule_based"
  },
  "summary": {}
}
```

这些元数据帮助你回答：

```text
这份报告是谁生成的？
是规则版还是 LLM 版？
使用了什么模型？
什么时候生成的？
```

---

## 12. `results.json` 新结构

FAQ 命中样本示例：

```json
{
  "name": "refund_timing_neutral",
  "query": "退款一般多久到账？",
  "passed": true,
  "actual_category": "billing",
  "actual_sentiment": "neutral",
  "actual_route": "billing_reply",
  "actual_faq_id": "refund_timing",
  "retrieved_contexts": [
    "退款申请审核通过后，原路退款通常在 3 至 5 个工作日到账。"
  ],
  "retrieval_score": 0.5833333333333334,
  "analysis_source": "llm",
  "response_source": "llm",
  "latency_ms": 412.631,
  "analysis_model_calls": 1,
  "response_model_calls": 1,
  "model_call_count": 2,
  "input_tokens": 180,
  "output_tokens": 38,
  "estimated_cost_usd": 0.0021
}
```

FAQ 未命中样本示例：

```json
{
  "name": "payment_status_neutral",
  "actual_faq_id": null,
  "retrieved_contexts": [],
  "retrieval_score": null,
  "analysis_source": "llm",
  "response_source": "local",
  "model_call_count": 1
}
```

这表示：

```text
分类调用了模型
FAQ 没有命中
回复阶段没有调用模型
```

---

## 13. `summary.json` 新结构

汇总文件中会增加：

```json
{
  "latency_ms_average": 1.06492,
  "latency_ms_p95": 1.422,
  "latency_observation_count": 50,
  "model_call_total": 0,
  "model_call_average": 0.0,
  "input_tokens_total": null,
  "output_tokens_total": null,
  "estimated_cost_usd_total": null,
  "token_observation_count": 0,
  "cost_observation_count": 0,
  "timeout_count": 0,
  "parse_failure_count": 0,
  "failure_type_counts": {}
}
```

规则基线中：

```text
模型调用总数为 0
Token 为 null
成本为 null
```

这是正确的，因为规则基线不调用模型。

---

## 14. 终端显示内容

`print_summary()` 现在会显示：

```text
分类准确率
情绪准确率
路由准确率
FAQ 状态准确率
FAQ ID 准确率
分析来源统计
回复来源统计
分析降级次数
回复降级次数
平均延迟
P95 延迟
模型调用总次数
平均每条模型调用次数
输入 Token 总数
输出 Token 总数
估算成本
Token 观测样本数
成本观测样本数
超时次数
解析失败次数
失败类型统计
复杂度分组指标
标签分组指标
```

`print_report()` 现在会对每条样本显示：

```text
检索上下文数量
检索分数
单条耗时
模型调用次数
分析错误
回复错误
```

完整上下文不会全部打印到终端，而是保存在：

```text
results.json
```

这样可以避免终端输出过长。

---

## 15. 单元测试新增内容

### 15.1 `test_evaluation_runner.py`

新增：

```text
test_evaluate_case_records_retrieval_and_runtime_metrics
test_build_summary_records_runtime_metrics_and_failures
```

第一个测试验证：

```text
retrieved_contexts 被保存
retrieval_score 被保存
模型调用次数被推断
耗时被记录
Token 被保留
成本被保留
回复错误被保留
```

第二个测试验证：

```text
平均延迟
P95 延迟
模型调用总数
平均模型调用次数
Token 总数
成本总数
Token 观测数量
成本观测数量
超时数量
解析失败数量
失败类型分布
```

### 15.2 `test_model_usage.py`

新增：

```text
test_extract_usage_and_estimate_cost
test_attach_usage_preserves_string_interface
```

第一个测试使用本地模拟响应：

```python
SimpleNamespace(
    usage=SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=20,
    )
)
```

不会请求真实 API。

第二个测试验证：

```python
UsageText
```

仍然可以当作普通字符串使用。

---

## 16. 当前测试结果

完整命令：

```powershell
Set-Location D:\codexTest\customer-service-agent-learning
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
```

当前结果：

```text
Ran 62 tests
OK
```

评估系统专用测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -p test_evaluation_runner.py -v
```

当前结果：

```text
Ran 15 tests
OK
```

模型 usage 测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -p test_model_usage.py -v
```

预期：

```text
Ran 2 tests
OK
```

---

## 17. 本地规则基线验证

规则基线不调用真实模型，可以安全运行：

```powershell
.\.venv\Scripts\python.exe -m src.rule_baseline_runner
```

本轮实际生成：

```text
报告目录：
reports\baselines\20260818-092823
```

结果：

```text
规则基线评估完成：44/50
```

对应汇总指标：

```text
分类正确：46/50
情绪正确：48/50
路由正确：45/50
FAQ 状态正确：50/50
FAQ ID 正确：50/50
平均延迟：约 1.06492 ms
P95 延迟：约 1.422 ms
模型调用总次数：0
超时次数：0
解析失败次数：0
```

规则基线的 Token 和成本为：

```json
{
  "input_tokens_total": null,
  "output_tokens_total": null,
  "estimated_cost_usd_total": null
}
```

这是因为规则基线没有模型请求。

---

## 18. 大模型候选评估注意事项

大模型候选评估会调用真实 API：

```powershell
.\.venv\Scripts\python.exe -m src.llm_candidate_runner --limit 3
```

建议先使用：

```text
--limit 3
```

确认：

```text
.env 配置正确
模型可以连接
JSON 输出可以解析
回复逻辑正常
报告可以保存
```

再考虑：

```powershell
.\.venv\Scripts\python.exe -m src.llm_candidate_runner --limit 50
```

调用成本由以下因素决定：

```text
样本数量
分类模型调用次数
回复模型调用次数
输入 Token
输出 Token
服务商价格
```

不要直接运行：

```powershell
python -m src.evaluation_runner
```

除非你明确知道当前 `main()` 默认使用的是大模型工作流。

更安全的入口是：

```text
规则基线：
python -m src.rule_baseline_runner

大模型候选：
python -m src.llm_candidate_runner --limit 3
```

---

## 19. 本轮评估系统完成度

本轮已经完成：

```text
1. EvaluationResult 保存检索上下文
2. EvaluationResult 保存检索分数
3. 记录单条样本延迟
4. 计算平均延迟
5. 计算 P95 延迟
6. 推断分类模型调用次数
7. 推断回复模型调用次数
8. 统计模型调用总数
9. 记录分析错误
10. 记录回复错误
11. 统计超时次数
12. 统计解析失败次数
13. 统计失败类型
14. 支持模型 usage 提取
15. 支持输入 Token 统计
16. 支持输出 Token 统计
17. 支持可选成本估算
18. 保存运行元数据
19. 终端展示新指标
20. 新增本地单元测试
21. 完整测试通过
```

目前仍然没有完成：

```text
RAGAS 评估执行器
100 条以上评估集
FastAPI
浏览器页面
Docker
评估图表
GitHub Actions
```

因此：

```text
当前评估系统本身已经达到可用版本
```

但它还不是最终的完整作品集系统。

---

## 20. 当前评估系统的边界

### 20.1 Token 和成本不是每次都有

只有模型响应包含：

```python
response.usage
```

评估器才能读取：

```text
prompt_tokens
completion_tokens
```

如果兼容服务不返回 usage：

```text
Token = null
成本 = null
```

这比虚构 Token 或成本更可靠。

### 20.2 模型调用次数是当前工作流下的推断值

当前每个阶段最多一次调用，因此可以根据来源推断。

如果未来把一个节点改成：

```text
自动重试三次
```

那么仅根据 `analysis_source` 就不够准确。

后续应改为在状态中保存：

```text
analysis_attempt_count
response_attempt_count
```

### 20.3 当前 P95 是单机运行指标

当前延迟来自本地 PowerShell 和本地 Python 进程。

它不能直接代表生产环境延迟，因为生产环境还会受到：

```text
网络
服务商响应
并发
服务器负载
模型队列
```

影响。

---

## 21. 你现在应该如何学习这份文档

建议按以下顺序阅读：

```text
第一遍：
第 3 节，理解完整调用链

第二遍：
第 4～7 节，理解一条 EvaluationResult 如何生成

第三遍：
第 8～11 节，理解汇总和 usage

第四遍：
第 12～14 节，对照 JSON 和终端输出

第五遍：
第 15～18 节，理解测试与真实运行

第六遍：
第 19～20 节，理解完成度和系统边界
```

阅读时建议打开以下两个文件对照：

```text
D:\codexTest\customer-service-agent-learning\src\evaluation_runner.py
D:\codexTest\customer-service-agent-learning\tests\test_evaluation_runner.py
```

使用编辑器搜索：

```text
EvaluationResult
evaluate_case
build_summary
save_evaluation_report
print_summary
print_report
```

---

## 22. 当前可以写进简历的评估能力

当前可以较准确地描述为：

> 构建多维度 Agent 评估系统，记录分类、情绪、路由、FAQ ID、检索上下文、检索分数、模型调用次数、单条延迟、P95 延迟和失败类型，并将逐条结果与汇总指标保存为可复现 JSON 报告。

如果 Token usage 和价格配置都可观测，还可以描述：

> 接入 OpenAI 兼容响应 usage，统计输入 Token、输出 Token 和可选成本估算，支持按运行批次追踪模型开销。

暂时不要写：

```text
RAGAS 得分 0.87
```

因为 RAGAS 还没有真正接入。

---

## 23. 下一阶段

评估系统完成后，推荐顺序是：

```text
1. 把评估集扩展到 100 条以上
2. 增加 RAGAS 输入准备
3. 接入 RAGAS 运行器
4. 增加 FastAPI 离线接口
5. 增加浏览器页面
6. 编写 Dockerfile
7. 编写 docker-compose.yml
8. 生成评估图表
9. 增加 GitHub Actions
10. 最后同步 README 和 GitHub
```

不要跳过确定性评估直接接入 RAGAS。

当前正确的工程顺序是：

```text
确定性结果
    ↓
检索证据
    ↓
运行指标
    ↓
RAGAS
```

---

## 24. 本文档对应的验证命令

完整测试：

```powershell
Set-Location D:\codexTest\customer-service-agent-learning
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
```

规则基线：

```powershell
.\.venv\Scripts\python.exe -m src.rule_baseline_runner
```

查看报告：

```powershell
Get-Content .\reports\baselines\<run_id>\summary.json
```

查看逐条结果：

```powershell
Get-Content .\reports\baselines\<run_id>\results.json
```

---

## 25. 复习问题

1. 为什么规则基线的 `model_call_total` 应该是 `0`，而不是 `50`？

2. 为什么 `input_tokens_total` 在没有 usage 信息时应该保存为 `null`，而不是 `0`？
