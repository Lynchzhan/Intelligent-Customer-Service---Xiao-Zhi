import json
import math
from collections import Counter
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path
from typing import Callable, NotRequired, TypedDict, cast
from src.agent import CustomerState

from src.evaluation_cases import (
    EVALUATION_CASES,
    Complexity,
    EvaluationCase,
    validate_evaluation_cases,
)
from src.knowledge_base import FaqId, RetrievalCandidate
from src.langgraph_llm_agent import run_langgraph_llm_customer_service_agent


# 定义 Agent 执行函数的统一接口。
#
# 只要函数接收一个字符串问题，
# 并返回一个 CustomerState，
# 就可以被评估器当作待测 Agent 使用。
AgentRunner = Callable[[str], CustomerState]

class EvaluationResult(TypedDict):
    # 样本名称，来自 EvaluationCase["name"]。
    name: str

    # 用户原始问题，来自 EvaluationCase["query"]。
    query: str

    # 这条样本整体是否通过。
    passed: bool

    # 分类字段是否符合预期。
    category_ok: bool

    # 情绪字段是否符合预期。
    sentiment_ok: bool

    # 路由字段是否符合预期。
    route_ok: bool

    # FAQ 是否命中的状态是否符合预期。
    faq_ok: bool

    # 实际分类结果。
    actual_category: str

    # 预期分类结果。
    expected_category: str

    # 实际情绪结果。
    actual_sentiment: str

    # 预期情绪结果。
    expected_sentiment: str

    # 实际路由结果。
    actual_route: str

    # 预期路由结果。
    expected_route: str

    # 实际最终状态中是否存在 faq_answer。
    actual_faq_in_state: bool

    # 预期最终状态中是否存在 faq_answer。
    expected_faq_in_state: bool

    # 实际命中的 FAQ 标识；未命中时为 None。
    actual_faq_id: FaqId | None

    # 人工标注的预期 FAQ 标识；不应命中时为 None。
    expected_faq_id: FaqId | None

    # 实际 FAQ 标识是否与预期一致。
    faq_id_ok: bool

    # 实际检索到的知识上下文。
    #
    # 真实评估结果会始终记录这个字段。
    # 使用 NotRequired 是为了兼容只测试汇总逻辑的旧测试夹具。
    retrieved_contexts: NotRequired[list[str]]

    # 实际检索候选的相关性分数。
    #
    # FAQ 未命中时保存 None，而不是伪造一个分数。
    retrieval_score: NotRequired[float | None]

    # 混合检索的两部分证据。FAQ 未命中时均为 None。
    retrieval_keyword_score: NotRequired[float | None]
    retrieval_text_score: NotRequired[float | None]

    # 可追踪本次使用了哪种算法、哪版知识库，以及保留的 Top-K 候选摘要。
    retrieval_method: NotRequired[str | None]
    retrieval_candidates: NotRequired[list[RetrievalCandidate]]
    knowledge_base_name: NotRequired[str | None]
    knowledge_base_version: NotRequired[str | None]

    # 样本难度和标签会被复制到结果中，方便分组统计。
    complexity: Complexity
    tags: list[str]

    # 分类和情绪分析来源，例如 llm 或 rule_fallback。
    analysis_source: str

    # 分类阶段发生的错误类型；没有错误时字段可以不存在。
    analysis_error: NotRequired[str | None]

    # 最终回复来源，例如 llm、faq_fallback 或 local。
    response_source: str

    # 回复阶段发生的错误类型；没有错误时字段可以不存在。
    response_error: NotRequired[str | None]

    # 单条样本从调用 Agent 到返回最终状态的耗时。
    latency_ms: NotRequired[float]

    # 分类阶段实际尝试的大模型调用次数。
    analysis_model_calls: NotRequired[int]

    # 回复阶段实际尝试的大模型调用次数。
    response_model_calls: NotRequired[int]

    # 当前样本的大模型调用总次数。
    model_call_count: NotRequired[int]

    # 如果模型客户端提供 usage 信息，则保存输入 Token 数。
    input_tokens: NotRequired[int | None]

    # 如果模型客户端提供 usage 信息，则保存输出 Token 数。
    output_tokens: NotRequired[int | None]

    # 如果调用方提供价格配置，则保存本条样本的估算成本。
    estimated_cost_usd: NotRequired[float | None]


class ComparisonMetric(TypedDict):
    # 基线方案正确的样本数。
    baseline_correct: int

    # 候选方案正确的样本数。
    candidate_correct: int

    # 两套方案共同评估的样本总数。
    total: int

    # 基线方案正确率，范围是 0.0 到 1.0。
    baseline_rate: float

    # 候选方案正确率，范围是 0.0 到 1.0。
    candidate_rate: float

    # 候选方案相对于基线的绝对提升。
    absolute_delta: float

    # 候选方案相对于基线的相对提升。
    # 基线为 0 时无法计算，因此允许为 None。
    relative_improvement: float | None


class ComparisonSummary(TypedDict):
    # 两套方案共同使用的样本总数。
    total: int

    # 每个指标的比较结果。
    metrics: dict[str, ComparisonMetric]


class EvaluationSummary(TypedDict):
    # 本次评估运行的样本总数。
    total: int

    # 四项检查都通过的样本数量。
    passed: int

    # 各项独立指标的正确数量。
    category_correct: int
    sentiment_correct: int
    route_correct: int
    faq_correct: int
    faq_id_correct: int

    # 分析阶段和回复阶段的来源分布。
    analysis_source_counts: dict[str, int]
    response_source_counts: dict[str, int]

    # 两种降级策略各自发生的次数。
    rule_fallback_count: int
    faq_fallback_count: int

    # 按复杂度和标签划分的分组指标。
    group_metrics: "GroupMetrics"

    # 运行效率指标。
    latency_observation_count: int
    latency_ms_average: float
    latency_ms_p95: float
    model_call_total: int
    model_call_average: float

    # Token 和成本指标。
    #
    # 如果模型响应没有提供 usage 信息，则使用 None，
    # 明确表示“没有观测到”，而不是“实际为零”。
    input_tokens_total: int | None
    output_tokens_total: int | None
    estimated_cost_usd_total: float | None
    token_observation_count: int
    cost_observation_count: int

    # 失败类型统计。
    timeout_count: int
    parse_failure_count: int
    failure_type_counts: dict[str, int]

    # RAG 检索可观测性。只统计真实命中，不把 FAQ 未命中伪装成零分检索。
    retrieval_hit_count: int
    retrieval_method_counts: dict[str, int]
    knowledge_base_version_counts: dict[str, int]


class GroupMetric(TypedDict):
    # 该分组包含的样本总数。
    total: int

    # 该分组中整体通过的样本数。
    passed: int

    # 该分组的通过率，范围为 0.0 到 1.0。
    pass_rate: float


class GroupMetrics(TypedDict):
    # 按 simple、medium、complex 分组。
    complexity: dict[str, GroupMetric]

    # 按 billing、faq_hit、boundary 等标签分组。
    tags: dict[str, GroupMetric]


def _calculate_p95(values: list[float]) -> float:
    """使用最近秩方法计算一个简单、可解释的 P95。"""

    # 没有样本时，返回 0.0，避免除零或索引错误。
    if not values:
        return 0.0

    # 排序后取覆盖 95% 样本的最小位置。
    ordered = sorted(values)
    rank = math.ceil(len(ordered) * 0.95) - 1
    index = max(0, min(rank, len(ordered) - 1))
    return ordered[index]


def _infer_model_call_counts(
    analysis_source: str,
    response_source: str,
) -> tuple[int, int]:
    """根据当前工作流的来源字段推断两个阶段的调用次数。"""

    # llm 表示分类模型成功调用；
    # rule_fallback 表示分类模型先尝试、随后才降级。
    analysis_calls = (
        1
        if analysis_source in {"llm", "rule_fallback"}
        else 0
    )

    # llm 表示回复模型成功调用；
    # faq_fallback 表示回复模型先尝试、随后回退到 FAQ。
    response_calls = (
        1
        if response_source in {"llm", "faq_fallback"}
        else 0
    )

    return analysis_calls, response_calls


def evaluate_case(
    case: EvaluationCase,
    agent_runner: AgentRunner | None = None,
) -> EvaluationResult:
    # 如果调用方没有传入特定 Agent，
    # 默认使用当前的大模型客服工作流。
    #
    # 这里故意使用 None，而不是把函数直接写成默认参数：
    #
    # agent_runner: AgentRunner = run_langgraph_llm_customer_service_agent
    #
    # 因为单元测试可能会临时替换
    # run_langgraph_llm_customer_service_agent。
    # 使用 None 可以让函数在真正调用时读取当前对象。
    if agent_runner is None:
        agent_runner = run_langgraph_llm_customer_service_agent

    # 记录单条样本的开始时间。
    # 这里测量的是完整 Agent 调用耗时，而不是某一个节点耗时。
    started_at = perf_counter()

    # 使用传入的 Agent 执行用户问题。
    final_state = agent_runner(case["query"])

    # 将高精度计时器转换成毫秒，方便写入 JSON 和终端。
    latency_ms = round(
        (perf_counter() - started_at) * 1000,
        3,
    )

    # 判断最终状态中是否存在 faq_answer 这个键。
    actual_faq_in_state = "faq_answer" in final_state

    # 读取实际 FAQ 标识。
    actual_faq_id = final_state.get("faq_id")

    # 比较实际分类、情绪、路线和 FAQ 状态。
    category_ok = final_state["category"] == case["expected_category"]
    sentiment_ok = final_state["sentiment"] == case["expected_sentiment"]
    route_ok = final_state["route"] == case["expected_route"]
    faq_ok = actual_faq_in_state == case["expected_faq_in_state"]
    faq_id_ok = actual_faq_id == case["expected_faq_id"]

    # 读取来源字段，后面用于推断模型调用次数。
    analysis_source = final_state.get(
        "analysis_source",
        "unknown",
    )
    response_source = final_state.get(
        "response_source",
        "unknown",
    )
    analysis_model_calls, response_model_calls = (
        _infer_model_call_counts(
            analysis_source,
            response_source,
        )
    )

    # 五项检查全部通过，样本才算整体通过。
    passed = (
        category_ok
        and sentiment_ok
        and route_ok
        and faq_ok
        and faq_id_ok
    )

    # 返回这一条样本的结构化评估结果。
    return {
        "name": case["name"],
        "query": case["query"],
        "passed": passed,
        "category_ok": category_ok,
        "sentiment_ok": sentiment_ok,
        "route_ok": route_ok,
        "faq_ok": faq_ok,
        "actual_category": final_state["category"],
        "expected_category": case["expected_category"],
        "actual_sentiment": final_state["sentiment"],
        "expected_sentiment": case["expected_sentiment"],
        "actual_route": final_state["route"],
        "expected_route": case["expected_route"],
        "actual_faq_in_state": actual_faq_in_state,
        "expected_faq_in_state": case["expected_faq_in_state"],
        "actual_faq_id": actual_faq_id,
        "expected_faq_id": case["expected_faq_id"],
        "faq_id_ok": faq_id_ok,
        "retrieved_contexts": final_state.get(
            "retrieved_contexts",
            [],
        ),
        "retrieval_score": final_state.get(
            "retrieval_score",
        ),
        "retrieval_keyword_score": final_state.get(
            "retrieval_keyword_score",
        ),
        "retrieval_text_score": final_state.get(
            "retrieval_text_score",
        ),
        "retrieval_method": final_state.get(
            "retrieval_method",
        ),
        "retrieval_candidates": final_state.get(
            "retrieval_candidates",
            [],
        ),
        "knowledge_base_name": final_state.get(
            "knowledge_base_name",
        ),
        "knowledge_base_version": final_state.get(
            "knowledge_base_version",
        ),
        "complexity": case["complexity"],
        "tags": case["tags"],
        "analysis_source": analysis_source,
        "analysis_error": final_state.get("analysis_error"),
        "response_source": response_source,
        "response_error": final_state.get("response_error"),
        "latency_ms": latency_ms,
        "analysis_model_calls": analysis_model_calls,
        "response_model_calls": response_model_calls,
        "model_call_count": (
            analysis_model_calls + response_model_calls
        ),
        "input_tokens": final_state.get("input_tokens"),
        "output_tokens": final_state.get("output_tokens"),
        "estimated_cost_usd": final_state.get(
            "estimated_cost_usd",
        ),
    }


def evaluate_all_cases(
    agent_runner: AgentRunner | None = None,
) -> list[EvaluationResult]:
    # 在执行任何 Agent 之前，先检查所有评估样本。
    # 如果数据非法，直接停止，不调用模型。
    validate_evaluation_cases(EVALUATION_CASES)

    # 使用同一个 Agent 执行所有评估样本。
    #
    # 这样规则基线和大模型 Agent
    # 会使用完全相同的测试集和评估逻辑。
    return [
        evaluate_case(
            case,
            agent_runner=agent_runner,
        )
        for case in EVALUATION_CASES
    ]


def count_passed(results: list[EvaluationResult]) -> int:
    # 统计 passed 为 True 的样本数量。
    return sum(1 for result in results if result["passed"])


def _record_group_metric(
    metrics: dict[str, GroupMetric],
    group_name: str,
    passed: bool,
) -> None:
    # 第一次看到某个分组时，先创建它的计数器。
    metric = metrics.setdefault(
        group_name,
        {
            "total": 0,
            "passed": 0,
            "pass_rate": 0.0,
        },
    )

    # 每条样本只为所属分组贡献一次总数。
    metric["total"] += 1

    # 只有整体通过的样本才增加 passed。
    if passed:
        metric["passed"] += 1

    # 每次更新后重新计算分组通过率。
    metric["pass_rate"] = metric["passed"] / metric["total"]


def build_group_metrics(
    results: list[EvaluationResult],
) -> GroupMetrics:
    # 分别创建复杂度和标签两套分组计数器。
    complexity_metrics: dict[str, GroupMetric] = {}
    tag_metrics: dict[str, GroupMetric] = {}

    for result in results:
        # 一条样本只属于一个复杂度分组。
        _record_group_metric(
            complexity_metrics,
            result["complexity"],
            result["passed"],
        )

        # 一条样本可以同时属于多个标签分组。
        for tag in result["tags"]:
            _record_group_metric(
                tag_metrics,
                tag,
                result["passed"],
            )

    return {
        "complexity": complexity_metrics,
        "tags": tag_metrics,
    }


def build_summary(results: list[EvaluationResult]) -> EvaluationSummary:
    # 统计本次评估包含多少条结果。
    total = len(results)

    # 统计整体通过数和四项独立检查的正确数。
    passed = count_passed(results)
    category_correct = sum(1 for result in results if result["category_ok"])
    sentiment_correct = sum(1 for result in results if result["sentiment_ok"])
    route_correct = sum(1 for result in results if result["route_ok"])
    faq_correct = sum(1 for result in results if result["faq_ok"])
    faq_id_correct = sum(1 for result in results if result["faq_id_ok"])

    # Counter 统计每一种分析来源和回复来源出现的次数。
    analysis_source_counts = dict(
        Counter(result["analysis_source"] for result in results)
    )
    response_source_counts = dict(
        Counter(result["response_source"] for result in results)
    )

    # get 的第二个参数 0 表示该来源不存在时按零次处理。
    rule_fallback_count = analysis_source_counts.get("rule_fallback", 0)
    faq_fallback_count = response_source_counts.get("faq_fallback", 0)

    # 生成复杂度和标签维度的分组指标。
    group_metrics = build_group_metrics(results)

    # 读取每条结果的耗时。
    latency_values = [
        float(result["latency_ms"])
        for result in results
        if (
            "latency_ms" in result
            and result["latency_ms"] is not None
        )
    ]

    # 读取每条结果推断出的模型调用次数。
    model_call_values = [
        int(
            result.get(
                "model_call_count",
                sum(
                    _infer_model_call_counts(
                        result["analysis_source"],
                        result["response_source"],
                    )
                ),
            )
        )
        for result in results
    ]

    # 只有真正观测到 Token 时，才计算总量。
    input_token_values = [
        result["input_tokens"]
        for result in results
        if result.get("input_tokens") is not None
    ]
    output_token_values = [
        result["output_tokens"]
        for result in results
        if result.get("output_tokens") is not None
    ]
    cost_values = [
        result["estimated_cost_usd"]
        for result in results
        if result.get("estimated_cost_usd") is not None
    ]

    # 收集分析阶段和回复阶段的错误类型。
    failure_types = [
        error_name
        for result in results
        for error_name in (
            result.get("analysis_error"),
            result.get("response_error"),
        )
        if error_name
    ]
    failure_type_counts = dict(Counter(failure_types))

    # 超时和解析失败分别统计，便于定位质量问题的来源。
    timeout_names = {
        "APITimeoutError",
        "TimeoutError",
    }
    timeout_count = sum(
        1
        for error_name in failure_types
        if error_name in timeout_names
    )
    parse_failure_count = sum(
        1
        for error_name in failure_types
        if error_name == "ValueError"
    )

    # 仅在状态真正存在 faq_answer 时算作检索命中。
    # 这样 FAQ 未命中不会被误解成“检索成功但分数为零”。
    retrieval_hit_results = [
        result
        for result in results
        if result["actual_faq_in_state"]
    ]
    retrieval_method_counts = dict(
        Counter(
            method
            for result in retrieval_hit_results
            if (method := result.get("retrieval_method"))
        )
    )
    knowledge_base_version_counts = dict(
        Counter(
            version
            for result in retrieval_hit_results
            if (version := result.get("knowledge_base_version"))
        )
    )

    # 返回普通 Python 字典，后续可以直接写入 JSON 文件。
    return {
        "total": total,
        "passed": passed,
        "category_correct": category_correct,
        "sentiment_correct": sentiment_correct,
        "route_correct": route_correct,
        "faq_correct": faq_correct,
        "faq_id_correct": faq_id_correct,
        "analysis_source_counts": analysis_source_counts,
        "response_source_counts": response_source_counts,
        "rule_fallback_count": rule_fallback_count,
        "faq_fallback_count": faq_fallback_count,
        "group_metrics": group_metrics,
        "latency_observation_count": len(latency_values),
        "latency_ms_average": (
            sum(latency_values) / len(latency_values)
            if latency_values
            else 0.0
        ),
        "latency_ms_p95": _calculate_p95(latency_values),
        "model_call_total": sum(model_call_values),
        "model_call_average": (
            sum(model_call_values) / total
            if total
            else 0.0
        ),
        "input_tokens_total": (
            sum(input_token_values)
            if input_token_values
            else None
        ),
        "output_tokens_total": (
            sum(output_token_values)
            if output_token_values
            else None
        ),
        "estimated_cost_usd_total": (
            round(sum(cost_values), 8)
            if cost_values
            else None
        ),
        "token_observation_count": sum(
            1
            for result in results
            if (
                result.get("input_tokens") is not None
                or result.get("output_tokens") is not None
            )
        ),
        "cost_observation_count": len(cost_values),
        "timeout_count": timeout_count,
        "parse_failure_count": parse_failure_count,
        "failure_type_counts": failure_type_counts,
        "retrieval_hit_count": len(retrieval_hit_results),
        "retrieval_method_counts": retrieval_method_counts,
        "knowledge_base_version_counts": knowledge_base_version_counts,
    }


def _build_comparison_metric(
    baseline_correct: int,
    candidate_correct: int,
    total: int,
) -> ComparisonMetric:
    # 没有样本时，避免除零错误。
    if total == 0:
        return {
            "baseline_correct": baseline_correct,
            "candidate_correct": candidate_correct,
            "total": 0,
            "baseline_rate": 0.0,
            "candidate_rate": 0.0,
            "absolute_delta": 0.0,
            "relative_improvement": None,
        }

    # 计算基线方案正确率。
    baseline_rate = baseline_correct / total

    # 计算候选方案正确率。
    candidate_rate = candidate_correct / total

    # 绝对提升是两个正确率直接相减。
    absolute_delta = candidate_rate - baseline_rate

    # 基线正确率为 0 时，不能计算相对提升。
    if baseline_rate == 0:
        relative_improvement = None
    else:
        relative_improvement = absolute_delta / baseline_rate

    return {
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "total": total,
        "baseline_rate": baseline_rate,
        "candidate_rate": candidate_rate,
        "absolute_delta": absolute_delta,
        "relative_improvement": relative_improvement,
    }


def build_comparison(
    baseline_results: list[EvaluationResult],
    candidate_results: list[EvaluationResult],
) -> ComparisonSummary:
    # 两套结果必须包含相同数量的样本。
    if len(baseline_results) != len(candidate_results):
        raise ValueError("基线和候选方案的样本数量不一致。")

    # 提取两套结果中的样本名称。
    baseline_names = [
        result["name"]
        for result in baseline_results
    ]
    candidate_names = [
        result["name"]
        for result in candidate_results
    ]

    # 防止把不同问题的结果进行错误比较。
    if baseline_names != candidate_names:
        raise ValueError("基线和候选方案的样本名称或顺序不一致。")

    # 分别汇总两套结果。
    baseline_summary = build_summary(baseline_results)
    candidate_summary = build_summary(candidate_results)

    # 指标名称对应 EvaluationSummary 中的字段。
    metric_fields = {
        "overall": "passed",
        "category": "category_correct",
        "sentiment": "sentiment_correct",
        "route": "route_correct",
        "faq": "faq_correct",
        "faq_id": "faq_id_correct",
    }

    metrics: dict[str, ComparisonMetric] = {}

    # 对所有指标使用同一套计算逻辑。
    for metric_name, field_name in metric_fields.items():
        metrics[metric_name] = _build_comparison_metric(
            baseline_correct=baseline_summary[field_name],
            candidate_correct=candidate_summary[field_name],
            total=baseline_summary["total"],
        )

    return {
        "total": baseline_summary["total"],
        "metrics": metrics,
    }

def load_evaluation_results(
    report_dir: Path,
) -> list[EvaluationResult]:
    # 拼接报告目录与固定文件名，得到 results.json 的完整路径。
    results_path = report_dir / "results.json"

    try:
        # 使用 UTF-8 读取 JSON 文本，再解析为 Python 数据。
        payload = json.loads(
            results_path.read_text(encoding="utf-8")
        )
    except FileNotFoundError as error:
        # 报告目录或 results.json 不存在时，
        # 转换为更清晰的业务错误。
        raise ValueError(
            f"评估结果文件不存在：{results_path}"
        ) from error
    except json.JSONDecodeError as error:
        # 文件存在但内容不是合法 JSON 时，
        # 不能继续把它当作评估结果使用。
        raise ValueError(
            f"评估结果文件不是合法 JSON：{results_path}"
        ) from error

    # results.json 的顶层结构必须是列表。
    #
    # 正确结构：
    # [
    #   {...第一条评估结果...},
    #   {...第二条评估结果...}
    # ]
    #
    # 错误结构示例：
    # {"summary": {...}}
    if not isinstance(payload, list):
        raise ValueError(
            "评估结果文件顶层结构必须是 JSON 列表。"
        )

    # json.loads 的返回类型在类型系统中较宽泛。
    # 这里把已确认是列表的 payload 标记为评估结果列表，
    # 供后续 build_comparison() 使用。
    return cast(list[EvaluationResult], payload)


def save_evaluation_report(
    results: list[EvaluationResult],
    output_root: Path = Path("reports/runs"),
    run_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> Path:
    # 没有显式传入 run_id 时，使用 UTC 时间生成本次运行的目录名。
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # 每次运行独立保存，避免后一次评估覆盖前一次评估。
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 两个 JSON 文件使用同一个时间值，保证元数据一致。
    created_at = datetime.now(timezone.utc).isoformat()

    # results.json 保存每一条样本的完整实际结果和预期结果。
    results_path = run_dir / "results.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # summary.json 保存汇总指标和本次运行的最小元数据。
    summary_payload = {
        "run_id": run_id,
        "created_at": created_at,
        "sample_count": len(results),
        "metadata": metadata or {},
        "summary": build_summary(results),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 返回目录路径，后续绘图和 README 生成可以继续使用它。
    return run_dir


def save_comparison_report(
    baseline_results: list[EvaluationResult],
    candidate_results: list[EvaluationResult],
    output_root: Path = Path("reports/comparisons"),
    comparison_id: str | None = None,
) -> Path:
    # 先执行比较。
    #
    # 这一步会检查：
    # 1. 两套结果数量是否相同。
    # 2. 两套结果的 name 和顺序是否一致。
    #
    # 如果比较条件不满足，函数会立即抛出 ValueError，
    # 不会创建不完整的报告目录。
    comparison = build_comparison(
        baseline_results,
        candidate_results,
    )

    # 分别生成两套方案的汇总结果。
    baseline_summary = build_summary(baseline_results)
    candidate_summary = build_summary(candidate_results)

    # 如果调用方没有传入比较 ID，
    # 使用当前 UTC 时间创建唯一目录名。
    if comparison_id is None:
        comparison_id = datetime.now(timezone.utc).strftime(
            "%Y%m%d-%H%M%S"
        )

    # 每次比较使用独立目录，避免覆盖旧报告。
    comparison_dir = output_root / comparison_id
    comparison_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 两套文件和汇总文件使用相同的创建时间。
    created_at = datetime.now(timezone.utc).isoformat()

    # 保存基线方案的逐条结果。
    baseline_path = comparison_dir / "baseline_results.json"
    baseline_path.write_text(
        json.dumps(
            baseline_results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 保存候选方案的逐条结果。
    candidate_path = comparison_dir / "candidate_results.json"
    candidate_path.write_text(
        json.dumps(
            candidate_results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 保存汇总和比较结果。
    comparison_payload = {
        "comparison_id": comparison_id,
        "created_at": created_at,
        "sample_count": len(baseline_results),
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "comparison": comparison,
    }

    comparison_path = comparison_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            comparison_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 返回本次比较报告目录。
    return comparison_dir


def format_rate(correct: int, total: int) -> str:
    # 没有样本时，避免出现除零错误。
    if total == 0:
        return "0/0 (0.0%)"

    # 将正确数量、总数量和百分比组合成可读文本。
    return f"{correct}/{total} ({correct / total:.1%})"

def print_summary(results: list[EvaluationResult]) -> None:
    # 统一计算结构化指标，避免打印逻辑和后续 JSON 逻辑重复统计。
    summary = build_summary(results)

    print("\n--- 汇总指标 ---")
    print(f"分类准确率：{format_rate(summary['category_correct'], summary['total'])}")
    print(f"情绪准确率：{format_rate(summary['sentiment_correct'], summary['total'])}")
    print(f"路由准确率：{format_rate(summary['route_correct'], summary['total'])}")
    print(f"FAQ 状态准确率：{format_rate(summary['faq_correct'], summary['total'])}")
    print(f"FAQ ID 准确率：{format_rate(summary['faq_id_correct'], summary['total'])}")
    print(f"分析来源统计：{summary['analysis_source_counts']}")
    print(f"回复来源统计：{summary['response_source_counts']}")
    print(f"分析降级次数：{summary['rule_fallback_count']}")
    print(f"回复降级次数：{summary['faq_fallback_count']}")
    print(
        "延迟观测样本数："
        f"{summary['latency_observation_count']}"
    )
    print(
        "平均延迟："
        f"{summary['latency_ms_average']:.3f} ms"
    )
    print(
        "P95 延迟："
        f"{summary['latency_ms_p95']:.3f} ms"
    )
    print(f"模型调用总次数：{summary['model_call_total']}")
    print(
        "平均每条模型调用次数："
        f"{summary['model_call_average']:.3f}"
    )
    print(f"输入 Token 总数：{summary['input_tokens_total']}")
    print(f"输出 Token 总数：{summary['output_tokens_total']}")
    print(
        "估算成本（美元）："
        f"{summary['estimated_cost_usd_total']}"
    )
    print(f"Token 观测样本数：{summary['token_observation_count']}")
    print(f"成本观测样本数：{summary['cost_observation_count']}")
    print(f"超时次数：{summary['timeout_count']}")
    print(f"解析失败次数：{summary['parse_failure_count']}")
    print(f"失败类型统计：{summary['failure_type_counts']}")
    print(f"RAG 检索命中次数：{summary['retrieval_hit_count']}")
    print(f"RAG 检索方法统计：{summary['retrieval_method_counts']}")
    print(
        "知识库版本统计："
        f"{summary['knowledge_base_version_counts']}"
    )
    print(f"复杂度分组指标：{summary['group_metrics']['complexity']}")
    print(f"标签分组指标：{summary['group_metrics']['tags']}")


def print_report(results: list[EvaluationResult]) -> None:
    # 先输出整体通过数量。
    total = len(results)
    passed = count_passed(results)
    print(f"评估结果：{passed}/{total} 通过")

    # 再逐条输出每个样本的详细对比结果。
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"

        print()
        print(f"[{status}] {result['name']}")
        print(f"问题：{result['query']}")
        print(
            "分类："
            f"{result['actual_category']} / 预期 {result['expected_category']} "
            f"({'OK' if result['category_ok'] else 'FAIL'})"
        )
        print(
            "情绪："
            f"{result['actual_sentiment']} / 预期 {result['expected_sentiment']} "
            f"({'OK' if result['sentiment_ok'] else 'FAIL'})"
        )
        print(
            "路线："
            f"{result['actual_route']} / 预期 {result['expected_route']} "
            f"({'OK' if result['route_ok'] else 'FAIL'})"
        )
        print(
            "FAQ："
            f"{result['actual_faq_in_state']} / 预期 {result['expected_faq_in_state']} "
            f"({'OK' if result['faq_ok'] else 'FAIL'})"
        )
        print(
            "FAQ ID："
            f"{result['actual_faq_id']} / 预期 {result['expected_faq_id']} "
            f"({'OK' if result['faq_id_ok'] else 'FAIL'})"
        )
        print(
            "检索上下文数量："
            f"{len(result.get('retrieved_contexts', []))}"
        )
        print(
            "检索分数："
            f"{result.get('retrieval_score')}"
        )
        print(
            "单条耗时："
            f"{(result.get('latency_ms') or 0.0):.3f} ms"
        )
        print(
            "模型调用次数："
            f"{result.get('model_call_count', 0)}"
        )
        print(f"分析来源：{result['analysis_source']}")
        print(f"分析错误：{result.get('analysis_error')}")
        print(f"回复来源：{result['response_source']}")
        print(f"回复错误：{result.get('response_error')}")


def main() -> None:
    # 执行全部样本评估。
    results = evaluate_all_cases()

    # 先输出整体统计指标。
    print_summary(results)

    # 再输出每条样本的详细结果。
    print_report(results)

    # 将本次评估的明细和汇总保存为结构化 JSON 文件。
    report_dir = save_evaluation_report(results)
    print(f"评估报告已保存：{report_dir}")


if __name__ == "__main__":
    # 只有直接运行 python -m src.evaluation_runner 时，才执行评估。
    main()
