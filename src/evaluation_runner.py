from collections import Counter
from typing import TypedDict

from src.evaluation_cases import EVALUATION_CASES, EvaluationCase
from src.langgraph_llm_agent import run_langgraph_llm_customer_service_agent


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

    # 分类和情绪分析来源，例如 llm 或 rule_fallback。
    analysis_source: str

    # 最终回复来源，例如 llm、faq_fallback 或 local。
    response_source: str


def evaluate_case(case: EvaluationCase) -> EvaluationResult:
    # 运行一条用户问题，得到客服 Agent 的最终状态。
    final_state = run_langgraph_llm_customer_service_agent(case["query"])

    # 判断最终状态中是否存在 faq_answer 这个键。
    actual_faq_in_state = "faq_answer" in final_state

    # 分别比较实际分类、情绪、路线、FAQ 命中状态是否符合预期。
    category_ok = final_state["category"] == case["expected_category"]
    sentiment_ok = final_state["sentiment"] == case["expected_sentiment"]
    route_ok = final_state["route"] == case["expected_route"]
    faq_ok = actual_faq_in_state == case["expected_faq_in_state"]

    # 只有四个检查全部通过，这条样本才算整体通过。
    passed = category_ok and sentiment_ok and route_ok and faq_ok

    # 把这一条样本的评估结果整理成一个字典返回。
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
        "analysis_source": final_state.get("analysis_source", "unknown"),
        "response_source": final_state.get("response_source", "unknown"),
    }


def evaluate_all_cases() -> list[EvaluationResult]:
    # 遍历全部评估样本，并逐条执行 evaluate_case。
    return [evaluate_case(case) for case in EVALUATION_CASES]


def count_passed(results: list[EvaluationResult]) -> int:
    # 统计 passed 为 True 的样本数量。
    return sum(1 for result in results if result["passed"])

def format_rate(correct: int, total: int) -> str:
    # 没有样本时，避免出现除零错误。
    if total == 0:
        return "0/0 (0.0%)"

    # 将正确数量、总数量和百分比组合成可读文本。
    return f"{correct}/{total} ({correct / total:.1%})"

def print_summary(results: list[EvaluationResult]) -> None:
    # 统计样本总数。
    total = len(results)

    # 分别统计四类检查通过了多少条。
    category_correct = sum(
        1 for result in results if result["category_ok"]
    )
    sentiment_correct = sum(
        1 for result in results if result["sentiment_ok"]
    )
    route_correct = sum(
        1 for result in results if result["route_ok"]
    )
    faq_correct = sum(
        1 for result in results if result["faq_ok"]
    )

    # 统计分析阶段和回复阶段分别使用了哪些来源。
    analysis_counts = Counter(
        result["analysis_source"] for result in results
    )
    response_counts = Counter(
        result["response_source"] for result in results
    )

    # 统计分析阶段发生规则降级的样本数量。
    rule_fallback_count = analysis_counts.get(
        "rule_fallback",
        0,
    )

    # 统计回复阶段使用 FAQ 兜底的样本数量。
    faq_fallback_count = response_counts.get(
        "faq_fallback",
        0,
    )

    print("\n--- 汇总指标 ---")
    print(
        "分类准确率："
        f"{format_rate(category_correct, total)}"
    )
    print(
        "情绪准确率："
        f"{format_rate(sentiment_correct, total)}"
    )
    print(
        "路由准确率："
        f"{format_rate(route_correct, total)}"
    )
    print(
        "FAQ 状态准确率："
        f"{format_rate(faq_correct, total)}"
    )
    print(f"分析来源统计：{dict(analysis_counts)}")
    print(f"回复来源统计：{dict(response_counts)}")
    print(f"分析降级次数：{rule_fallback_count}")
    print(f"回复降级次数：{faq_fallback_count}")


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
        print(f"分析来源：{result['analysis_source']}")
        print(f"回复来源：{result['response_source']}")


def main() -> None:
    # 执行全部样本评估。
    results = evaluate_all_cases()

    # 先输出整体统计指标。
    print_summary(results)

    # 再输出每条样本的详细结果。
    print_report(results)


if __name__ == "__main__":
    # 只有直接运行 python -m src.evaluation_runner 时，才执行评估。
    main()
