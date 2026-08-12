import argparse
from pathlib import Path

from src.evaluation_cases import (
    EVALUATION_CASES,
    EvaluationCase,
    validate_evaluation_cases,
)
from src.evaluation_runner import (
    EvaluationResult,
    build_summary,
    evaluate_case,
    save_evaluation_report,
)
from src.langgraph_llm_agent import (
    run_langgraph_llm_customer_service_agent,
)


def select_evaluation_cases(
    limit: int,
) -> list[EvaluationCase]:
    # 样本数量至少为 1。
    # 否则后续运行没有业务意义，也无法验证模型链路。
    if limit < 1:
        raise ValueError("候选评估样本数量必须大于 0。")

    # 不允许请求超过当前评估集数量的样本。
    # 这样可以避免调用方误以为实际运行了 100 条，
    # 但切片实际上只返回了 50 条。
    if limit > len(EVALUATION_CASES):
        raise ValueError("候选评估样本数量超过当前评估集。")

    # 列表切片取得从第 0 条到第 limit - 1 条的样本。
    # 例如 limit 为 3 时，返回前三条评估样本。
    selected_cases = EVALUATION_CASES[:limit]

    # 在模型调用前，先验证当前子集的结构是否合法。
    # 结构错误时应停止，避免为错误数据产生 API 费用。
    validate_evaluation_cases(selected_cases)

    # 返回已经通过结构校验的评估样本。
    return selected_cases


def run_llm_candidate_evaluation(
    limit: int,
) -> tuple[list[EvaluationResult], Path]:
    # 选出并校验本次需要评估的样本。
    cases = select_evaluation_cases(limit)

    # 对每条样本调用大模型 LangGraph 工作流。
    #
    # 工作流会优先使用大模型完成分类和情绪分析；
    # 请求失败或模型输出不合法时，自动降级到本地规则。
    results = [
        evaluate_case(
            case,
            agent_runner=run_langgraph_llm_customer_service_agent,
        )
        for case in cases
    ]

    # 将候选方案的逐条结果和汇总结果保存到独立目录。
    # 每次运行都会自动生成新的 UTC 时间目录。
    report_dir = save_evaluation_report(
        results,
        output_root=Path("reports/candidates"),
    )

    # 同时返回内存中的结果和实际报告目录。
    return results, report_dir


def main() -> None:
    # 创建命令行参数解析器。
    parser = argparse.ArgumentParser(
        description="运行大模型客服 Agent 的小样本候选评估。",
    )

    # --limit 控制本次评估的样本数。
    # 默认只运行 3 条，先控制模型调用数量。
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="本次候选评估使用的样本数量，默认是 3。",
    )

    # 读取终端参数，例如 --limit 5。
    args = parser.parse_args()

    try:
        # 执行候选评估并获得真实报告目录。
        results, report_dir = run_llm_candidate_evaluation(
            args.limit,
        )
    except ValueError as error:
        # 参数或评估数据无效时，以命令行错误形式结束。
        # 此时尚未启动大模型调用。
        parser.error(str(error))

    # 汇总本次实际评估结果。
    summary = build_summary(results)

    # 输出本次候选方案的整体通过数量。
    print(
        "大模型候选评估完成："
        f"{summary['passed']}/{summary['total']}"
    )

    # 输出来源分布，用于观察模型成功、规则降级或 FAQ 回退。
    print(f"分析来源：{summary['analysis_source_counts']}")
    print(f"回复来源：{summary['response_source_counts']}")

    # 输出报告保存位置。
    print(f"报告目录：{report_dir}")


if __name__ == "__main__":
    # 仅在 python -m src.llm_candidate_runner 时运行评估。
    main()