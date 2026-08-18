from pathlib import Path

from src.evaluation_runner import (
    build_summary,
    evaluate_all_cases,
    save_evaluation_report,
)
from src.langgraph_agent import run_langgraph_customer_service_agent


def main() -> None:
    # 使用规则版 LangGraph Agent 执行全部评估样本。
    #
    # 这个 Agent 使用本地分类、情绪判断、路由和 FAQ 检索，
    # 不会请求大模型 API。
    results = evaluate_all_cases(
        agent_runner=run_langgraph_customer_service_agent,
    )

    # 对全部评估结果进行汇总。
    summary = build_summary(results)

    # 将逐条结果和汇总结果保存到 reports/baselines 目录。
    #
    # 不传 run_id 时，save_evaluation_report 会使用 UTC 时间
    # 自动生成独立目录，避免覆盖旧报告。
    report_dir = save_evaluation_report(
        results,
        output_root=Path("reports/baselines"),
        metadata={
            "runner": "rule_baseline",
            "mode": "offline",
            "model_name": "rule_based",
        },
    )

    # 在终端展示最重要的基线结果。
    print(
        "规则基线评估完成："
        f"{summary['passed']}/{summary['total']}"
    )

    # 输出实际报告目录，方便后续查看 JSON 文件。
    print(f"报告目录：{report_dir}")


if __name__ == "__main__":
    # 只有执行 python -m src.rule_baseline_runner 时，
    # 才运行完整规则基线评估。
    main()
