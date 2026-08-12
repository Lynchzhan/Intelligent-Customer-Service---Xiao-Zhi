import argparse
from pathlib import Path

from src.evaluation_runner import (
    ComparisonSummary,
    build_comparison,
    load_evaluation_results,
    save_comparison_report,
)


def compare_report_directories(
    baseline_dir: Path,
    candidate_dir: Path,
    output_root: Path = Path("reports/comparisons"),
) -> tuple[ComparisonSummary, Path]:
    """读取两份已保存的评估结果，完成正式比较并保存报告。"""

    # 从规则基线报告目录读取 results.json。
    # 这只是读取磁盘中的 JSON 文件，不运行 Agent，也不请求模型。
    baseline_results = load_evaluation_results(baseline_dir)

    # 从大模型候选报告目录读取 results.json。
    # 同样只读取历史快照，因此不会产生 API 费用。
    candidate_results = load_evaluation_results(candidate_dir)

    # 在内存中比较两套结果。
    # build_comparison() 会验证：
    # 1. 两套报告的样本数相同；
    # 2. 样本 name 相同；
    # 3. 样本顺序相同。
    comparison = build_comparison(
        baseline_results,
        candidate_results,
    )

    # 校验通过后，把基线明细、候选明细和比较指标写入新目录。
    # output_root 允许单元测试传入临时目录，避免污染真实 reports/。
    report_dir = save_comparison_report(
        baseline_results,
        candidate_results,
        output_root=output_root,
    )

    # 返回内存中的比较结果和实际写入的报告目录。
    # main()、单元测试、后续图表脚本都可以复用这两个结果。
    return comparison, report_dir


def main() -> None:
    # 创建命令行参数解析器。
    # 这个程序只读取已经存在的报告，不执行 Agent。
    parser = argparse.ArgumentParser(
        description="比较已保存的规则基线和大模型候选评估报告。",
    )

    # 基线报告目录必须由调用方明确提供。
    # 目录中必须包含 results.json。
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
        help="规则基线报告目录。",
    )

    # 候选报告目录同样必须明确提供。
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        required=True,
        help="大模型候选报告目录。",
    )

    # 读取终端传入的参数。
    args = parser.parse_args()

    try:
        # 调用可复用函数完成“读取 -> 校验 -> 比较 -> 保存”。
        # 传入命令行给出的两个历史报告目录。
        comparison, report_dir = compare_report_directories(
            baseline_dir=args.baseline_dir,
            candidate_dir=args.candidate_dir,
        )
    except ValueError as error:
        # 报告不存在、JSON 非法或样本无法公平比较时，
        # argparse 以清晰错误信息结束命令。
        parser.error(str(error))

    # 读取整体通过率的比较结果。
    overall = comparison["metrics"]["overall"]

    # 输出基线和候选的整体通过数量。
    print(
        "正式比较完成："
        f"基线 {overall['baseline_correct']}/{overall['total']}，"
        f"候选 {overall['candidate_correct']}/{overall['total']}"
    )

    # absolute_delta 是小数，例如 0.10。
    # .1% 会把它显示为 10.0%。
    print(f"绝对提升：{overall['absolute_delta']:.1%}")

    # 基线正确率为 0 时，相对提升不存在。
    # 当前真实基线不是 0，但保留这个分支使代码完整。
    if overall["relative_improvement"] is None:
        print("相对提升：无法计算（基线正确率为 0）")
    else:
        print(
            "相对提升："
            f"{overall['relative_improvement']:.1%}"
        )

    # 输出正式比较报告目录。
    print(f"比较报告目录：{report_dir}")


if __name__ == "__main__":
    # 只有直接运行模块时，才执行 main()。
    main()