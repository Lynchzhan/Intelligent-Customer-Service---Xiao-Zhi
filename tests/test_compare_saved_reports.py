import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from src.compare_saved_reports import compare_report_directories
from src.evaluation_runner import (
    EvaluationResult,
    save_evaluation_report,
)


class CompareSavedReportsTests(unittest.TestCase):
    def test_compare_report_directories_reads_reports_and_writes_comparison(
        self,
    ) -> None:
        # 构造一条完全本地的评估结果。
        # 它模拟“退款时效”问题在基线和候选中都通过。
        #
        # 注意：这里只是普通 Python 字典，
        # 不调用 LangGraph、不调用 Agent、更不调用模型 API。
        result: EvaluationResult = {
            "name": "compare_directory_sample",
            "query": "退款一般多久到账？",
            "passed": True,
            "category_ok": True,
            "sentiment_ok": True,
            "route_ok": True,
            "faq_ok": True,
            "actual_category": "billing",
            "expected_category": "billing",
            "actual_sentiment": "neutral",
            "expected_sentiment": "neutral",
            "actual_route": "billing_reply",
            "expected_route": "billing_reply",
            "actual_faq_in_state": True,
            "expected_faq_in_state": True,
            "actual_faq_id": "refund_timing",
            "expected_faq_id": "refund_timing",
            "faq_id_ok": True,
            "complexity": "simple",
            "tags": ["billing", "faq_hit", "neutral"],
            "analysis_source": "local",
            "response_source": "local",
        }

        # 在 tests 目录创建唯一的临时根目录。
        # uuid4() 保证每次测试的目录名不同，避免相互冲突。
        temp_dir = (
            Path(__file__).resolve().parent
            / f".tmp_compare_directories_{uuid4().hex}"
        )

        try:
            # 创建一份模拟的基线评估报告。
            baseline_dir = save_evaluation_report(
                [result],
                output_root=temp_dir / "baselines",
                run_id="baseline-test",
            )

            # 创建一份模拟的候选评估报告。
            #
            # 当前两份结果完全相同，
            # 所以正式比较后的整体通过率都应为 100%。
            candidate_dir = save_evaluation_report(
                [result],
                output_root=temp_dir / "candidates",
                run_id="candidate-test",
            )

            # 替换真实 Agent 调用对象。
            # 本测试没有评估新问题，只读取已有 JSON，
            # 所以这个 mock 理论上不应被调用。
            with patch(
                "src.evaluation_runner.run_langgraph_llm_customer_service_agent"
            ) as mock_run_agent:
                # 调用本轮新增的可复用函数。
                comparison, report_dir = compare_report_directories(
                    baseline_dir=baseline_dir,
                    candidate_dir=candidate_dir,
                    output_root=temp_dir / "comparisons",
                )

            # 验证整个过程中没有执行真实 Agent。
            # 因此也不会产生模型 API 请求或费用。
            mock_run_agent.assert_not_called()

            # 读取整体比较指标。
            overall = comparison["metrics"]["overall"]

            # 基线和候选各有 1 条样本，且都通过。
            self.assertEqual(overall["baseline_correct"], 1)
            self.assertEqual(overall["candidate_correct"], 1)
            self.assertEqual(overall["total"], 1)

            # 1 / 1 = 1.0，即 100%。
            self.assertEqual(overall["baseline_rate"], 1.0)
            self.assertEqual(overall["candidate_rate"], 1.0)

            # 两个方案表现相同，所以没有绝对提升。
            self.assertEqual(overall["absolute_delta"], 0.0)

            # 基线并非 0，因此相对提升可以计算，结果也是 0。
            self.assertEqual(overall["relative_improvement"], 0.0)

            # 验证比较函数确实在指定目录中创建了报告。
            self.assertEqual(
                report_dir.parent,
                temp_dir / "comparisons",
            )

            # 正式比较报告必须包含三份 JSON 文件。
            self.assertTrue(
                (report_dir / "baseline_results.json").exists()
            )
            self.assertTrue(
                (report_dir / "candidate_results.json").exists()
            )
            self.assertTrue(
                (report_dir / "comparison.json").exists()
            )

        finally:
            # 无论断言成功或失败，都删除测试生成的临时文件。
            # 不影响你的真实 reports/ 历史报告。
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()