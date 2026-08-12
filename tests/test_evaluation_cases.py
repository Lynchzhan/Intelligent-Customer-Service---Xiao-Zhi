import unittest
from typing import cast

from src.evaluation_cases import (
    EVALUATION_CASES,
    EvaluationCase,
    validate_evaluation_cases,
)


class EvaluationCasesTests(unittest.TestCase):
    def test_current_cases_pass_structural_validation(self) -> None:
        # 合法的当前评估集应在本地校验阶段顺利通过。
        validate_evaluation_cases(EVALUATION_CASES)

    def test_cases_include_metadata_for_future_benchmarking(self) -> None:
        # 当前评估集已经达到第一批人工核心样本的最低数量。
        self.assertGreaterEqual(len(EVALUATION_CASES), 50)

        # 三种业务分类和三种情绪都必须有覆盖。
        self.assertEqual(
            {case["expected_category"] for case in EVALUATION_CASES},
            {"technical", "billing", "general"},
        )
        self.assertEqual(
            {case["expected_sentiment"] for case in EVALUATION_CASES},
            {"positive", "negative", "neutral"},
        )

        for case in EVALUATION_CASES:
            # 每条样本都要有复杂度和至少一个业务标签。
            self.assertIn(case["complexity"], {"simple", "medium", "complex"})
            self.assertGreaterEqual(len(case["tags"]), 1)

            # FAQ 正例必须有稳定 ID，负例必须明确为 None。
            if case["expected_faq_in_state"]:
                self.assertIsNotNone(case["expected_faq_id"])
            else:
                self.assertIsNone(case["expected_faq_id"])

    def test_inconsistent_faq_metadata_is_rejected(self) -> None:
        # 复制一条 FAQ 正例，并故意制造“有 ID 但标记未命中”的矛盾。
        invalid_case = cast(EvaluationCase, EVALUATION_CASES[0].copy())
        invalid_case["expected_faq_id"] = None

        # 结构校验必须在请求模型前拒绝这条数据。
        with self.assertRaises(ValueError):
            validate_evaluation_cases([invalid_case])


if __name__ == "__main__":
    unittest.main()
