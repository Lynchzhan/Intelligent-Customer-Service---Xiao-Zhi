import unittest

from src.evaluation_cases import EVALUATION_CASES
from src.llm_candidate_runner import select_evaluation_cases


class LlmCandidateRunnerTests(unittest.TestCase):
    def test_select_evaluation_cases_returns_requested_prefix(
        self,
    ) -> None:
        # 选择前三条样本。
        selected_cases = select_evaluation_cases(3)

        # 确认实际返回了三条样本。
        self.assertEqual(len(selected_cases), 3)

        # 确认返回的是评估集最前面的三条，
        # 保证小样本试运行的选择过程可复现。
        self.assertEqual(
            selected_cases,
            EVALUATION_CASES[:3],
        )

    def test_select_evaluation_cases_rejects_invalid_limit(
        self,
    ) -> None:
        # 0 条样本没有评估意义，必须拒绝。
        with self.assertRaises(ValueError):
            select_evaluation_cases(0)

        # 请求超过评估集总数的样本，也必须拒绝。
        with self.assertRaises(ValueError):
            select_evaluation_cases(
                len(EVALUATION_CASES) + 1,
            )


if __name__ == "__main__":
    unittest.main()