import unittest
from unittest.mock import patch

from src.evaluation_cases import EvaluationCase
from src.evaluation_runner import evaluate_case


class EvaluationRunnerTests(unittest.TestCase):
    @patch("src.evaluation_runner.run_langgraph_llm_customer_service_agent")
    def test_rule_fallback_with_correct_result_is_semantically_passed(
        self,
        mock_run_agent,
    ) -> None:
        # 构造一条人工预先标注好的技术负面问题。
        case: EvaluationCase = {
            "name": "fallback_technical_negative",
            "query": "软件打开后一直崩溃，太差了！",
            "expected_category": "technical",
            "expected_sentiment": "negative",
            "expected_route": "human_handoff",
            "expected_faq_in_state": False,
        }

        # 模拟完整客服 Agent 在模型分析失败后得到的最终状态。
        # 这里不请求真实 API，而是直接构造降级后的结果。
        mock_run_agent.return_value = {
            "query": case["query"],
            "category": "technical",
            "sentiment": "negative",
            "analysis_source": "rule_fallback",
            "analysis_error": "ValueError",
            "route": "human_handoff",
            "response": (
                "系统当前繁忙，已使用备用方式继续处理您的问题。\n"
                "您的问题已转交人工客服，请稍候。"
            ),
            "response_source": "local",
        }

        # 执行评估器，而不是直接执行真实客服工作流。
        result = evaluate_case(case)

        # 验证语义结果全部正确，所以这条样本整体通过。
        self.assertTrue(result["passed"])

        # 验证分类、情绪、路线和 FAQ 状态分别通过。
        self.assertTrue(result["category_ok"])
        self.assertTrue(result["sentiment_ok"])
        self.assertTrue(result["route_ok"])
        self.assertTrue(result["faq_ok"])

        # 验证评估器仍然记录了本次发生了规则降级。
        self.assertEqual(result["analysis_source"], "rule_fallback")

        # 验证最终回复确实来自本地 Python 回复逻辑。
        self.assertEqual(result["response_source"], "local")

        # 验证评估器只调用了一次客服 Agent。
        mock_run_agent.assert_called_once_with(case["query"])

    @patch("src.evaluation_runner.run_langgraph_llm_customer_service_agent")
    def test_faq_reply_fallback_is_recorded_separately(
        self,
        mock_run_agent,
    ) -> None:
        # 准备一条能够命中退款 FAQ 的中性账单问题。
        faq_answer = "退款申请审核通过后，原路退款通常在 3 至 5 个工作日到账。"

        case: EvaluationCase = {
            "name": "faq_reply_fallback",
            "query": "退款一般多久到账？",
            "expected_category": "billing",
            "expected_sentiment": "neutral",
            "expected_route": "billing_reply",
            "expected_faq_in_state": True,
        }

        # 模拟分析模型成功，但回复模型超时后的最终状态。
        # FAQ 原文仍然被保留，因此最终回复仍然可靠。
        mock_run_agent.return_value = {
            "query": case["query"],
            "category": "billing",
            "sentiment": "neutral",
            "analysis_source": "llm",
            "route": "billing_reply",
            "faq_answer": faq_answer,
            "response": faq_answer,
            "response_source": "faq_fallback",
            "response_error": "APITimeoutError",
        }

        # 执行评估器。
        result = evaluate_case(case)

        # 分类、情绪、路线和 FAQ 状态都正确。
        self.assertTrue(result["passed"])
        self.assertTrue(result["category_ok"])
        self.assertTrue(result["sentiment_ok"])
        self.assertTrue(result["route_ok"])
        self.assertTrue(result["faq_ok"])

        # 分析阶段来自大模型，说明分类分析本身没有失败。
        self.assertEqual(result["analysis_source"], "llm")

        # 回复阶段使用 FAQ 原文兜底。
        self.assertEqual(result["response_source"], "faq_fallback")

        # 验证评估器只运行了一次客服 Agent。
        mock_run_agent.assert_called_once_with(case["query"])



if __name__ == "__main__":
    # 直接运行本文件时，启动 unittest 测试。
    unittest.main()