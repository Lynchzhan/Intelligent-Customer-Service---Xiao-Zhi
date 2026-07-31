import unittest
from unittest.mock import patch

from httpx import Request
from openai import APITimeoutError

from src.langgraph_llm_agent import run_langgraph_llm_customer_service_agent


class LangGraphLlmAgentTests(unittest.TestCase):
    @patch("src.langgraph_llm_agent.classify_with_model")
    def test_negative_technical_query_is_handed_to_human(
        self,
        mock_classify_with_model,
    ) -> None:
        # 准备一条技术问题且带有负面情绪的用户输入。
        query = "软件打开后一直崩溃，太差了！"

        # 模拟大模型分类成功。
        mock_classify_with_model.return_value = {"category": "technical"}

        # 运行完整的大模型版 LangGraph 工作流。
        result = run_langgraph_llm_customer_service_agent(query)

        # 验证分类结果来自大模型。
        self.assertEqual(result["category"], "technical")
        self.assertEqual(result["classification_source"], "llm")

        # 验证真实情绪判断和路由节点仍然正常执行。
        self.assertEqual(result["sentiment"], "negative")
        self.assertEqual(result["route"], "human_handoff")

        # 验证负面问题最终转交人工客服。
        self.assertEqual(
            result["response"],
            "您的问题已转交人工客服，请稍候。",
        )

        # 验证原始用户问题只传给大模型分类函数一次。
        mock_classify_with_model.assert_called_once_with(query)

    @patch("src.langgraph_llm_agent.classify_with_model")
    def test_timeout_falls_back_to_rule_based_classification(
        self,
        mock_classify_with_model,
    ) -> None:
        # 准备一条规则分类器能够识别的技术问题。
        query = "软件打开后一直崩溃"

        # 模拟模型 API 请求超时。
        mock_classify_with_model.side_effect = APITimeoutError(
            request=Request(
                "POST",
                "https://example.test/v1/chat/completions",
            )
        )

        # 运行工作流；模型超时后不应中断整个客服流程。
        result = run_langgraph_llm_customer_service_agent(query)

        # 验证规则分类器成功接管分类任务。
        self.assertEqual(result["category"], "technical")
        self.assertEqual(
            result["classification_source"],
            "rule_fallback",
        )

        # 验证状态中记录了触发降级的错误类型。
        self.assertEqual(
            result["classification_error"],
            "APITimeoutError",
        )

        # 验证降级后，后续节点仍然正常执行。
        self.assertEqual(result["sentiment"], "neutral")
        self.assertEqual(result["route"], "technical_reply")
        self.assertEqual(
            result["response"],
            "抱歉给您带来不便。请尝试重新登录或重启应用。",
        )

        # 验证系统确实先尝试过一次大模型分类。
        mock_classify_with_model.assert_called_once_with(query)