import unittest
from unittest.mock import patch

from src.langgraph_llm_agent import run_langgraph_llm_customer_service_agent


class LangGraphLlmAgentTests(unittest.TestCase):
    @patch("src.langgraph_llm_agent.classify_with_model")
    def test_negative_technical_query_is_handed_to_human(
        self,
        mock_classify_with_model,
    ) -> None:
        # 准备一条技术问题且带有负面情绪的用户输入。
        query = "软件打开后一直崩溃，太差了！"

        # 模拟大模型已经完成分类后的返回结果。
        mock_classify_with_model.return_value = {"category": "technical"}

        # 运行完整的大模型版 LangGraph 工作流。
        result = run_langgraph_llm_customer_service_agent(query)

        # 验证大模型分类结果已合并进最终状态。
        self.assertEqual(result["category"], "technical")

        # 验证原有情绪判断和路由规则仍然正常复用。
        self.assertEqual(result["sentiment"], "negative")
        self.assertEqual(result["route"], "human_handoff")

        # 验证人工转接路线生成正确回复。
        self.assertEqual(result["response"], "您的问题已转交人工客服，请稍候。")

        # 验证分类节点将原始用户问题传给大模型分类函数一次。
        mock_classify_with_model.assert_called_once_with(query)