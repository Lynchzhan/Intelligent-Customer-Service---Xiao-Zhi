import unittest
from unittest.mock import patch

from httpx import Request
from openai import APITimeoutError

from src.langgraph_llm_agent import run_langgraph_llm_customer_service_agent


class LangGraphLlmAgentTests(unittest.TestCase):
    @patch("src.langgraph_llm_agent.generate_reply_with_model")
    @patch("src.langgraph_llm_agent.analyze_with_model")
    def test_negative_technical_query_is_handed_to_human(
        self,
        mock_analyze_with_model,
        mock_generate_reply,
    ) -> None:
        # 准备一条技术问题且带有负面情绪的用户输入。
        query = "软件打开后一直崩溃，太差了！"

        # 模拟一次模型分析同时返回分类和负面情绪。
        mock_analyze_with_model.return_value = {
            "category": "technical",
            "sentiment": "negative",
        }

        # 运行完整的大模型版 LangGraph 工作流。
        result = run_langgraph_llm_customer_service_agent(query)

        # 验证分类结果来自大模型。
        self.assertEqual(result["category"], "technical")
        self.assertEqual(result["analysis_source"], "llm")

        # 验证真实情绪判断和路由节点仍然正常执行。
        self.assertEqual(result["sentiment"], "negative")
        self.assertEqual(result["route"], "human_handoff")

        # 验证负面问题最终转交人工客服。
        self.assertEqual(
            result["response"],
            "您的问题已转交人工客服，请稍候。",
        )
        self.assertEqual(result["response_source"], "local")

        # 验证原始用户问题只传给大模型分类函数一次。
        mock_analyze_with_model.assert_called_once_with(query)

        # 人工转接路线不能调用回复模型。
        mock_generate_reply.assert_not_called()

    @patch("src.langgraph_llm_agent.generate_reply_with_model")
    @patch("src.langgraph_llm_agent.analyze_with_model")
    def test_timeout_falls_back_to_rule_based_classification(
        self,
        mock_analyze_with_model,
        mock_generate_reply,
    ) -> None:
        # 准备一条规则分类器能够识别的技术问题。
        query = "软件打开后一直崩溃"

        # 模拟模型 API 请求超时。
        mock_analyze_with_model.side_effect = APITimeoutError(
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
            result["analysis_source"],
            "rule_fallback",
        )

        # 验证状态中记录了触发降级的错误类型。
        self.assertEqual(
            result["analysis_error"],
            "APITimeoutError",
        )

        # 验证降级后，后续节点仍然正常执行。
        self.assertEqual(result["sentiment"], "neutral")
        self.assertEqual(result["route"], "technical_reply")
        self.assertEqual(
            result["response"],
            "系统当前繁忙，已使用备用方式继续处理您的问题。\n"
            "抱歉给您带来不便。请尝试重新登录或重启应用。",
        )
        self.assertEqual(result["response_source"], "local")

        # 验证系统确实先尝试过一次大模型分类。
        mock_analyze_with_model.assert_called_once_with(query)

        # 分类模型已经失败时，不连续调用同一服务生成回复。
        mock_generate_reply.assert_not_called()

    @patch("src.langgraph_llm_agent.generate_reply_with_model")
    @patch("src.langgraph_llm_agent.analyze_with_model")
    def test_faq_answer_is_rewritten_by_controlled_model(
        self,
        mock_analyze_with_model,
        mock_generate_reply,
    ) -> None:
        # 这条问题能够命中退款时效 FAQ。
        query = "退款一般多久到账？"
        faq_answer = "退款申请审核通过后，原路退款通常在 3 至 5 个工作日到账。"

        # 模拟分类模型和回复模型都成功。
        mock_analyze_with_model.return_value = {
            "category": "billing",
            "sentiment": "neutral",
        }
        mock_generate_reply.return_value = {
            "response": "退款审核通过后，通常会在 3 至 5 个工作日内原路退回。"
        }

        result = run_langgraph_llm_customer_service_agent(query)

        # FAQ 仍然保留在状态中，模型只负责组织最终表达。
        self.assertEqual(result["faq_answer"], faq_answer)
        self.assertEqual(
            result["response"],
            "退款审核通过后，通常会在 3 至 5 个工作日内原路退回。",
        )
        self.assertEqual(result["response_source"], "llm")

        mock_generate_reply.assert_called_once_with(query, faq_answer)

    @patch("src.langgraph_llm_agent.generate_reply_with_model")
    @patch("src.langgraph_llm_agent.analyze_with_model")
    def test_reply_timeout_falls_back_to_faq_answer(
        self,
        mock_analyze_with_model,
        mock_generate_reply,
    ) -> None:
        # 准备一条能够命中 FAQ 的账单问题。
        query = "退款一般多久到账？"
        faq_answer = "退款申请审核通过后，原路退款通常在 3 至 5 个工作日到账。"

        # 分类成功，但回复模型发生超时。
        mock_analyze_with_model.return_value = {
            "category": "billing",
            "sentiment": "neutral",
        }
        mock_generate_reply.side_effect = APITimeoutError(
            request=Request(
                "POST",
                "https://example.test/v1/chat/completions",
            )
        )

        result = run_langgraph_llm_customer_service_agent(query)

        # 回复模型失败后，最终回复必须回退到可信 FAQ 原文。
        self.assertEqual(result["response"], faq_answer)
        self.assertEqual(result["response_source"], "faq_fallback")
        self.assertEqual(result["response_error"], "APITimeoutError")

        mock_generate_reply.assert_called_once_with(query, faq_answer)

    @patch("src.langgraph_llm_agent.generate_reply_with_model")
    @patch("src.langgraph_llm_agent.analyze_with_model")
    def test_missing_faq_uses_local_reply_without_model_call(
        self,
        mock_analyze_with_model,
        mock_generate_reply,
    ) -> None:
        # 这条技术问题不会命中当前 FAQ 知识库。
        query = "软件打开后一直崩溃"
        mock_analyze_with_model.return_value = {
            "category": "technical",
            "sentiment": "neutral",
        }

        result = run_langgraph_llm_customer_service_agent(query)

        # 没有可信 FAQ 时，只能使用本地模板回复。
        self.assertNotIn("faq_answer", result)
        self.assertEqual(result["response_source"], "local")
        self.assertEqual(
            result["response"],
            "抱歉给您带来不便。请尝试重新登录或重启应用。",
        )

        # 不允许模型在没有知识依据时自由回答。
        mock_generate_reply.assert_not_called()

    @patch("src.langgraph_llm_agent.generate_reply_with_model")
    @patch("src.langgraph_llm_agent.analyze_with_model")
    def test_negative_billing_query_skips_faq_and_reply_model(
        self,
        mock_analyze_with_model,
        mock_generate_reply,
    ) -> None:
        # 准备一条同时包含“退款”账单信息和“太差了”负面情绪的问题。
        query = "退款一个月还没到账，太差了！"

        # 模拟一次综合大模型请求。
        # 这里不是真实请求 API，而是直接构造模型成功返回的数据。
        mock_analyze_with_model.return_value = {
            "category": "billing",
            "sentiment": "negative",
        }

        # 执行完整的大模型版客服工作流。
        result = run_langgraph_llm_customer_service_agent(query)

        # 验证业务分类结果。
        self.assertEqual(result["category"], "billing")

        # 验证负面情绪结果。
        self.assertEqual(result["sentiment"], "negative")

        # 验证综合分析确实来自大模型。
        self.assertEqual(result["analysis_source"], "llm")

        # 验证负面情绪优先级高于账单分类。
        self.assertEqual(result["route"], "human_handoff")

        # 验证最终回复使用本地人工转接模板。
        self.assertEqual(
            result["response"],
            "您的问题已转交人工客服，请稍候。",
        )

        # 验证回复来源是本地代码，而不是回复模型。
        self.assertEqual(result["response_source"], "local")

        # 人工转接路线不需要查询 FAQ。
        self.assertNotIn("faq_answer", result)

        # 综合分析模型应该只调用一次。
        mock_analyze_with_model.assert_called_once_with(query)

        # 人工转接路线不能调用回复模型。
        mock_generate_reply.assert_not_called()

    @patch("src.langgraph_llm_agent.generate_reply_with_model")
    @patch("src.langgraph_llm_agent.analyze_with_model")
    def test_invalid_model_analysis_falls_back_to_rules(
        self,
        mock_analyze_with_model,
        mock_generate_reply,
    ) -> None:
        # 准备一条包含技术关键词和负面情绪的问题。
        query = "软件打开后一直崩溃，太差了！"

        # 模拟模型返回非法分析结果。
        # ValueError 表示模型返回的数据没有通过本地校验。
        mock_analyze_with_model.side_effect = ValueError(
            "模型返回了不支持的情绪结果。"
        )

        # 执行完整的大模型版客服工作流。
        result = run_langgraph_llm_customer_service_agent(query)

        # 模型分析失败后，规则分类器应该识别为技术问题。
        self.assertEqual(result["category"], "technical")

        # 模型分析失败后，规则情绪判断应该识别为负面。
        self.assertEqual(result["sentiment"], "negative")

        # 记录本次分析是规则降级结果。
        self.assertEqual(result["analysis_source"], "rule_fallback")

        # 保存异常类型，方便开发者排查问题。
        self.assertEqual(result["analysis_error"], "ValueError")

        # 负面情绪仍然优先触发人工转接。
        self.assertEqual(result["route"], "human_handoff")

        # 降级后仍然使用本地人工转接回复。
        self.assertEqual(
            result["response"],
            "系统当前繁忙，已使用备用方式继续处理您的问题。\n"
            "您的问题已转交人工客服，请稍候。",
        )

        # 回复来源是本地，而不是回复模型。
        self.assertEqual(result["response_source"], "local")

        # 验证模型分析确实只尝试了一次。
        mock_analyze_with_model.assert_called_once_with(query)

        # 分析已经失败，不应该继续调用回复模型。
        mock_generate_reply.assert_not_called()

