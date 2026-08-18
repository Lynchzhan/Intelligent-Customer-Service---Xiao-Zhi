import unittest
from unittest.mock import patch

from src.agent import retrieve_faq_answer, run_customer_service_agent

from src.langgraph_agent import run_langgraph_customer_service_agent


class CustomerServiceAgentTests(unittest.TestCase):
    @patch("src.agent.search_faq_entries")
    def test_low_retrieval_score_does_not_enter_faq_context(
        self,
        mock_search_faq_entries,
    ) -> None:
        # 模拟检索器返回一个相关度很低的候选。
        # 这个候选只用于验证阈值分支，不调用真实模型。
        mock_search_faq_entries.return_value = [
            {
                "entry": {
                    "faq_id": "refund_timing",
                    "title": "退款到账时效",
                    "content": "退款申请审核通过后，原路退款通常在 3 至 5 个工作日到账。",
                    "category": "billing",
                    "source": "project_faq",
                    "version": "1.0",
                    "updated_at": "2026-08-12",
                    "required_keywords": ("退款",),
                    "intent_keywords": ("多久",),
                    "answer": "退款申请审核通过后，原路退款通常在 3 至 5 个工作日到账。",
                },
                "score": 0.3,
            }
        ]

        # 低于 MIN_RETRIEVAL_SCORE 时，检索节点应返回空更新。
        update = retrieve_faq_answer({"query": "退款问题"})

        self.assertEqual(update, {})

    def test_negative_billing_query_is_handed_to_human(self) -> None:
        # 准备一条同时包含账单信息和负面情绪的问题。
        query = "我申请的退款一个月还没到账，你们到底管不管，太差了！"

        # 执行完整客服工作流。
        result = run_customer_service_agent(query)

        # 验证分类、情绪和最终路线是否符合业务规则。
        self.assertEqual(result["category"], "billing")
        self.assertEqual(result["sentiment"], "negative")
        self.assertEqual(result["route"], "human_handoff")

    def test_neutral_general_query_receives_general_reply(self) -> None:
        # 准备一条不含账单、技术或明显情绪关键词的通用咨询。
        query = "你们的客服工作时间是什么时候？"

        # 执行完整客服工作流。
        result = run_customer_service_agent(query)

        # 验证它被识别为中性的通用咨询。
        self.assertEqual(result["category"], "general")
        self.assertEqual(result["sentiment"], "neutral")
        self.assertEqual(result["route"], "general_reply")

        # 验证自动回复内容与当前规则一致。
        self.assertEqual(result["response"], "客服工作时间为每日 9:00 至 18:00。")

        # 分类和情绪来自规则节点。
        self.assertEqual(result["analysis_source"], "rule")

        # 最终回复来自本地 FAQ 或本地回复模板。
        self.assertEqual(result["response_source"], "local")

    def test_negative_technical_query_is_handed_to_human(self) -> None:
        # 准备一条同时包含技术关键词和负面情绪的问题。
        query = "软件打开后一直崩溃，太差了！"

        # 执行完整客服工作流。
        result = run_customer_service_agent(query)

        # 问题类别仍然是技术类。
        self.assertEqual(result["category"], "technical")

        # 负面情绪被正确识别。
        self.assertEqual(result["sentiment"], "negative")

        # 负面情绪优先级高于技术自动回复，因此必须转人工。
        self.assertEqual(result["route"], "human_handoff")

        # 转人工路线应生成对应的提示语。
        self.assertEqual(result["response"], "您的问题已转交人工客服，请稍候。")

        # 分类和情绪分析来自本地规则。
        self.assertEqual(result["analysis_source"], "rule")

        # 人工转接提示也是本地生成的。
        self.assertEqual(result["response_source"], "local")

    def test_langgraph_matches_rule_based_agent_for_negative_billing_query(self) -> None:
        # 使用一条会触发人工转接的关键业务场景。
        query = "我申请的退款一个月还没到账，你们到底管不管，太差了！"

        # 分别执行手写协调版和 LangGraph 协调版。
        rule_based_result = run_customer_service_agent(query)
        langgraph_result = run_langgraph_customer_service_agent(query)

        # 两个实现应产生完全相同的最终状态。
        self.assertEqual(langgraph_result, rule_based_result)

    def test_unhappy_general_query_is_handed_to_human(self) -> None:
        # “不满意”应被视为负面情绪，即使问题不属于账单或技术类别。
        query = "我对这个处理结果很不满意。"

        # 执行已经验证过的规则版 Agent。
        result = run_customer_service_agent(query)

        # 负面情绪应触发人工转接。
        self.assertEqual(result["sentiment"], "negative")
        self.assertEqual(result["route"], "human_handoff")

    def test_langgraph_refund_timing_query_uses_faq_answer(self) -> None:
        # 这条问题会命中退款时效 FAQ。
        query = "退款一般多久到账？"

        # LangGraph 工作流应执行 FAQ 检索节点。
        result = run_langgraph_customer_service_agent(query)

        # 将预期答案保存为变量，避免在断言中重复长文本。
        expected_answer = "退款申请审核通过后，原路退款通常在 3 至 5 个工作日到账。"

        # 验证仍然属于中性的账单问题。
        self.assertEqual(result["category"], "billing")
        self.assertEqual(result["sentiment"], "neutral")
        self.assertEqual(result["route"], "billing_reply")

        # 验证检索结果和最终回复都采用 FAQ 的具体答案。
        self.assertEqual(result["faq_answer"], expected_answer)
        self.assertEqual(result["response"], expected_answer)

        # 验证状态保留了实际检索到的知识正文。
        # 这是后续 RAG 评估判断“模型基于什么上下文回复”的依据。
        self.assertEqual(
            result["retrieved_contexts"],
            [expected_answer],
        )

        # 混合检索同时保留三层可解释证据：
        # 1. keyword_score：主题/意图规则分；
        # 2. text_score：本地 TF-IDF 文本相似度；
        # 3. retrieval_score：两者融合后的最终分。
        self.assertAlmostEqual(
            result["retrieval_keyword_score"],
            0.5 + 0.5 * (1 / 6),
            places=5,
        )
        self.assertGreater(result["retrieval_text_score"], 0.0)
        self.assertGreaterEqual(
            result["retrieval_score"],
            result["retrieval_keyword_score"],
        )
        self.assertEqual(
            result["retrieval_method"],
            "keyword_tfidf_hybrid_v1",
        )

        # 检索证据还应记录知识库版本和排名候选摘要。
        self.assertEqual(
            result["knowledge_base_version"],
            "2026.08.18",
        )
        self.assertEqual(
            result["retrieval_candidates"][0]["faq_id"],
            "refund_timing",
        )



















