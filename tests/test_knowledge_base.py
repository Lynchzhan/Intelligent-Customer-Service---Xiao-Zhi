import unittest

from src.knowledge_base import (
    find_faq_answer,
    find_faq_entry,
    search_faq_entries,
)


class KnowledgeBaseTests(unittest.TestCase):
    def test_search_faq_entries_returns_scored_top_candidate(
        self,
    ) -> None:
        # 查询同时命中退款主题和“多久”意图。
        matches = search_faq_entries(
            "退款一般多久到账？",
            top_k=1,
        )

        # 只请求一个候选时，返回列表长度应为 1。
        self.assertEqual(len(matches), 1)

        # 候选中保留完整 FAQ 条目，第一名应为退款时效知识。
        self.assertEqual(
            matches[0]["entry"]["faq_id"],
            "refund_timing",
        )

        # 当前分数由确定性关键词规则计算，必须大于严格门槛的 0.5，
        # 也不应超过满分 1.0。
        self.assertGreater(matches[0]["score"], 0.5)
        self.assertLessEqual(matches[0]["score"], 1.0)

    def test_search_faq_entries_returns_empty_for_negative_control(
        self,
    ) -> None:
        # 用户已经确认退款到账，不是在询问到账时效。
        matches = search_faq_entries(
            "退款已经到账，谢谢客服！",
        )

        # 严格主题和意图规则不满足时，不应伪造候选上下文。
        self.assertEqual(matches, [])

    def test_refund_timing_entry_returns_stable_id(self) -> None:
        # 读取完整 FAQ 条目，而不只是兼容旧接口的答案字符串。
        entry = find_faq_entry("退款一般多久到账？")

        # 命中后必须同时拥有稳定 ID 和可信答案。
        self.assertIsNotNone(entry)
        self.assertEqual(entry["faq_id"], "refund_timing")
        self.assertEqual(
            entry["answer"],
            "退款申请审核通过后，原路退款通常在 3 至 5 个工作日到账。",
        )

    def test_refund_timing_query_returns_refund_answer(self) -> None:
        # 这条问题包含退款时效 FAQ 的全部关键词。
        answer = find_faq_answer("退款一般多久到账？")

        # 验证返回了对应的退款时效答案。
        self.assertEqual(
            answer,
            "退款申请审核通过后，原路退款通常在 3 至 5 个工作日到账。",
        )

    def test_incomplete_refund_query_returns_none(self) -> None:
        # 这条问题只包含“退款”，不足以匹配当前的严格规则。
        answer = find_faq_answer("我想退款")

        # 验证未命中时明确返回 None。
        self.assertIsNone(answer)

    def test_refund_timing_query_with_two_keywords_returns_answer(self) -> None:
        # 用户没有使用“多久”，但“退款”和“到账”已足以表达时效问题。
        answer = find_faq_answer("退款什么时候到账？")

        # 期望使用退款时效 FAQ 的答案。
        self.assertEqual(
            answer,
            "退款申请审核通过后，原路退款通常在 3 至 5 个工作日到账。",
        )

    def test_already_received_refund_query_does_not_match_timing_faq(
        self,
    ) -> None:
        # 用户已经说明退款到账，不是在询问到账时间。
        answer = find_faq_answer("退款已经到账，谢谢客服！")

        # 这类问题不应该使用退款时效 FAQ。
        self.assertIsNone(answer)

    def test_password_reset_query_returns_password_answer(self) -> None:
        # 密码和重置分别命中主题关键词和意图关键词。
        answer = find_faq_answer("我忘记密码了，怎么重置？")

        # 命中后应该返回密码重置 FAQ 的完整答案。
        self.assertEqual(
            answer,
            "请在登录页选择“忘记密码”，按提示完成密码重置。",
        )
