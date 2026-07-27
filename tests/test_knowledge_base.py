import unittest

from src.knowledge_base import find_faq_answer


class KnowledgeBaseTests(unittest.TestCase):
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