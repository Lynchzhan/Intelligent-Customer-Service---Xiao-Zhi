import unittest

from src.observability import format_observability


class ObservabilityTests(unittest.TestCase):
    def test_llm_sources_are_displayed(self) -> None:
        # 模拟分类和回复都由大模型成功完成的状态。
        state = {
            "query": "退款一般多久到账？",
            "analysis_source": "llm",
            "response_source": "llm",
        }

        self.assertEqual(
            format_observability(state),
            ["分析来源：llm", "回复来源：llm"],
        )

    def test_error_sources_are_displayed(self) -> None:
        # 模拟分类和回复都发生降级的状态。
        state = {
            "query": "退款一般多久到账？",
            "analysis_source": "rule_fallback",
            "analysis_error": "APITimeoutError",
            "response_source": "faq_fallback",
            "response_error": "APIConnectionError",
        }

        self.assertEqual(
            format_observability(state),
            [
                "分析来源：rule_fallback",
                "回复来源：faq_fallback",
                "分析错误：APITimeoutError",
                "回复错误：APIConnectionError",
            ],
        )

    def test_missing_optional_fields_use_local_defaults(self) -> None:
        # 规则版状态没有大模型来源和错误字段。
        state = {"query": "软件打开后一直崩溃"}

        self.assertEqual(
            format_observability(state),
            ["分析来源：local", "回复来源：local"],
        )
