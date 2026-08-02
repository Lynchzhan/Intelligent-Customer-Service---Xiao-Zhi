import unittest
from unittest.mock import patch

from src.llm_classifier import analyze_with_model, parse_model_analysis


class LlmClassifierTests(unittest.TestCase):
    def test_valid_analysis_returns_dictionary(self) -> None:
        # 准备一段同时包含合法分类和情绪的模型输出。
        content = '{"category": "billing", "sentiment": "neutral"}'

        result = parse_model_analysis(content)

        self.assertEqual(
            result,
            {"category": "billing", "sentiment": "neutral"},
        )

    def test_repeated_identical_analysis_returns_one_result(self) -> None:
        # 模拟兼容服务重复输出两个完全相同的分析对象。
        item = '{"category": "billing", "sentiment": "negative"}'

        result = parse_model_analysis(item + item)

        self.assertEqual(
            result,
            {"category": "billing", "sentiment": "negative"},
        )

    def test_invalid_json_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            parse_model_analysis("这不是 JSON")

    def test_unsupported_category_raises_value_error(self) -> None:
        content = '{"category": "other", "sentiment": "neutral"}'

        with self.assertRaises(ValueError):
            parse_model_analysis(content)

    def test_unsupported_sentiment_rejects_complete_analysis(self) -> None:
        # category 合法，但 sentiment 不属于允许范围，整次分析必须失败。
        content = '{"category": "billing", "sentiment": "angry"}'

        with self.assertRaises(ValueError):
            parse_model_analysis(content)

    def test_missing_sentiment_rejects_complete_analysis(self) -> None:
        # 缺少 sentiment 时，不能只采用 category。
        content = '{"category": "billing"}'

        with self.assertRaises(ValueError):
            parse_model_analysis(content)

    def test_conflicting_analysis_objects_raise_value_error(self) -> None:
        content = (
            '{"category": "billing", "sentiment": "neutral"}'
            '{"category": "billing", "sentiment": "negative"}'
        )

        with self.assertRaises(ValueError):
            parse_model_analysis(content)

    @patch("src.llm_classifier.request_model_analysis")
    def test_analyze_with_model_returns_validated_analysis(
        self,
        mock_request,
    ) -> None:
        # 用本地模拟文本替代真实 API 请求。
        mock_request.return_value = (
            '{"category": "technical", "sentiment": "negative"}'
        )

        result = analyze_with_model("软件打开后一直崩溃，太差了！")

        self.assertEqual(
            result,
            {"category": "technical", "sentiment": "negative"},
        )
        mock_request.assert_called_once_with("软件打开后一直崩溃，太差了！")
