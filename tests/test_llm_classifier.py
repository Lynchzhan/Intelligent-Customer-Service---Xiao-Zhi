import unittest
from unittest.mock import patch

from src.llm_classifier import (
    classify_with_model,
    parse_model_classification,
)


class LlmClassifierTests(unittest.TestCase):
    def test_valid_billing_category_returns_dictionary(self) -> None:
        # 准备一段模拟模型返回的合法 JSON 文本。
        content = '{"category": "billing"}'

        # 调用本地解析函数。
        result = parse_model_classification(content)

        # 验证解析结果与预期字典完全一致。
        self.assertEqual(result, {"category": "billing"})

    def test_repeated_identical_json_objects_return_one_category(self) -> None:
        # 模拟兼容服务连续重复输出相同 JSON 对象。
        content = '{"category": "billing"}{"category": "billing"}'

        # 解析器应恢复出一个经过验证的分类字典。
        result = parse_model_classification(content)

        # 重复且一致的结果最终只保留一个分类。
        self.assertEqual(result, {"category": "billing"})

    def test_invalid_json_raises_value_error(self) -> None:
        # 模拟模型返回无法解析为 JSON 的普通文本。
        content = "这不是 JSON"

        # 解析器应将 JSON 语法问题转换为清晰的 ValueError。
        with self.assertRaises(ValueError):
            parse_model_classification(content)

    def test_unsupported_category_raises_value_error(self) -> None:
        # JSON 格式正确，但分类值不在允许范围内。
        content = '{"category": "other"}'

        # 业务分类校验必须拒绝该结果。
        with self.assertRaises(ValueError):
            parse_model_classification(content)

    def test_conflicting_json_objects_raise_value_error(self) -> None:
        # 模拟模型返回两个互相矛盾的分类对象。
        content = '{"category": "billing"}{"category": "technical"}'

        # 解析器不能任意选择一个，必须主动报错。
        with self.assertRaises(ValueError):
            parse_model_classification(content)

    @patch("src.llm_classifier.request_model_classification")
    def test_classify_with_model_returns_validated_category(
        self,
        mock_request,
    ) -> None:
        # 指定模拟请求函数要返回的原始 JSON 文本。
        mock_request.return_value = '{"category": "technical"}'

        # 调用协调函数；内部请求步骤会使用模拟函数。
        result = classify_with_model("软件打开后一直崩溃")

        # 验证协调函数返回经过解析和校验的分类字典。
        self.assertEqual(result, {"category": "technical"})

        # 验证用户问题被原样传给请求函数，并且只调用一次。
        mock_request.assert_called_once_with("软件打开后一直崩溃")