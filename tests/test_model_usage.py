import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.model_usage import (
    UsageText,
    attach_usage,
    extract_model_usage,
)


class ModelUsageTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "MODEL_INPUT_COST_PER_1K": "0.01",
            "MODEL_OUTPUT_COST_PER_1K": "0.02",
        },
        clear=False,
    )
    def test_extract_usage_and_estimate_cost(self) -> None:
        # 用一个与 OpenAI 兼容响应结构相似的本地对象，
        # 不发起真实网络请求。
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
            )
        )

        usage = extract_model_usage(response)

        self.assertEqual(usage["input_tokens"], 100)
        self.assertEqual(usage["output_tokens"], 20)
        self.assertEqual(usage["total_tokens"], 120)
        self.assertAlmostEqual(
            usage["estimated_cost_usd"],
            0.0014,
        )

    def test_attach_usage_preserves_string_interface(self) -> None:
        # 没有 usage 时，仍然返回一个可以当作普通字符串使用的对象。
        response = SimpleNamespace(usage=None)
        content = attach_usage(
            '{"category": "billing"}',
            response,
        )

        self.assertIsInstance(content, str)
        self.assertIsInstance(content, UsageText)
        self.assertEqual(str(content), '{"category": "billing"}')
        self.assertEqual(content.usage, {})


if __name__ == "__main__":
    unittest.main()
