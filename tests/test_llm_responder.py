import unittest
from unittest.mock import patch

from src.llm_responder import (
    generate_reply_with_model,
    parse_model_reply,
    request_model_reply,
)


class LlmResponderTests(unittest.TestCase):
    def test_valid_model_reply_returns_dictionary(self) -> None:
        # 准备一段符合约定格式的模拟模型输出。
        content = '{"response": "退款通常会在 3 至 5 个工作日内原路退回。"}'

        # 解析过程完全在本地执行。
        result = parse_model_reply(content)

        # 最终得到只包含 response 的 Python 字典。
        self.assertEqual(
            result,
            {"response": "退款通常会在 3 至 5 个工作日内原路退回。"},
        )

    def test_repeated_identical_replies_return_one_reply(self) -> None:
        # 模拟兼容服务重复输出两个完全相同的 JSON 对象。
        item = '{"response": "退款将在 3 至 5 个工作日内到账。"}'

        # 相同的重复结果可以安全恢复为一个回复。
        result = parse_model_reply(item + item)

        self.assertEqual(
            result,
            {"response": "退款将在 3 至 5 个工作日内到账。"},
        )

    def test_conflicting_replies_raise_value_error(self) -> None:
        # 两个回复内容不同，程序不能擅自选择其中一个。
        content = (
            '{"response": "退款将在 3 个工作日内到账。"}'
            '{"response": "退款将在 5 个工作日内到账。"}'
        )

        with self.assertRaises(ValueError):
            parse_model_reply(content)

    def test_empty_response_raises_value_error(self) -> None:
        # JSON 格式正确，但 response 内容为空。
        content = '{"response": "   "}'

        with self.assertRaises(ValueError):
            parse_model_reply(content)

    def test_empty_faq_answer_stops_before_model_request(self) -> None:
        # 没有可信知识时，回复模型不应该被调用。
        with self.assertRaises(ValueError):
            request_model_reply("退款什么时候到账？", "   ")

    @patch("src.llm_responder.request_model_reply")
    def test_generate_reply_with_model_returns_validated_reply(
        self,
        mock_request,
    ) -> None:
        # 用本地模拟文本替代真实 API 请求。
        mock_request.return_value = '{"response": "退款通常会原路退回。"}'

        result = generate_reply_with_model(
            "退款一般多久到账？",
            "退款申请审核通过后，原路退款通常在 3 至 5 个工作日到账。",
        )

        self.assertEqual(result, {"response": "退款通常会原路退回。"})
        mock_request.assert_called_once_with(
            "退款一般多久到账？",
            "退款申请审核通过后，原路退款通常在 3 至 5 个工作日到账。",
        )


if __name__ == "__main__":
    unittest.main()
