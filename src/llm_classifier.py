import json
from typing import Literal, TypedDict

from src.model_client import create_openai_client
from src.model_config import load_model_config


# 限制分类结果只能是三种客服问题之一。
Category = Literal["technical", "billing", "general"]


class ModelClassification(TypedDict):
    # 分类结果必须包含 category 这个字段。
    category: Category


# 要求模型只根据用户问题返回固定格式的分类结果。
CLASSIFICATION_SYSTEM_PROMPT = """
你是智能客服的问题分类助手。

请把用户问题只分类为以下三种之一：
- technical：软件崩溃、登录失败、功能异常等技术问题。
- billing：付款、退款、订单、发票等账单问题。
- general：客服时间、服务范围等通用咨询。

必须且只能返回一个 JSON 对象。
输出结束后立即停止，不能重复 JSON 对象。
不要解释，不要使用 Markdown。

返回格式必须类似：
{"category": "technical"}
""".strip()


def request_model_classification(query: str) -> str:
    # 读取 .env 中的模型配置。
    config = load_model_config()

    # 创建已配置 Base URL 和 API Key 的模型客户端。
    client = create_openai_client()

    # 组合系统分类规则和当前用户问题。
    messages = [
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    # 向模型发送分类请求，并保存完整响应。
    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        # 要求服务端返回一个 JSON 对象。
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=16,
    )

    # 从完整响应中取出模型生成的文本。
    content = response.choices[0].message.content

    # 模型未返回文本时，停止处理并给出明确错误。
    if content is None:
        raise ValueError("模型没有返回分类结果。")

    return content


def parse_model_classification(content: str) -> ModelClassification:
    # 创建 JSON 解码器，用于逐个读取连续的 JSON 对象。
    decoder = json.JSONDecoder()
    remaining = content.strip()
    parsed_values: list[object] = []

    # 兼容服务异常重复输出相同 JSON 对象的情况。
    while remaining:
        try:
            data, end_index = decoder.raw_decode(remaining)
        except json.JSONDecodeError as error:
            raise ValueError("模型返回的分类结果不是有效 JSON。") from error

        parsed_values.append(data)
        remaining = remaining[end_index:].strip()

    # 防御性检查：模型不应返回空内容。
    if not parsed_values:
        raise ValueError("模型没有返回分类结果。")

    first_data = parsed_values[0]

    # 分类结果必须是 JSON 对象，而不是列表、字符串或数字。
    if not isinstance(first_data, dict):
        raise ValueError("模型返回的分类结果必须是 JSON 对象。")

    # 只接受完全相同的重复对象，拒绝互相矛盾的多个分类。
    if any(data != first_data for data in parsed_values[1:]):
        raise ValueError("模型返回了互相矛盾的多个分类结果。")

    # 读取并验证分类字段。
    category = first_data.get("category")
    if not isinstance(category, str) or category not in {
        "technical",
        "billing",
        "general",
    }:
        raise ValueError("模型返回了不支持的分类结果。")

    # 只返回项目需要的、已经验证过的字段。
    return {"category": category}


def classify_with_model(query: str) -> ModelClassification:
    # 请求模型，得到原始 JSON 文本。
    content = request_model_classification(query)

    # 解析并验证模型返回的分类结果。
    return parse_model_classification(content)