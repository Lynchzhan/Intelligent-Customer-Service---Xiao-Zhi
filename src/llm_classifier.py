import json
from typing import Literal, TypedDict

from src.model_client import create_openai_client
from src.model_config import load_model_config


# 限制问题分类只能是三种客服问题之一。
Category = Literal["technical", "billing", "general"]

# 限制情绪分析结果只能是三种情绪之一。
Sentiment = Literal["positive", "negative", "neutral"]


class ModelAnalysis(TypedDict):
    # 一次模型分析必须同时包含问题分类和情绪判断。
    category: Category
    sentiment: Sentiment


# 要求模型一次返回固定格式的问题分类和情绪分析结果。
ANALYSIS_SYSTEM_PROMPT = """
你是智能客服的问题分类和情绪分析助手。

请把用户问题只分类为以下三种之一：
- technical：软件崩溃、登录失败、功能异常等技术问题。
- billing：付款、退款、订单、发票等账单问题。
- general：客服时间、服务范围等通用咨询。

请把用户情绪只分析为以下三种之一：
- positive：用户表达感谢、满意、认可或赞扬。
- negative：用户表达不满、愤怒、失望、投诉或明显抱怨。
- neutral：用户主要在陈述问题或询问信息，没有明显正面或负面情绪。

必须且只能返回一个 JSON 对象。
输出结束后立即停止，不能重复 JSON 对象。
不要解释，不要使用 Markdown。

返回格式必须类似：
{"category": "technical", "sentiment": "negative"}
""".strip()


def request_model_analysis(query: str) -> str:
    # 读取 .env 中的模型配置。
    config = load_model_config()

    # 创建已配置 Base URL 和 API Key 的模型客户端。
    client = create_openai_client()

    # 组合系统分析规则和当前用户问题。
    messages = [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    # 一次请求同时获得问题分类和情绪判断。
    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        # 要求服务端返回一个 JSON 对象。
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=32,
    )

    # 从完整响应中取出模型生成的文本。
    content = response.choices[0].message.content

    # 模型未返回文本时，停止处理并给出明确错误。
    if content is None:
        raise ValueError("模型没有返回分析结果。")

    return content


def parse_model_analysis(content: str) -> ModelAnalysis:
    # 创建 JSON 解码器，用于逐个读取连续的 JSON 对象。
    decoder = json.JSONDecoder()
    remaining = content.strip()
    parsed_values: list[object] = []

    # 兼容服务异常重复输出相同 JSON 对象的情况。
    while remaining:
        try:
            data, end_index = decoder.raw_decode(remaining)
        except json.JSONDecodeError as error:
            raise ValueError("模型返回的分析结果不是有效 JSON。") from error

        parsed_values.append(data)
        remaining = remaining[end_index:].strip()

    # 防御性检查：模型不应返回空内容。
    if not parsed_values:
        raise ValueError("模型没有返回分析结果。")

    first_data = parsed_values[0]

    # 分类结果必须是 JSON 对象，而不是列表、字符串或数字。
    if not isinstance(first_data, dict):
        raise ValueError("模型返回的分析结果必须是 JSON 对象。")

    # 只接受完全相同的重复对象，拒绝互相矛盾的多个分析结果。
    if any(data != first_data for data in parsed_values[1:]):
        raise ValueError("模型返回了互相矛盾的多个分析结果。")

    # 读取并验证分类字段。
    category = first_data.get("category")
    if not isinstance(category, str) or category not in {
        "technical",
        "billing",
        "general",
    }:
        raise ValueError("模型返回了不支持的分类结果。")

    # 情绪字段也必须存在，并且只能属于项目支持的三种情绪。
    sentiment = first_data.get("sentiment")
    if not isinstance(sentiment, str) or sentiment not in {
        "positive",
        "negative",
        "neutral",
    }:
        raise ValueError("模型返回了不支持的情绪分析结果。")

    # 两个字段全部验证通过后，才返回完整分析结果。
    return {
        "category": category,
        "sentiment": sentiment,
    }


def analyze_with_model(query: str) -> ModelAnalysis:
    # 请求模型，得到同时包含分类和情绪的原始 JSON 文本。
    content = request_model_analysis(query)

    # 两个字段都通过本地校验后，返回完整分析结果。
    return parse_model_analysis(content)
