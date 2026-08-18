import os
from collections.abc import Mapping
from typing import NotRequired, TypedDict


class ModelUsage(TypedDict):
    # 模型输入 Token 数量。
    input_tokens: NotRequired[int]

    # 模型输出 Token 数量。
    output_tokens: NotRequired[int]

    # 输入和输出 Token 的总数量。
    total_tokens: NotRequired[int]

    # 根据可选价格配置计算出的估算成本。
    estimated_cost_usd: NotRequired[float]


class UsageText(str):
    """保留原始文本，同时在字符串对象上附加 usage 元数据。"""

    usage: ModelUsage

    def __new__(
        cls,
        value: str,
        usage: ModelUsage | None = None,
    ) -> "UsageText":
        # 先创建一个普通字符串对象，
        # 因此所有原有的 str 操作仍然可以继续使用。
        instance = str.__new__(cls, value)

        # 再把 usage 保存到字符串对象上，
        # 供 analyze_with_model() 或 generate_reply_with_model() 读取。
        instance.usage = usage or {}
        return instance


def _read_optional_rate(name: str) -> float | None:
    """读取每 1000 Token 的可选价格配置。"""

    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return None

    try:
        rate = float(raw_value)
    except ValueError:
        return None

    # 负价格没有业务意义，直接视为未配置。
    if rate < 0:
        return None

    return rate


def extract_model_usage(response: object) -> ModelUsage:
    """从 OpenAI 兼容响应中提取 usage，并计算可选成本。"""

    # 不同兼容服务可能没有 usage 字段，
    # 因此所有读取都使用 getattr 防止评估流程崩溃。
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    if isinstance(usage, Mapping):
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
    else:
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)

    result: ModelUsage = {}

    if isinstance(input_tokens, int):
        result["input_tokens"] = input_tokens

    if isinstance(output_tokens, int):
        result["output_tokens"] = output_tokens

    if (
        isinstance(input_tokens, int)
        and isinstance(output_tokens, int)
    ):
        result["total_tokens"] = input_tokens + output_tokens

        # 价格配置是可选的，因为不同模型服务商的价格不同。
        input_rate = _read_optional_rate(
            "MODEL_INPUT_COST_PER_1K"
        )
        output_rate = _read_optional_rate(
            "MODEL_OUTPUT_COST_PER_1K"
        )

        if input_rate is not None and output_rate is not None:
            result["estimated_cost_usd"] = round(
                input_tokens / 1000 * input_rate
                + output_tokens / 1000 * output_rate,
                8,
            )

    return result


def attach_usage(
    content: str,
    response: object,
) -> UsageText:
    """把响应文本和 usage 封装成兼容 str 的对象。"""

    return UsageText(
        content,
        usage=extract_model_usage(response),
    )
