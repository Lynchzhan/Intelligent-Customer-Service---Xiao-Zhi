from typing import Literal, TypedDict


# 评估集允许出现的业务分类。
Category = Literal["technical", "billing", "general"]

# 评估集允许出现的情绪类型。
Sentiment = Literal["positive", "negative", "neutral"]

# 评估集允许出现的处理路线。
Route = Literal[
    "technical_reply",
    "billing_reply",
    "general_reply",
    "human_handoff",
]


class EvaluationCase(TypedDict):
    # 每条样本的唯一名称，方便输出和定位问题。
    name: str

    # 实际传给客服 Agent 的用户问题。
    query: str

    # 人工预先判断的正确分类。
    expected_category: Category

    # 人工预先判断的正确情绪。
    expected_sentiment: Sentiment

    # 根据分类和情绪预期得到的正确路线。
    expected_route: Route

    # 最终状态中是否应该存在 faq_answer。
    expected_faq_in_state: bool


# 多样本评估集。
# 这些不是模型输出，而是我们提前写好的参考答案。
EVALUATION_CASES: list[EvaluationCase] = [
    {
        "name": "refund_timing_neutral",
        "query": "退款一般多久到账？",
        "expected_category": "billing",
        "expected_sentiment": "neutral",
        "expected_route": "billing_reply",
        "expected_faq_in_state": True,
    },
    {
        "name": "payment_status_neutral",
        "query": "我已经付款了，为什么订单还是没有支付？",
        "expected_category": "billing",
        "expected_sentiment": "neutral",
        "expected_route": "billing_reply",
        "expected_faq_in_state": False,
    },
    {
        "name": "technical_crash_neutral",
        "query": "软件打开后一直崩溃",
        "expected_category": "technical",
        "expected_sentiment": "neutral",
        "expected_route": "technical_reply",
        "expected_faq_in_state": False,
    },
    {
        "name": "technical_crash_negative",
        "query": "软件打开后一直崩溃，太差了！",
        "expected_category": "technical",
        "expected_sentiment": "negative",
        "expected_route": "human_handoff",
        "expected_faq_in_state": False,
    },
    {
        "name": "service_hours_neutral",
        "query": "你们的客服工作时间是什么时候？",
        "expected_category": "general",
        "expected_sentiment": "neutral",
        "expected_route": "general_reply",
        "expected_faq_in_state": True,
    },
    {
        "name": "service_positive",
        "query": "你们的服务很好，谢谢！",
        "expected_category": "general",
        "expected_sentiment": "positive",
        "expected_route": "general_reply",
        "expected_faq_in_state": False,
    },
    {
        "name": "refund_overdue_negative",
        "query": "退款一个月还没到账，太差了！",
        "expected_category": "billing",
        "expected_sentiment": "negative",
        "expected_route": "human_handoff",
        "expected_faq_in_state": False,
    },
    {
        "name": "refund_already_received_positive",
        "query": "退款已经到账，谢谢客服！",
        "expected_category": "billing",
        "expected_sentiment": "positive",
        "expected_route": "billing_reply",
        "expected_faq_in_state": False,
    },
    {
        "name": "password_reset_neutral",
        "query": "我忘记密码了，怎么重置？",
        "expected_category": "technical",
        "expected_sentiment": "neutral",
        "expected_route": "technical_reply",
        "expected_faq_in_state": True,
    },
]