from collections import Counter
from typing import Literal, TypedDict

from src.knowledge_base import FaqId


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

# 样本复杂度，用于分别统计简单、边界和复杂问题的表现。
Complexity = Literal["simple", "medium", "complex"]


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

    # 预期命中的 FAQ 标识；不应该命中 FAQ 时使用 None。
    expected_faq_id: FaqId | None

    # 评估样本的难度等级。
    complexity: Complexity

    # 便于按业务、情绪、路线或边界类型筛选样本的标签。
    tags: list[str]

class EvaluationCaseDistribution(TypedDict):
    # 当前评估集中的样本总数。
    total: int

    # 按预期分类统计样本数量。
    category_counts: dict[str, int]

    # 按预期情绪统计样本数量。
    sentiment_counts: dict[str, int]

    # 按预期路线统计样本数量。
    route_counts: dict[str, int]

    # 按 simple、medium、complex 统计样本数量。
    complexity_counts: dict[str, int]

    # 按标签统计样本数量。
    tag_counts: dict[str, int]


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
        "expected_faq_id": "refund_timing",
        "complexity": "simple",
        "tags": ["billing", "faq_hit", "neutral"],
    },
    {
        "name": "payment_status_neutral",
        "query": "我已经付款了，为什么订单还是没有支付？",
        "expected_category": "billing",
        "expected_sentiment": "neutral",
        "expected_route": "billing_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "medium",
        "tags": ["billing", "faq_miss", "neutral"],
    },
    {
        "name": "technical_crash_neutral",
        "query": "软件打开后一直崩溃",
        "expected_category": "technical",
        "expected_sentiment": "neutral",
        "expected_route": "technical_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "simple",
        "tags": ["technical", "faq_miss", "neutral"],
    },
    {
        "name": "technical_crash_negative",
        "query": "软件打开后一直崩溃，太差了！",
        "expected_category": "technical",
        "expected_sentiment": "negative",
        "expected_route": "human_handoff",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "medium",
        "tags": ["technical", "negative", "human_handoff"],
    },
    {
        "name": "service_hours_neutral",
        "query": "你们的客服工作时间是什么时候？",
        "expected_category": "general",
        "expected_sentiment": "neutral",
        "expected_route": "general_reply",
        "expected_faq_in_state": True,
        "expected_faq_id": "service_hours",
        "complexity": "simple",
        "tags": ["general", "faq_hit", "neutral"],
    },
    {
        "name": "service_positive",
        "query": "你们的服务很好，谢谢！",
        "expected_category": "general",
        "expected_sentiment": "positive",
        "expected_route": "general_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "simple",
        "tags": ["general", "positive", "faq_miss"],
    },
    {
        "name": "refund_overdue_negative",
        "query": "退款一个月还没到账，太差了！",
        "expected_category": "billing",
        "expected_sentiment": "negative",
        "expected_route": "human_handoff",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "complex",
        "tags": ["billing", "negative", "human_handoff"],
    },
    {
        "name": "refund_already_received_positive",
        "query": "退款已经到账，谢谢客服！",
        "expected_category": "billing",
        "expected_sentiment": "positive",
        "expected_route": "billing_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "medium",
        "tags": ["billing", "positive", "faq_negative_control"],
    },
    {
        "name": "password_reset_neutral",
        "query": "我忘记密码了，怎么重置？",
        "expected_category": "technical",
        "expected_sentiment": "neutral",
        "expected_route": "technical_reply",
        "expected_faq_in_state": True,
        "expected_faq_id": "password_reset",
        "complexity": "simple",
        "tags": ["technical", "faq_hit", "neutral"],
    },
    {
        "name": "refund_workdays_neutral",
        "query": "退款审核通过后一般几个工作日到账？",
        "expected_category": "billing",
        "expected_sentiment": "neutral",
        "expected_route": "billing_reply",
        "expected_faq_in_state": True,
        "expected_faq_id": "refund_timing",
        "complexity": "simple",
        "tags": ["billing", "faq_hit", "neutral", "paraphrase"],
    },
    {
        "name": "refund_bank_card_neutral",
        "query": "退款什么时候能退回银行卡？",
        "expected_category": "billing",
        "expected_sentiment": "neutral",
        "expected_route": "billing_reply",
        "expected_faq_in_state": True,
        "expected_faq_id": "refund_timing",
        "complexity": "medium",
        "tags": ["billing", "faq_hit", "neutral", "paraphrase"],
    },
    {
        "name": "payment_failed_neutral",
        "query": "订单扣款失败了，应该怎么处理？",
        "expected_category": "billing",
        "expected_sentiment": "neutral",
        "expected_route": "billing_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "medium",
        "tags": ["billing", "faq_miss", "neutral"],
    },
    {
        "name": "invoice_request_neutral",
        "query": "我可以申请电子发票吗？",
        "expected_category": "billing",
        "expected_sentiment": "neutral",
        "expected_route": "billing_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "simple",
        "tags": ["billing", "faq_miss", "neutral"],
    },
    {
        "name": "login_failed_neutral",
        "query": "我一直登录失败，应该怎么办？",
        "expected_category": "technical",
        "expected_sentiment": "neutral",
        "expected_route": "technical_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "simple",
        "tags": ["technical", "faq_miss", "neutral"],
    },
    {
        "name": "password_recovery_neutral",
        "query": "密码忘记了，要怎么找回？",
        "expected_category": "technical",
        "expected_sentiment": "neutral",
        "expected_route": "technical_reply",
        "expected_faq_in_state": True,
        "expected_faq_id": "password_reset",
        "complexity": "simple",
        "tags": ["technical", "faq_hit", "neutral", "paraphrase"],
    },
    {
        "name": "upload_error_neutral",
        "query": "上传文件时一直提示报错，怎么办？",
        "expected_category": "technical",
        "expected_sentiment": "neutral",
        "expected_route": "technical_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "medium",
        "tags": ["technical", "faq_miss", "neutral"],
    },
    {
        "name": "login_failed_negative",
        "query": "登录失败这么久了，真的太失望了！",
        "expected_category": "technical",
        "expected_sentiment": "negative",
        "expected_route": "human_handoff",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "medium",
        "tags": ["technical", "negative", "human_handoff"],
    },
    {
        "name": "service_hours_variant_neutral",
        "query": "人工客服几点开始上班？",
        "expected_category": "general",
        "expected_sentiment": "neutral",
        "expected_route": "general_reply",
        "expected_faq_in_state": True,
        "expected_faq_id": "service_hours",
        "complexity": "simple",
        "tags": ["general", "faq_hit", "neutral", "paraphrase"],
    },
    {
        "name": "service_scope_neutral",
        "query": "你们主要提供哪些服务？",
        "expected_category": "general",
        "expected_sentiment": "neutral",
        "expected_route": "general_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "simple",
        "tags": ["general", "faq_miss", "neutral"],
    },
    {
        "name": "service_professional_positive",
        "query": "客服回答得很专业，谢谢你们！",
        "expected_category": "general",
        "expected_sentiment": "positive",
        "expected_route": "general_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "simple",
        "tags": ["general", "positive", "faq_miss"],
    },
    {
        "name": "service_attitude_negative",
        "query": "客服态度太差了，我真的很生气！",
        "expected_category": "general",
        "expected_sentiment": "negative",
        "expected_route": "human_handoff",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "medium",
        "tags": ["general", "negative", "human_handoff"],
    },
    {
        "name": "refund_status_neutral",
        "query": "退款申请提交后，怎么查看处理进度？",
        "expected_category": "billing",
        "expected_sentiment": "neutral",
        "expected_route": "billing_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "medium",
        "tags": ["billing", "faq_miss", "neutral", "boundary"],
    },
    {
        "name": "payment_success_positive",
        "query": "支付成功了，谢谢提醒。",
        "expected_category": "billing",
        "expected_sentiment": "positive",
        "expected_route": "billing_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "simple",
        "tags": ["billing", "positive", "faq_miss"],
    },
    {
        "name": "duplicate_charge_negative",
        "query": "订单被重复扣款了，太差了！",
        "expected_category": "billing",
        "expected_sentiment": "negative",
        "expected_route": "human_handoff",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "complex",
        "tags": ["billing", "negative", "human_handoff", "boundary"],
    },
    {
        "name": "application_crash_variant_neutral",
        "query": "应用一打开就闪退，怎么排查？",
        "expected_category": "technical",
        "expected_sentiment": "neutral",
        "expected_route": "technical_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "medium",
        "tags": ["technical", "faq_miss", "neutral", "paraphrase"],
    },
    {
        "name": "network_login_error_neutral",
        "query": "登录时显示网络错误，应该如何处理？",
        "expected_category": "technical",
        "expected_sentiment": "neutral",
        "expected_route": "technical_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "medium",
        "tags": ["technical", "faq_miss", "neutral", "boundary"],
    },
    {
        "name": "password_change_neutral",
        "query": "我想修改登录密码。",
        "expected_category": "technical",
        "expected_sentiment": "neutral",
        "expected_route": "technical_reply",
        "expected_faq_in_state": True,
        "expected_faq_id": "password_reset",
        "complexity": "simple",
        "tags": ["technical", "faq_hit", "neutral", "paraphrase"],
    },
    {
        "name": "contact_customer_service_neutral",
        "query": "怎么联系你们的客服？",
        "expected_category": "general",
        "expected_sentiment": "neutral",
        "expected_route": "general_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "medium",
        "tags": ["general", "faq_miss", "neutral", "boundary"],
    },
    {
        "name": "service_quality_positive",
        "query": "这次服务很不错，感谢客服。",
        "expected_category": "general",
        "expected_sentiment": "positive",
        "expected_route": "general_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "simple",
        "tags": ["general", "positive", "faq_miss"],
    },
    {
        "name": "refund_delay_negative",
        "query": "退款问题拖了很久，太差了！",
        "expected_category": "billing",
        "expected_sentiment": "negative",
        "expected_route": "human_handoff",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "complex",
        "tags": ["billing", "negative", "human_handoff", "boundary"],
    },
    {
        "name": "payment_refund_mixed_neutral",
        "query": "付款成功后订单状态一直没更新，我还需要申请退款。",
        "expected_category": "billing",
        "expected_sentiment": "neutral",
        "expected_route": "billing_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "complex",
        "tags": ["billing", "neutral", "faq_miss", "mixed_intent"],
    },
    {
        "name": "password_login_failure_negative",
        "query": "我重置密码后还是登录不上，真的很失望！",
        "expected_category": "technical",
        "expected_sentiment": "negative",
        "expected_route": "human_handoff",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "complex",
        "tags": ["technical", "negative", "human_handoff", "mixed_intent"],
    },
    {
        "name": "login_success_positive",
        "query": "这次终于登录成功了，谢谢客服！",
        "expected_category": "technical",
        "expected_sentiment": "positive",
        "expected_route": "technical_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "medium",
        "tags": ["technical", "positive", "faq_miss"],
    },
    {
        "name": "password_reset_colloquial",
        "query": "密码忘了，咋重置？",
        "expected_category": "technical",
        "expected_sentiment": "neutral",
        "expected_route": "technical_reply",
        "expected_faq_in_state": True,
        "expected_faq_id": "password_reset",
        "complexity": "medium",
        "tags": ["technical", "faq_hit", "neutral", "colloquial"],
    },
    {
        "name": "service_hours_repetition",
        "query": "客服几点上班？我想再问一次客服几点上班？",
        "expected_category": "general",
        "expected_sentiment": "neutral",
        "expected_route": "general_reply",
        "expected_faq_in_state": True,
        "expected_faq_id": "service_hours",
        "complexity": "medium",
        "tags": ["general", "faq_hit", "neutral", "repetition"],
    },
    {
        "name": "service_scope_positive",
        "query": "你们的服务范围很全面，谢谢客服！",
        "expected_category": "general",
        "expected_sentiment": "positive",
        "expected_route": "general_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "simple",
        "tags": ["general", "positive", "faq_miss"],
    },
    {
        "name": "service_hours_negative",
        "query": "客服工作时间太短了，真的很失望！",
        "expected_category": "general",
        "expected_sentiment": "negative",
        "expected_route": "human_handoff",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "medium",
        "tags": ["general", "negative", "human_handoff", "boundary"],
    },
    {
        "name": "refund_overdue_neutral_boundary",
        "query": "退款超过十天还没到账，应该怎么查询进度？",
        "expected_category": "billing",
        "expected_sentiment": "neutral",
        "expected_route": "billing_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "complex",
        "tags": ["billing", "neutral", "faq_miss", "boundary"],
    },
    {
        "name": "password_change_positive",
        "query": "按照客服建议，我已经成功修改密码了，谢谢！",
        "expected_category": "technical",
        "expected_sentiment": "positive",
        "expected_route": "technical_reply",
        "expected_faq_in_state": True,
        "expected_faq_id": "password_reset",
        "complexity": "medium",
        "tags": ["technical", "positive", "faq_hit"],
    },
    {
        "name": "order_status_negative_mixed",
        "query": "付款成功但订单状态一直不对，太失望了！",
        "expected_category": "billing",
        "expected_sentiment": "negative",
        "expected_route": "human_handoff",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "complex",
        "tags": ["billing", "negative", "human_handoff", "mixed_intent"],
    },
    {
        "name": "service_contact_scope_mixed",
        "query": "我想了解你们提供哪些服务，也想知道怎么联系人工客服。",
        "expected_category": "general",
        "expected_sentiment": "neutral",
        "expected_route": "general_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "complex",
        "tags": ["general", "neutral", "faq_miss", "mixed_intent"],
    },
    {
        "name": "service_handling_fast_positive",
        "query": "这次客服处理得很快，服务让我很满意。",
        "expected_category": "general",
        "expected_sentiment": "positive",
        "expected_route": "general_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "simple",
        "tags": ["general", "positive", "faq_miss"],
    },
    {
        "name": "refund_amount_negative",
        "query": "退款已经到账了，但金额不对，太失望了！",
        "expected_category": "billing",
        "expected_sentiment": "negative",
        "expected_route": "human_handoff",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "complex",
        "tags": ["billing", "negative", "human_handoff", "boundary"],
    },
    {
        "name": "invoice_received_positive",
        "query": "电子发票已经收到了，谢谢客服！",
        "expected_category": "billing",
        "expected_sentiment": "positive",
        "expected_route": "billing_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "simple",
        "tags": ["billing", "positive", "faq_miss"],
    },
    {
        "name": "upload_error_negative",
        "query": "上传文件还是报错，真的太生气了！",
        "expected_category": "technical",
        "expected_sentiment": "negative",
        "expected_route": "human_handoff",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "medium",
        "tags": ["technical", "negative", "human_handoff"],
    },
    {
        "name": "refund_progress_neutral_complex",
        "query": "我申请退款后一直没有进度通知，能帮我查一下吗？",
        "expected_category": "billing",
        "expected_sentiment": "neutral",
        "expected_route": "billing_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "complex",
        "tags": ["billing", "neutral", "faq_miss", "boundary"],
    },
    {
        "name": "application_crash_login_mixed",
        "query": "软件打开就崩溃，重装后还是无法登录。",
        "expected_category": "technical",
        "expected_sentiment": "neutral",
        "expected_route": "technical_reply",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "complex",
        "tags": ["technical", "neutral", "faq_miss", "mixed_intent"],
    },
    {
        "name": "service_hours_scope_mixed",
        "query": "你们提供哪些服务，人工客服什么时候在线？",
        "expected_category": "general",
        "expected_sentiment": "neutral",
        "expected_route": "general_reply",
        "expected_faq_in_state": True,
        "expected_faq_id": "service_hours",
        "complexity": "complex",
        "tags": ["general", "neutral", "faq_hit", "mixed_intent"],
    },
    {
        "name": "invoice_missing_negative",
        "query": "电子发票一直没收到，太失望了！",
        "expected_category": "billing",
        "expected_sentiment": "negative",
        "expected_route": "human_handoff",
        "expected_faq_in_state": False,
        "expected_faq_id": None,
        "complexity": "medium",
        "tags": ["billing", "negative", "human_handoff"],
    },
    {
        "name": "service_hours_positive_faq",
        "query": "客服工作时间很清楚，谢谢！",
        "expected_category": "general",
        "expected_sentiment": "positive",
        "expected_route": "general_reply",
        "expected_faq_in_state": True,
        "expected_faq_id": "service_hours",
        "complexity": "medium",
        "tags": ["general", "positive", "faq_hit"],
    },
]


def validate_evaluation_cases(cases: list[EvaluationCase]) -> None:
    # 评估运行前先检查样本结构，避免错误数据触发真实模型请求。
    names = [case["name"] for case in cases]

    # 样本名称用于定位失败项，因此必须非空且不能重复。
    if any(not name.strip() for name in names):
        raise ValueError("评估样本名称不能为空。")
    if len(names) != len(set(names)):
        raise ValueError("评估样本名称不能重复。")

    for case in cases:
        # 每条样本至少要有一个标签，后续才能按标签分组统计。
        if not case["tags"]:
            raise ValueError(f"评估样本缺少标签：{case['name']}")
        if len(case["tags"]) != len(set(case["tags"])):
            raise ValueError(f"评估样本标签不能重复：{case['name']}")

        # complexity 必须使用固定的三档难度名称。
        if case["complexity"] not in {"simple", "medium", "complex"}:
            raise ValueError(f"评估样本难度无效：{case['name']}")

        # FAQ 命中状态和 FAQ ID 必须保持一致。
        has_expected_faq = case["expected_faq_id"] is not None
        if has_expected_faq != case["expected_faq_in_state"]:
            raise ValueError(f"FAQ 预期字段矛盾：{case['name']}")


def build_case_distribution(
    cases: list[EvaluationCase],
) -> EvaluationCaseDistribution:
    # 统计之前先检查评估数据是否合法。
    # 如果样本名称重复、标签为空或 FAQ 元数据矛盾，
    # 就不应该继续生成分布报告。
    validate_evaluation_cases(cases)

    # 统计每条样本的业务分类。
    category_counts = Counter(
        case["expected_category"]
        for case in cases
    )

    # 统计每条样本的情绪。
    sentiment_counts = Counter(
        case["expected_sentiment"]
        for case in cases
    )

    # 统计每条样本预期经过的处理路线。
    route_counts = Counter(
        case["expected_route"]
        for case in cases
    )

    # 统计每条样本所属的复杂度。
    complexity_counts = Counter(
        case["complexity"]
        for case in cases
    )

    # 一条样本可以拥有多个标签。
    # 因此需要先遍历样本，再遍历该样本的所有标签。
    tag_counts = Counter(
        tag
        for case in cases
        for tag in case["tags"]
    )

    # Counter 是字典的子类。
    # 转成普通 dict 后，更适合写入 JSON 文件。
    return {
        "total": len(cases),
        "category_counts": dict(category_counts),
        "sentiment_counts": dict(sentiment_counts),
        "route_counts": dict(route_counts),
        "complexity_counts": dict(complexity_counts),
        "tag_counts": dict(tag_counts),
    }
