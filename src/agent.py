from typing import Literal, TypedDict
from src.knowledge_base import (
    FaqId,
    KNOWLEDGE_BASE_METADATA,
    RetrievalCandidate,
)
from src.rag_retriever import search_faq_entries


# 模型降级时展示给用户的友好提示，不暴露底层异常名称。
FALLBACK_USER_NOTICE = "系统当前繁忙，已使用备用方式继续处理您的问题。"

# FAQ 只有达到这个最低分数，才会进入最终回复上下文。
# 0.55 能保留当前已验证的 FAQ 问法（退款时效约为 0.5833），
# 同时为明显低相关候选保留拒绝空间。
MIN_RETRIEVAL_SCORE = 0.55


class CustomerState(TypedDict, total=False):
    query: str
    category: Literal["technical", "billing", "general"]
    analysis_source: Literal["llm", "rule", "rule_fallback"]
    analysis_error: str
    response_source: Literal["llm", "faq_fallback", "local"]
    response_error: str
    sentiment: Literal["positive", "negative", "neutral"]
    route: Literal[
        "technical_reply",
        "billing_reply",
        "general_reply",
        "human_handoff",
    ]
    faq_id: FaqId
    faq_answer: str
    retrieved_contexts: list[str]
    retrieval_score: float
    retrieval_keyword_score: float
    retrieval_text_score: float
    retrieval_method: str
    retrieval_candidates: list[RetrievalCandidate]
    knowledge_base_name: str
    knowledge_base_version: str

    # 模型响应中可选的输入 Token 数量。
    input_tokens: int

    # 模型响应中可选的输出 Token 数量。
    output_tokens: int

    # 根据环境变量价格配置计算出的估算成本。
    estimated_cost_usd: float

    response: str


def create_customer_state(query: str) -> CustomerState:
    return {"query": query}

def categorize_query(state: CustomerState) -> CustomerState:
    query = state["query"]

    billing_keywords = ("付款", "支付", "退款", "账单", "扣款")
    technical_keywords = ("登录", "密码", "报错", "无法打开", "崩溃")

    if any(keyword in query for keyword in billing_keywords):
        category = "billing"
    elif any(keyword in query for keyword in technical_keywords):
        category = "technical"
    else:
        category = "general"

    return {
        # 保存规则分类结果。
        "category": category,

        # 明确说明分类来自本地规则，
        # 不是大模型分析。
        "analysis_source": "rule",
    }


def analyze_sentiment(state: CustomerState) -> CustomerState:
    query = state["query"]

    negative_keywords = ("不满意", "太差", "投诉", "生气", "失望", "垃圾", "一直没", "根本")
    positive_keywords = ("谢谢", "满意", "很好", "不错", "赞")

    if any(keyword in query for keyword in negative_keywords):
        sentiment = "negative"
    elif any(keyword in query for keyword in positive_keywords):
        sentiment = "positive"
    else:
        sentiment = "neutral"

    return {
        # 保存规则情绪结果。
        "sentiment": sentiment,

        # 规则分类和规则情绪共同构成规则分析阶段。
        # 这里继续使用 rule 作为来源标记。
        "analysis_source": "rule",
    }

def choose_route(state: CustomerState) -> CustomerState:
    if state["sentiment"] == "negative":
        route = "human_handoff"
    elif state["category"] == "technical":
        route = "technical_reply"
    elif state["category"] == "billing":
        route = "billing_reply"
    else:
        route = "general_reply"

    return {"route": route}


def retrieve_faq_answer(state: CustomerState) -> CustomerState:
    # 使用用户原始问题检索本地 FAQ 知识库。
    # 检索器会返回 Top-K 候选；当前受控回复只采用第一名正文，
    # 其余候选以摘要形式留下，供评估、调试和后续网页展示使用。
    matches = search_faq_entries(
        state["query"],
        top_k=3,
    )

    # 未命中知识时，不更新状态。
    if not matches:
        return {}

    # 取出排名第一的候选及其检索分数。
    best_match = matches[0]
    entry = best_match["entry"]
    retrieval_score = best_match["score"]

    # 分数低于最低阈值时，不把候选伪装成可信 FAQ。
    # 返回空更新后，后续回复节点会使用本地通用模板。
    if retrieval_score < MIN_RETRIEVAL_SCORE:
        return {}

    # 将 Top-K 候选转成紧凑、可序列化的检索证据。
    # 正文只保存在 retrieved_contexts 中，避免状态和报告重复膨胀。
    retrieval_candidates: list[RetrievalCandidate] = [
        {
            "rank": index,
            "faq_id": match["entry"]["faq_id"],
            "chunk_id": match["entry"]["chunk_id"],
            "title": match["entry"]["title"],
            "source": match["entry"]["source"],
            "version": match["entry"]["version"],
            "score": match["score"],
            "keyword_score": match["keyword_score"],
            "text_score": match["text_score"],
        }
        for index, match in enumerate(matches, start=1)
    ]

    # 命中知识时，同时保存稳定 ID、兼容旧流程的答案文本，
    # 以及实际检索到的原始上下文。
    return {
        "faq_id": entry["faq_id"],
        "faq_answer": entry["answer"],
        "retrieved_contexts": [entry["content"]],
        "retrieval_score": retrieval_score,
        "retrieval_keyword_score": best_match["keyword_score"],
        "retrieval_text_score": best_match["text_score"],
        "retrieval_method": best_match["retrieval_method"],
        "retrieval_candidates": retrieval_candidates,
        "knowledge_base_name": KNOWLEDGE_BASE_METADATA["name"],
        "knowledge_base_version": KNOWLEDGE_BASE_METADATA["version"],
    }


# def generate_response(state: CustomerState) -> CustomerState:
#     # 读取路由节点已经决定好的处理路线。
#     route = state["route"]
#
#     # 将每条路线映射到对应的客服回复模板。
#     responses = {
#         "technical_reply": "抱歉给您带来不便。请尝试重新登录或重启应用。",
#         "billing_reply": "您的账单问题已收到，请提供订单号以便进一步核实。",
#         "general_reply": "客服工作时间为每日 9:00 至 18:00。",
#         "human_handoff": "您的问题已转交人工客服，请稍候。",
#     }
#
#     # 根据当前路线取出回复模板，并作为状态更新返回。
#     return {"response": responses[route]}

def generate_response(state: CustomerState) -> CustomerState:
    # 读取路由节点已经决定好的处理路线。
    route = state["route"]

    # 先计算业务层面的基础回复。
    if route == "human_handoff":
        base_response = "您的问题已转交人工客服，请稍候。"
    else:
        # faq_answer 是可选字段；未命中知识库时不会存在。
        faq_answer = state.get("faq_answer")

        # 自动回复路线中，优先使用知识库检索到的具体答案。
        if faq_answer is not None:
            base_response = faq_answer
        else:
            # 未命中 FAQ 时，回退到原来的通用路线模板。
            fallback_responses = {
                "technical_reply": "抱歉给您带来不便。请尝试重新登录或重启应用。",
                "billing_reply": "您的账单问题已收到，请提供订单号以便进一步核实。",
                "general_reply": "客服工作时间为每日 9:00 至 18:00。",
            }

            # 根据自动回复路线取出兜底模板。
            base_response = fallback_responses[route]

    # 只在发生模型降级时增加用户可理解的提示。
    if state.get("analysis_source") == "rule_fallback":
        return {
            # 保存降级提示和本地生成的基础回复。
            "response": f"{FALLBACK_USER_NOTICE}\n{base_response}",

            # 即使发生规则降级，最终回复仍然来自本地逻辑。
            "response_source": "local",
        }

    return {
        # 保存最终给用户看的客服回复。
        "response": base_response,

        # 规则版 Agent 的回复来自本地逻辑：
        # 可能是 FAQ 原文，也可能是路线模板。
        "response_source": "local",
    }


def run_customer_service_agent(query: str) -> CustomerState:
    # 创建只包含用户问题的初始状态。
    state = create_customer_state(query)

    # 依次执行每个节点，并将节点返回的局部更新合并到总状态。
    category_update = categorize_query(state)
    state.update(category_update)

    sentiment_update = analyze_sentiment(state)
    state.update(sentiment_update)

    route_update = choose_route(state)
    state.update(route_update)

    response_update = generate_response(state)
    state.update(response_update)

    # 返回经过完整工作流处理后的状态。
    return state


if __name__ == "__main__":
    # 模拟用户发送给客服 Agent 的问题。
    user_query = "我申请的退款一个月还没到账，你们到底管不管，太差了！"

    # 执行完整客服工作流，得到最终状态。
    final_state = run_customer_service_agent(user_query)

    # 开发时输出完整状态，便于检查每个节点的结果。
    print("最终状态：", final_state)

    # 实际面向用户时，只展示最终客服回复。
    print("客服回复：", final_state["response"])
