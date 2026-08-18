from openai import APIConnectionError, APIStatusError, APITimeoutError
from langgraph.graph import END, START, StateGraph

from src.agent import (
    CustomerState,
    analyze_sentiment,
    categorize_query,
    choose_route,
    generate_response,
    retrieve_faq_answer,
)
from src.langgraph_agent import should_retrieve_faq
from src.llm_classifier import analyze_with_model
from src.llm_responder import generate_reply_with_model


def _copy_usage_fields(
    source: dict[str, object],
    target: CustomerState,
) -> None:
    """把模型客户端提供的 usage 字段复制到工作流状态。"""

    # usage 是可选信息；兼容服务没有提供时不写入状态。
    for key in (
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
    ):
        value = source.get(key)
        if value is not None:
            target[key] = value


def analyze_query_with_model(state: CustomerState) -> CustomerState:
    # 从工作流状态中读取用户原始问题。
    query = state["query"]

    try:
        # 优先使用一次大模型请求完成分类和情绪分析。
        model_update = analyze_with_model(query)
    except (
        APITimeoutError,
        APIConnectionError,
        APIStatusError,
        ValueError,
    ) as error:
        # 模型超时、网络失败、服务端报错或分类结果不合法时，
        # 同时使用本地分类和情绪节点补齐完整分析结果。
        category_update = categorize_query(state)
        sentiment_update = analyze_sentiment(state)

        return {
            "category": category_update["category"],
            "sentiment": sentiment_update["sentiment"],
            "analysis_source": "rule_fallback",
            "analysis_error": type(error).__name__,
        }

    # 两个字段都验证成功时，记录本次综合分析来自大模型。
    result: CustomerState = {
        "category": model_update["category"],
        "sentiment": model_update["sentiment"],
        "analysis_source": "llm",
    }
    _copy_usage_fields(model_update, result)
    return result


def generate_controlled_response(state: CustomerState) -> CustomerState:
    # 先判断当前状态是否满足调用回复模型的全部条件。
    can_use_model = (
        state["route"] != "human_handoff"
        and state.get("analysis_source") == "llm"
        and state.get("faq_answer") is not None
    )

    # 人工转接、分类已降级或 FAQ 未命中时，直接使用本地回复。
    if not can_use_model:
        local_update = generate_response(state)
        return {
            "response": local_update["response"],
            "response_source": "local",
        }

    try:
        # 只有满足条件时，才让模型根据 FAQ 答案组织自然回复。
        model_update = generate_reply_with_model(
            state["query"],
            state["faq_answer"],
        )
    except (
        APITimeoutError,
        APIConnectionError,
        APIStatusError,
        ValueError,
    ) as error:
        # 回复模型失败时，保留 FAQ 原文作为可靠答案。
        fallback_update = generate_response(state)
        return {
            "response": fallback_update["response"],
            "response_source": "faq_fallback",
            "response_error": type(error).__name__,
        }

    # 模型回复成功时，记录最终回复来自受控模型。
    result: CustomerState = {
        "response": model_update["response"],
        "response_source": "llm",
    }
    _copy_usage_fields(model_update, result)
    return result


# 使用统一的客服状态结构创建大模型版工作流。
workflow = StateGraph(CustomerState)

# 注册各个工作流节点。
workflow.add_node("analyze_query", analyze_query_with_model)
workflow.add_node("choose_route", choose_route)
workflow.add_node("retrieve_faq_answer", retrieve_faq_answer)
workflow.add_node("generate_response", generate_controlled_response)

# 定义工作流的固定执行顺序。
workflow.add_edge(START, "analyze_query")
workflow.add_edge("analyze_query", "choose_route")

# 根据路由结果，选择是否需要执行 FAQ 检索。
workflow.add_conditional_edges(
    "choose_route",
    should_retrieve_faq,
    {
        "retrieve_faq_answer": "retrieve_faq_answer",
        "generate_response": "generate_response",
    },
)

# FAQ 检索完成后进入回复节点。
workflow.add_edge("retrieve_faq_answer", "generate_response")

# 回复生成后结束整个工作流。
workflow.add_edge("generate_response", END)

# 将设计好的图编译为可执行对象。
app = workflow.compile()


def run_langgraph_llm_customer_service_agent(query: str) -> CustomerState:
    # 以用户问题作为 LangGraph 的初始状态。
    initial_state: CustomerState = {"query": query}

    # 执行完整工作流，并返回最终状态。
    return app.invoke(initial_state)
