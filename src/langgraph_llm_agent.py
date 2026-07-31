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
from src.llm_classifier import classify_with_model


def categorize_with_model(state: CustomerState) -> CustomerState:
    # 从工作流状态中读取用户原始问题。
    query = state["query"]

    try:
        # 优先使用大模型完成分类。
        model_update = classify_with_model(query)
    except (
        APITimeoutError,
        APIConnectionError,
        APIStatusError,
        ValueError,
    ) as error:
        # 模型超时、网络失败、服务端报错或分类结果不合法时，
        # 使用已有规则分类函数继续处理用户问题。
        fallback_update = categorize_query(state)

        return {
            "category": fallback_update["category"],
            "classification_source": "rule_fallback",
            "classification_error": type(error).__name__,
        }

    # 模型分类成功时，记录结果来自大模型。
    return {
        "category": model_update["category"],
        "classification_source": "llm",
    }


# 使用统一的客服状态结构创建大模型版工作流。
workflow = StateGraph(CustomerState)

# 注册各个工作流节点。
workflow.add_node("categorize", categorize_with_model)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("choose_route", choose_route)
workflow.add_node("retrieve_faq_answer", retrieve_faq_answer)
workflow.add_node("generate_response", generate_response)

# 定义工作流的固定执行顺序。
workflow.add_edge(START, "categorize")
workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_edge("analyze_sentiment", "choose_route")

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