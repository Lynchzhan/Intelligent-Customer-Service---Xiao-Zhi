from langgraph.graph import END, START, StateGraph

from src.agent import (
    CustomerState,
    analyze_sentiment,
    choose_route,
    generate_response,
    retrieve_faq_answer,
)
from src.langgraph_agent import should_retrieve_faq
from src.llm_classifier import ModelClassification, classify_with_model


def categorize_with_model(state: CustomerState) -> ModelClassification:
    # 从工作流状态中读取用户原始问题。
    query = state["query"]

    # 调用大模型分类，并返回符合分类格式的结果。
    return classify_with_model(query)


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