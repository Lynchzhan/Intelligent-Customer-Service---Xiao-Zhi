from langgraph.graph import END, START, StateGraph

# 复用已经在 agent.py 中实现并测试过的状态结构和节点函数。
from src.agent import (
    CustomerState,
    analyze_sentiment,
    categorize_query,
    choose_route,
    generate_response,
    retrieve_faq_answer,
)

def should_retrieve_faq(state: CustomerState) -> str:
    # 已经转人工的问题不再执行自动 FAQ 检索。
    if state["route"] == "human_handoff":
        return "generate_response"

    # 其他自动回复路线继续进入 FAQ 检索节点。
    return "retrieve_faq_answer"

# 使用 CustomerState 定义整个工作流中的共享状态结构。
workflow = StateGraph(CustomerState)

# 注册四个节点：左侧是图中的节点名称，右侧是实际执行的函数。
workflow.add_node("categorize", categorize_query)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("choose_route", choose_route)
workflow.add_node("generate_response", generate_response)
# 注册 FAQ 检索节点，用于从本地知识库补充候选答案。
workflow.add_node("retrieve_faq_answer", retrieve_faq_answer)


# 定义工作流从起点到终点的执行路径。
workflow.add_edge(START, "categorize")
workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_edge("analyze_sentiment", "choose_route")
# 路由节点根据 route 决定是否需要执行 FAQ 检索。
workflow.add_conditional_edges(
    "choose_route",
    should_retrieve_faq,
    {
        "retrieve_faq_answer": "retrieve_faq_answer",
        "generate_response": "generate_response",
    },
)

# FAQ 检索完成后，再进入最终回复节点。
workflow.add_edge("retrieve_faq_answer", "generate_response")
workflow.add_edge("retrieve_faq_answer", "generate_response")
workflow.add_edge("generate_response", END)


# 编译工作流，将设计好的图转换为可执行对象。
app = workflow.compile()


def run_langgraph_customer_service_agent(query: str) -> CustomerState:
    # 将用户问题作为初始状态传入图的 START 节点。
    initial_state: CustomerState = {"query": query}

    # LangGraph 会沿着定义好的边依次执行节点，并返回最终状态。
    return app.invoke(initial_state)




