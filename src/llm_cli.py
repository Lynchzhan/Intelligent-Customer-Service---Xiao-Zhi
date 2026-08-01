from src.langgraph_llm_agent import run_langgraph_llm_customer_service_agent
from src.observability import format_observability


def main() -> None:
    # 从终端读取用户问题，并删除首尾空白字符。
    query = input("请输入您的问题：").strip()

    # 空问题没有处理意义，直接提示并结束程序。
    if not query:
        print("问题不能为空，请重新运行后输入内容。")
        return

    # 执行包含模型分类和受控回复的 LangGraph 工作流。
    final_state = run_langgraph_llm_customer_service_agent(query)

    print("\n--- 大模型客服处理结果 ---")
    print(f"分类：{final_state['category']}")
    print(f"情绪：{final_state['sentiment']}")
    print(f"路线：{final_state['route']}")

    # 展示分类和回复分别来自哪条路径，便于开发者排查问题。
    for line in format_observability(final_state):
        print(line)

    print(f"客服回复：{final_state['response']}")


if __name__ == "__main__":
    # 只有直接运行本文件时才启动交互程序。
    main()
