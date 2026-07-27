from src.langgraph_agent import run_langgraph_customer_service_agent


def main() -> None:
    # 从终端读取用户输入，并去掉首尾空白字符。
    query = input("请输入您的问题：").strip()

    # 空问题没有业务意义，直接提示并结束程序。
    if not query:
        print("问题不能为空，请重新运行后输入内容。")
        return

    # 使用 LangGraph 版本执行完整客服工作流。
    final_state = run_langgraph_customer_service_agent(query)

    # 输出处理过程中的关键状态，便于当前学习和调试。
    print("\n--- 客服处理结果 ---")
    print(f"分类：{final_state['category']}")
    print(f"情绪：{final_state['sentiment']}")
    print(f"路线：{final_state['route']}")

    # 单独输出最终给用户的客服回复。
    print(f"客服回复：{final_state['response']}")


if __name__ == "__main__":
    # 只有直接运行此文件时，才启动终端交互。
    main()