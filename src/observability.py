from src.agent import CustomerState


def format_observability(state: CustomerState) -> list[str]:
    # 读取可选的来源和错误字段；规则版状态可能没有这些键。
    analysis_source = state.get("analysis_source", "local")
    analysis_error = state.get("analysis_error")
    response_source = state.get("response_source", "local")
    response_error = state.get("response_error")

    lines = [
        f"分析来源：{analysis_source}",
        f"回复来源：{response_source}",
    ]

    # 只有发生错误时才展示错误类型，避免输出无意义的空字段。
    if analysis_error:
        lines.append(f"分析错误：{analysis_error}")

    if response_error:
        lines.append(f"回复错误：{response_error}")

    return lines
