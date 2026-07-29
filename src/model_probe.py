from src.model_client import create_openai_client
from src.model_config import load_model_config


def probe_model_connection() -> str:
    # 读取配置，主要是取得本次请求要使用的模型名称。
    config = load_model_config()

    # 创建已经配置好 Base URL 和 API Key 的客户端。
    client = create_openai_client()

    # 发起一次最小的聊天补全请求。
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: CONNECTION_OK",
            }
        ],
        temperature=0,
        max_tokens=16,
    )

    # 从模型响应中取出第一条候选回复的文本内容。
    answer = response.choices[0].message.content

    # 防止服务返回了响应对象，但没有实际文本。
    if not answer:
        raise RuntimeError("模型响应中没有文本内容。")

    # 去掉回复首尾可能存在的空白字符。
    return answer.strip()


if __name__ == "__main__":
    # 只有直接运行当前模块时，才发起连通性测试。
    print(probe_model_connection())