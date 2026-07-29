from openai import OpenAI

from src.model_config import load_model_config


def create_openai_client() -> OpenAI:
    # 从本地 .env 读取并校验模型配置。
    config = load_model_config()

    # 创建 OpenAI 兼容客户端；这里只创建对象，不会发送网络请求。
    return OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=30.0,
        max_retries=0,
    )