import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class ModelConfig:
    # OpenAI 兼容接口的基础地址。
    base_url: str

    # 只在本地内存中使用的 API Key。
    api_key: str

    # 当前要调用的模型名称。
    model: str


def required_env(name: str) -> str:
    # 从环境变量中读取指定配置。
    value = os.getenv(name)

    # 配置缺失时，给出明确错误，而不是等到模型调用时才失败。
    if not value:
        raise RuntimeError(f"缺少环境变量：{name}")

    # 去掉可能误输入在首尾的空格。
    return value.strip()


def load_model_config() -> ModelConfig:
    # 读取项目根目录的 .env 文件到环境变量中。
    load_dotenv()

    # 创建并返回统一的模型配置对象。
    return ModelConfig(
        base_url=required_env("OPENAI_COMPATIBLE_BASE_URL").rstrip("/"),
        api_key=required_env("OPENAI_COMPATIBLE_API_KEY"),
        model=required_env("OPENAI_COMPATIBLE_MODEL"),
    )