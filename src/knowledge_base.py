"""本地 FAQ 知识文档的加载、校验与兼容接口。

本模块只处理“知识是什么”：文档位置、字段结构与加载校验。
具体如何计算文本相似度和排序，放在 rag_retriever.py，避免数据层和
检索算法层互相缠绕。
"""

import json
from pathlib import Path
from typing import Literal, TypedDict, cast


# FAQ 的稳定标识。评估集使用它判断是否命中了正确知识。
FaqId = Literal["refund_timing", "service_hours", "password_reset"]


class KnowledgeBaseMetadata(TypedDict):
    """描述一次检索使用的知识库快照。"""

    name: str
    version: str


class FaqEntry(TypedDict):
    """一个可独立检索、可直接作为回复依据的 FAQ 文档块。"""

    # FAQ 稳定标识。正文改写时应尽量保持此字段不变。
    faq_id: FaqId

    # 文档块标识。当前每篇短 FAQ 只含一个块，因此使用 <faq_id>#0。
    # 以后长文切块时，可以在同一个 faq_id 下增加 #1、#2。
    chunk_id: str

    # 供界面、日志与人工审核阅读的标题和正文。
    title: str
    content: str

    # 知识元数据，用于按业务、来源、版本追踪检索证据。
    category: Literal["technical", "billing", "general"]
    source: str
    version: str
    updated_at: str

    # 严格规则门槛：所有主题词和至少一个意图词都必须命中。
    required_keywords: tuple[str, ...]
    intent_keywords: tuple[str, ...]

    # 保留旧字段，避免现有回复模块与调用代码同时大范围变动。
    # 当前加载时会检查它与 content 一致。
    answer: str


class FaqMatch(TypedDict):
    """混合检索返回的一条候选及其可解释分数。"""

    entry: FaqEntry
    score: float
    keyword_score: float
    text_score: float
    retrieval_method: str


class RetrievalCandidate(TypedDict):
    """写入工作流状态和报告的候选摘要，不重复保存整份正文。"""

    rank: int
    faq_id: FaqId
    chunk_id: str
    title: str
    source: str
    version: str
    score: float
    keyword_score: float
    text_score: float


# 使用绝对路径定位项目文件，避免从不同 PowerShell 工作目录启动时找不到数据。
KNOWLEDGE_BASE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"
)


def _required_string(
    value: object,
    field_name: str,
    document_index: int,
) -> str:
    """校验知识 JSON 中必须存在的非空字符串字段。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "知识库第 "
            f"{document_index} 条文档的 {field_name} 必须是非空字符串。"
        )
    return value.strip()


def _required_string_list(
    value: object,
    field_name: str,
    document_index: int,
) -> tuple[str, ...]:
    """校验 JSON 数组并转成不可变元组，保护加载后的知识数据。"""

    if not isinstance(value, list) or not value:
        raise ValueError(
            "知识库第 "
            f"{document_index} 条文档的 {field_name} 必须是非空字符串数组。"
        )

    items = tuple(
        _required_string(item, field_name, document_index)
        for item in value
    )
    if len(items) != len(set(items)):
        raise ValueError(
            "知识库第 "
            f"{document_index} 条文档的 {field_name} 不能包含重复关键词。"
        )
    return items


def load_knowledge_base(
    path: Path = KNOWLEDGE_BASE_PATH,
) -> tuple[KnowledgeBaseMetadata, list[FaqEntry]]:
    """从 JSON 读取并校验知识文档，返回元数据和 FAQ 列表。"""

    try:
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"找不到知识库文件：{path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"知识库 JSON 格式错误：{path}") from error

    if not isinstance(raw_payload, dict):
        raise ValueError("知识库顶层必须是 JSON 对象。")

    metadata: KnowledgeBaseMetadata = {
        "name": _required_string(
            raw_payload.get("knowledge_base_name"),
            "knowledge_base_name",
            0,
        ),
        "version": _required_string(
            raw_payload.get("knowledge_base_version"),
            "knowledge_base_version",
            0,
        ),
    }

    raw_documents = raw_payload.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ValueError("知识库 documents 必须是非空数组。")

    entries: list[FaqEntry] = []
    seen_faq_ids: set[str] = set()
    seen_chunk_ids: set[str] = set()
    allowed_categories = {"technical", "billing", "general"}

    for index, raw_document in enumerate(raw_documents, start=1):
        if not isinstance(raw_document, dict):
            raise ValueError(f"知识库第 {index} 条文档必须是 JSON 对象。")

        faq_id = _required_string(raw_document.get("faq_id"), "faq_id", index)
        chunk_id = _required_string(
            raw_document.get("chunk_id"),
            "chunk_id",
            index,
        )
        category = _required_string(
            raw_document.get("category"),
            "category",
            index,
        )
        if category not in allowed_categories:
            raise ValueError(
                f"知识库第 {index} 条文档的 category 不受支持：{category}"
            )
        if faq_id in seen_faq_ids:
            raise ValueError(f"知识库 faq_id 重复：{faq_id}")
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"知识库 chunk_id 重复：{chunk_id}")

        content = _required_string(raw_document.get("content"), "content", index)
        answer = _required_string(raw_document.get("answer"), "answer", index)
        if answer != content:
            raise ValueError(
                f"知识库第 {index} 条文档的 answer 必须与 content 一致。"
            )

        entries.append(
            {
                # 当前项目只有三种固定 FAQ ID；cast 仅在已校验 JSON
                # 的边界处把动态字符串写入静态类型结构。
                "faq_id": cast(FaqId, faq_id),
                "chunk_id": chunk_id,
                "title": _required_string(raw_document.get("title"), "title", index),
                "content": content,
                "category": cast(
                    Literal["technical", "billing", "general"],
                    category,
                ),
                "source": _required_string(raw_document.get("source"), "source", index),
                "version": _required_string(raw_document.get("version"), "version", index),
                "updated_at": _required_string(
                    raw_document.get("updated_at"),
                    "updated_at",
                    index,
                ),
                "required_keywords": _required_string_list(
                    raw_document.get("required_keywords"),
                    "required_keywords",
                    index,
                ),
                "intent_keywords": _required_string_list(
                    raw_document.get("intent_keywords"),
                    "intent_keywords",
                    index,
                ),
                "answer": answer,
            }
        )
        seen_faq_ids.add(faq_id)
        seen_chunk_ids.add(chunk_id)

    return metadata, entries


# 进程启动时读取一次。知识文件有格式错误时，应在运行服务前直接失败，
# 而不是在某个用户请求到达后才悄悄返回错误答案。
KNOWLEDGE_BASE_METADATA, FAQ_ENTRIES = load_knowledge_base()


def search_faq_entries(query: str, top_k: int = 3) -> list[FaqMatch]:
    """兼容旧入口：委托给独立的 RAG 检索模块。"""

    # 延迟导入避免 knowledge_base 与 rag_retriever 在模块初始化时循环依赖。
    from src.rag_retriever import search_faq_entries as search

    return search(query, top_k=top_k)


def find_faq_entry(query: str) -> FaqEntry | None:
    """兼容旧接口：返回排名第一的知识条目。"""

    matches = search_faq_entries(query, top_k=1)
    return matches[0]["entry"] if matches else None


def find_faq_answer(query: str) -> str | None:
    """兼容旧接口：只返回排名第一条知识的答案字符串。"""

    entry = find_faq_entry(query)
    return entry["answer"] if entry is not None else None
