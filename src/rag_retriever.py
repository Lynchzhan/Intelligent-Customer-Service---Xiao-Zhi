"""轻量本地混合 RAG 检索器。

当前知识库很小，因此使用标准库实现 TF-IDF 余弦相似度：不需要向量数据库、
不下载嵌入模型，也能在离线模式运行。它不是语义向量检索；中文同义改写能力
有限，所以仍由严格主题/意图规则先决定“能否进入候选集”。
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from src.knowledge_base import FAQ_ENTRIES, FaqEntry, FaqMatch


# 用于写入运行状态和报告，明确本次检索采用的具体算法版本。
RETRIEVAL_METHOD = "keyword_tfidf_hybrid_v1"


def _normalize_text(text: str) -> str:
    """统一大小写并移除空白、标点，使中文字符 n-gram 更稳定。"""

    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", text.lower())


def tokenize_for_tfidf(text: str) -> list[str]:
    """把中文文本转为单字和双字 n-gram 词项。"""

    normalized = _normalize_text(text)
    if not normalized:
        return []

    # 单字让“退款”“密码”等短中文关键词仍可产生重叠；
    # 双字提供比单字更具体的局部语义，例如“到账”“重置”。
    unigrams = list(normalized)
    bigrams = [
        normalized[index : index + 2]
        for index in range(len(normalized) - 1)
    ]
    return unigrams + bigrams


def _entry_index_text(entry: FaqEntry) -> str:
    """选择进入本地索引的字段，不把运行时答案之外的数据混入上下文。"""

    return " ".join(
        (
            entry["title"],
            entry["content"],
            *entry["required_keywords"],
            *entry["intent_keywords"],
        )
    )


@dataclass(frozen=True)
class LocalTfidfIndex:
    """不可变的本地 TF-IDF 索引，保存文档词频和逆文档频率。"""

    entries: tuple[FaqEntry, ...]
    document_term_counts: tuple[Counter[str], ...]
    document_lengths: tuple[int, ...]
    inverse_document_frequency: dict[str, float]

    def cosine_similarity(self, query: str, document_index: int) -> float:
        """计算一个查询与指定知识文档之间的 TF-IDF 余弦相似度。"""

        query_counts = Counter(tokenize_for_tfidf(query))
        query_length = sum(query_counts.values())
        if not query_counts or query_length == 0:
            return 0.0

        document_counts = self.document_term_counts[document_index]
        document_length = self.document_lengths[document_index]
        if document_length == 0:
            return 0.0

        query_weights = {
            term: (count / query_length)
            * self.inverse_document_frequency.get(term, 0.0)
            for term, count in query_counts.items()
        }
        document_weights = {
            term: (count / document_length)
            * self.inverse_document_frequency.get(term, 0.0)
            for term, count in document_counts.items()
        }

        dot_product = sum(
            query_weight * document_weights.get(term, 0.0)
            for term, query_weight in query_weights.items()
        )
        query_norm = math.sqrt(
            sum(weight * weight for weight in query_weights.values())
        )
        document_norm = math.sqrt(
            sum(weight * weight for weight in document_weights.values())
        )
        if query_norm == 0.0 or document_norm == 0.0:
            return 0.0

        # 浮点误差极小概率会让结果超过 1，因此显式限定范围。
        return max(0.0, min(1.0, dot_product / (query_norm * document_norm)))


def build_tfidf_index(entries: Iterable[FaqEntry]) -> LocalTfidfIndex:
    """由任意 FAQ 条目构建离线可复现的 TF-IDF 索引。"""

    frozen_entries = tuple(entries)
    if not frozen_entries:
        raise ValueError("不能为零条知识文档构建检索索引。")

    document_term_counts = tuple(
        Counter(tokenize_for_tfidf(_entry_index_text(entry)))
        for entry in frozen_entries
    )
    document_lengths = tuple(
        sum(term_counts.values())
        for term_counts in document_term_counts
    )
    document_frequency = Counter(
        term
        for term_counts in document_term_counts
        for term in term_counts
    )
    document_count = len(frozen_entries)
    inverse_document_frequency = {
        # 平滑公式避免极端高频/低频词出现除零。
        term: math.log((document_count + 1) / (frequency + 1)) + 1.0
        for term, frequency in document_frequency.items()
    }

    return LocalTfidfIndex(
        entries=frozen_entries,
        document_term_counts=document_term_counts,
        document_lengths=document_lengths,
        inverse_document_frequency=inverse_document_frequency,
    )


DEFAULT_TFIDF_INDEX = build_tfidf_index(FAQ_ENTRIES)


def _keyword_score(query: str, entry: FaqEntry) -> float | None:
    """执行严格规则门槛；通过时返回可解释的关键词相关性分数。"""

    # 所有业务主题关键词都必须出现。
    if not all(keyword in query for keyword in entry["required_keywords"]):
        return None

    matched_intent_count = sum(
        keyword in query
        for keyword in entry["intent_keywords"]
    )
    # 用户必须表达至少一个具体意图，防止“退款已经到账”命中时效 FAQ。
    if matched_intent_count == 0:
        return None

    # 主题门槛提供 0.5；后半部分来自命中的意图词覆盖比例。
    return 0.5 + 0.5 * (
        matched_intent_count / len(entry["intent_keywords"])
    )


def _hybrid_score(keyword_score: float, text_score: float) -> float:
    """把规则置信度和 TF-IDF 文本证据融合为最终检索分数。"""

    # keyword_score 已经代表严格主题/意图通过后的最低可信度。
    # TF-IDF 只能补足剩余不确定性，不能让未通过规则的内容进入候选。
    return keyword_score + (1.0 - keyword_score) * text_score


def search_faq_entries(query: str, top_k: int = 3) -> list[FaqMatch]:
    """返回按混合分数排序的 Top-K FAQ 候选。"""

    if top_k < 1:
        raise ValueError("top_k 必须大于 0。")
    if not query.strip():
        return []

    matches: list[FaqMatch] = []
    for document_index, entry in enumerate(DEFAULT_TFIDF_INDEX.entries):
        keyword_score = _keyword_score(query, entry)
        if keyword_score is None:
            continue

        text_score = DEFAULT_TFIDF_INDEX.cosine_similarity(
            query,
            document_index,
        )
        matches.append(
            {
                "entry": entry,
                "score": round(_hybrid_score(keyword_score, text_score), 6),
                "keyword_score": round(keyword_score, 6),
                "text_score": round(text_score, 6),
                "retrieval_method": RETRIEVAL_METHOD,
            }
        )

    # faq_id 作为并列时的第二排序键，使保存的报告在不同运行中保持稳定。
    matches.sort(
        key=lambda match: (-match["score"], match["entry"]["faq_id"]),
    )
    return matches[:top_k]
