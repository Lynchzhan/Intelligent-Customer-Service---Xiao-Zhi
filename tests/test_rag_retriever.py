import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.knowledge_base import (
    KNOWLEDGE_BASE_METADATA,
    FAQ_ENTRIES,
    load_knowledge_base,
)
from src.rag_retriever import (
    RETRIEVAL_METHOD,
    build_tfidf_index,
    search_faq_entries,
    tokenize_for_tfidf,
)


class RagRetrieverTests(unittest.TestCase):
    def test_json_knowledge_base_has_versioned_document_chunks(self) -> None:
        # 知识内容已经从 Python 常量迁移到可独立维护的 JSON 快照。
        self.assertEqual(
            KNOWLEDGE_BASE_METADATA["name"],
            "customer_service_faq",
        )
        self.assertEqual(
            KNOWLEDGE_BASE_METADATA["version"],
            "2026.08.18",
        )
        self.assertEqual(len(FAQ_ENTRIES), 3)
        self.assertEqual(FAQ_ENTRIES[0]["chunk_id"], "refund_timing#0")

    def test_loader_rejects_content_and_answer_mismatch(self) -> None:
        # answer 是旧调用方兼容字段，必须和真正检索上下文 content 一致。
        payload = """{
          "knowledge_base_name": "test",
          "knowledge_base_version": "1",
          "documents": [{
            "faq_id": "refund_timing",
            "chunk_id": "refund_timing#0",
            "title": "退款",
            "content": "真实正文",
            "category": "billing",
            "source": "test",
            "version": "1.0",
            "updated_at": "2026-08-18",
            "required_keywords": ["退款"],
            "intent_keywords": ["多久"],
            "answer": "不一致正文"
          }]
        }"""

        with NamedTemporaryFile(
            mode="w",
            suffix=".json",
            encoding="utf-8",
            delete=False,
        ) as temporary_file:
            temporary_file.write(payload)
            temporary_path = Path(temporary_file.name)

        try:
            with self.assertRaises(ValueError):
                load_knowledge_base(temporary_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def test_tokenizer_contains_chinese_unigrams_and_bigrams(self) -> None:
        tokens = tokenize_for_tfidf("退款什么时候到账？")

        # 单字用于短文本覆盖，双字用于提高“到账”等局部短语的区分度。
        self.assertIn("退", tokens)
        self.assertIn("退款", tokens)
        self.assertIn("到账", tokens)

    def test_hybrid_search_returns_explainable_scores(self) -> None:
        matches = search_faq_entries("退款一般多久到账？", top_k=3)

        self.assertEqual(len(matches), 1)
        match = matches[0]
        self.assertEqual(match["entry"]["faq_id"], "refund_timing")
        self.assertEqual(match["retrieval_method"], RETRIEVAL_METHOD)
        self.assertGreater(match["keyword_score"], 0.5)
        self.assertGreater(match["text_score"], 0.0)
        self.assertGreaterEqual(match["score"], match["keyword_score"])
        self.assertLessEqual(match["score"], 1.0)

    def test_negative_control_does_not_enter_hybrid_candidates(self) -> None:
        # 这是“已经到账”的陈述，不是“多久到账”的查询。
        self.assertEqual(
            search_faq_entries("退款已经到账，谢谢客服！"),
            [],
        )

    def test_tfidf_index_can_be_built_for_a_small_local_corpus(self) -> None:
        index = build_tfidf_index(FAQ_ENTRIES)

        # 索引是纯本地确定性对象，不依赖网络或模型服务。
        self.assertEqual(len(index.entries), 3)
        self.assertGreater(
            index.cosine_similarity("如何重置密码？", 2),
            0.0,
        )

