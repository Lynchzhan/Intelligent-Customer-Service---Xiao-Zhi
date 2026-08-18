import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch
from src.agent import CustomerState

from src.evaluation_cases import (
    EVALUATION_CASES,
    EvaluationCase,
    build_case_distribution,
    validate_evaluation_cases,
)

from src.evaluation_runner import (
    EvaluationResult,
    _build_comparison_metric,
    build_comparison,
    build_summary,
    evaluate_case,
    evaluate_all_cases,
    load_evaluation_results,
    save_comparison_report,
    save_evaluation_report,
)


class EvaluationRunnerTests(unittest.TestCase):
    @patch("src.evaluation_runner.run_langgraph_llm_customer_service_agent")
    @patch("src.evaluation_runner.validate_evaluation_cases")
    def test_invalid_case_validation_stops_before_model_requests(
        self,
        mock_validate,
        mock_run_agent,
    ) -> None:
        # 模拟评估集结构校验失败。
        mock_validate.side_effect = ValueError("评估样本名称不能重复。")

        # 评估入口应直接抛出校验错误。
        with self.assertRaises(ValueError):
            evaluate_all_cases()

        # 结构无效时，任何模型调用都不应该发生。
        mock_run_agent.assert_not_called()

    def test_save_evaluation_report_writes_detail_and_summary_files(self) -> None:
        # 构造一条完整的本地评估结果，不调用真实客服 Agent。
        result: EvaluationResult = {
            "name": "report_sample",
            "query": "退款一般多久到账？",
            "passed": True,
            "category_ok": True,
            "sentiment_ok": True,
            "route_ok": True,
            "faq_ok": True,
            "actual_category": "billing",
            "expected_category": "billing",
            "actual_sentiment": "neutral",
            "expected_sentiment": "neutral",
            "actual_route": "billing_reply",
            "expected_route": "billing_reply",
            "actual_faq_in_state": True,
            "expected_faq_in_state": True,
            "actual_faq_id": "refund_timing",
            "expected_faq_id": "refund_timing",
            "faq_id_ok": True,
            "complexity": "simple",
            "tags": ["billing", "faq_hit", "neutral"],
            "analysis_source": "llm",
            "response_source": "llm",
        }

        # 在项目 tests 目录创建唯一临时路径，适配没有系统 Temp 目录的环境。
        temp_dir = (
            Path(__file__).resolve().parent
            / f".tmp_evaluation_report_{uuid4().hex}"
        )
        try:
            report_dir = save_evaluation_report(
                [result],
                output_root=temp_dir,
                run_id="test-run",
                metadata={
                    "runner": "test",
                    "mode": "offline",
                },
            )

            # 验证函数返回了约定的运行目录。
            self.assertEqual(report_dir, Path(temp_dir) / "test-run")

            # 读取两个 JSON 文件，确认它们都是可解析的结构化数据。
            results_payload = json.loads(
                (report_dir / "results.json").read_text(encoding="utf-8")
            )
            summary_payload = json.loads(
                (report_dir / "summary.json").read_text(encoding="utf-8")
            )

            # 明细文件保留完整样本，中文没有被转成 Unicode 转义序列。
            self.assertEqual(results_payload[0]["query"], "退款一般多久到账？")

            # 汇总文件包含运行标识、样本数量和统计结果。
            self.assertEqual(summary_payload["run_id"], "test-run")
            self.assertEqual(summary_payload["sample_count"], 1)
            self.assertEqual(
                summary_payload["metadata"],
                {
                    "runner": "test",
                    "mode": "offline",
                },
            )
            self.assertIn("created_at", summary_payload)
            self.assertEqual(summary_payload["summary"]["passed"], 1)
            self.assertEqual(
                summary_payload["summary"]["group_metrics"]["complexity"]["simple"],
                {"total": 1, "passed": 1, "pass_rate": 1.0},
            )
            self.assertEqual(
                summary_payload["summary"]["group_metrics"]["tags"]["billing"],
                {"total": 1, "passed": 1, "pass_rate": 1.0},
            )
        finally:
            # 无论断言成功还是失败，都删除本次测试创建的目录。
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_evaluate_case_records_retrieval_and_runtime_metrics(self) -> None:
        # 构造一条 FAQ 命中且回复模型回退的测试样本。
        case: EvaluationCase = {
            "name": "runtime_evidence_sample",
            "query": "退款一般多久到账？",
            "expected_category": "billing",
            "expected_sentiment": "neutral",
            "expected_route": "billing_reply",
            "expected_faq_in_state": True,
            "expected_faq_id": "refund_timing",
            "complexity": "simple",
            "tags": ["billing", "faq_hit", "neutral"],
        }

        faq_answer = (
            "退款申请审核通过后，"
            "原路退款通常在 3 至 5 个工作日到账。"
        )

        # 使用本地函数模拟完整 Agent，完全不调用真实模型。
        def fake_agent_runner(query: str) -> CustomerState:
            return {
                "query": query,
                "category": "billing",
                "sentiment": "neutral",
                "route": "billing_reply",
                "faq_id": "refund_timing",
                "faq_answer": faq_answer,
                "retrieved_contexts": [faq_answer],
                "retrieval_score": 0.834833,
                "retrieval_keyword_score": 0.583333,
                "retrieval_text_score": 0.603599,
                "retrieval_method": "keyword_tfidf_hybrid_v1",
                "retrieval_candidates": [
                    {
                        "rank": 1,
                        "faq_id": "refund_timing",
                        "chunk_id": "refund_timing#0",
                        "title": "退款到账时效",
                        "source": "project_faq",
                        "version": "1.0",
                        "score": 0.834833,
                        "keyword_score": 0.583333,
                        "text_score": 0.603599,
                    }
                ],
                "knowledge_base_name": "customer_service_faq",
                "knowledge_base_version": "2026.08.18",
                "analysis_source": "llm",
                "response_source": "faq_fallback",
                "response_error": "APITimeoutError",
                "response": faq_answer,
                "input_tokens": 18,
                "output_tokens": 7,
                "estimated_cost_usd": 0.0012,
            }

        # 评估器会在 Agent 调用外层测量耗时，
        # 并根据来源字段推断两个模型阶段的调用次数。
        result = evaluate_case(
            case,
            agent_runner=fake_agent_runner,
        )

        # 验证检索证据被逐条结果保存。
        self.assertEqual(
            result["retrieved_contexts"],
            [faq_answer],
        )
        self.assertAlmostEqual(
            result["retrieval_score"],
            0.834833,
        )
        self.assertAlmostEqual(
            result["retrieval_keyword_score"],
            0.583333,
        )
        self.assertGreater(result["retrieval_text_score"], 0.0)
        self.assertEqual(
            result["retrieval_method"],
            "keyword_tfidf_hybrid_v1",
        )
        self.assertEqual(
            result["knowledge_base_version"],
            "2026.08.18",
        )
        self.assertEqual(
            result["retrieval_candidates"][0]["chunk_id"],
            "refund_timing#0",
        )

        # llm 分类一次，faq_fallback 表示回复模型也尝试过一次。
        self.assertEqual(result["analysis_model_calls"], 1)
        self.assertEqual(result["response_model_calls"], 1)
        self.assertEqual(result["model_call_count"], 2)

        # 验证耗时、Token 和成本字段都被保留。
        self.assertGreaterEqual(result["latency_ms"], 0.0)
        self.assertEqual(result["input_tokens"], 18)
        self.assertEqual(result["output_tokens"], 7)
        self.assertAlmostEqual(
            result["estimated_cost_usd"],
            0.0012,
        )
        self.assertEqual(result["response_error"], "APITimeoutError")

    def test_build_summary_records_runtime_metrics_and_failures(self) -> None:
        # 用一个局部工厂函数减少重复字段，
        # 让测试重点集中在运行指标统计。
        def make_result(
            name: str,
            latency_ms: float,
            model_call_count: int,
            analysis_error: str | None = None,
            response_error: str | None = None,
            input_tokens: int | None = None,
            output_tokens: int | None = None,
            estimated_cost_usd: float | None = None,
        ) -> EvaluationResult:
            return {
                "name": name,
                "query": "测试问题",
                "passed": True,
                "category_ok": True,
                "sentiment_ok": True,
                "route_ok": True,
                "faq_ok": True,
                "actual_category": "general",
                "expected_category": "general",
                "actual_sentiment": "neutral",
                "expected_sentiment": "neutral",
                "actual_route": "general_reply",
                "expected_route": "general_reply",
                "actual_faq_in_state": False,
                "expected_faq_in_state": False,
                "actual_faq_id": None,
                "expected_faq_id": None,
                "faq_id_ok": True,
                "complexity": "simple",
                "tags": ["general"],
                "analysis_source": "llm",
                "analysis_error": analysis_error,
                "response_source": "local",
                "response_error": response_error,
                "latency_ms": latency_ms,
                "model_call_count": model_call_count,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost_usd": estimated_cost_usd,
            }

        results = [
            make_result(
                "runtime_one",
                latency_ms=10.0,
                model_call_count=2,
                input_tokens=100,
                output_tokens=20,
                estimated_cost_usd=0.001,
            ),
            make_result(
                "runtime_two",
                latency_ms=20.0,
                model_call_count=0,
                analysis_error="ValueError",
            ),
            make_result(
                "runtime_three",
                latency_ms=40.0,
                model_call_count=1,
                response_error="APITimeoutError",
                input_tokens=50,
                output_tokens=10,
                estimated_cost_usd=0.002,
            ),
        ]

        summary = build_summary(results)

        # 平均延迟为 (10 + 20 + 40) / 3。
        self.assertAlmostEqual(
            summary["latency_ms_average"],
            70 / 3,
        )

        # 三个样本中覆盖 95% 的最近秩样本是 40ms。
        self.assertEqual(summary["latency_ms_p95"], 40.0)

        # 模型调用总数为 2 + 0 + 1。
        self.assertEqual(summary["model_call_total"], 3)
        self.assertAlmostEqual(
            summary["model_call_average"],
            1.0,
        )

        # 只有两条结果提供了 Token 和成本数据。
        self.assertEqual(summary["input_tokens_total"], 150)
        self.assertEqual(summary["output_tokens_total"], 30)
        self.assertAlmostEqual(
            summary["estimated_cost_usd_total"],
            0.003,
        )
        self.assertEqual(summary["token_observation_count"], 2)
        self.assertEqual(summary["cost_observation_count"], 2)

        # 分别统计超时、解析失败和所有错误类型。
        self.assertEqual(summary["timeout_count"], 1)
        self.assertEqual(summary["parse_failure_count"], 1)
        self.assertEqual(
            summary["failure_type_counts"],
            {
                "ValueError": 1,
                "APITimeoutError": 1,
            },
        )

    def test_build_summary_counts_metrics_and_fallbacks(self) -> None:
        # 构造一条大模型分析和大模型回复都成功的结果。
        model_result: EvaluationResult = {
            "name": "model_success",
            "query": "退款一般多久到账？",
            "passed": True,
            "category_ok": True,
            "sentiment_ok": True,
            "route_ok": True,
            "faq_ok": True,
            "actual_category": "billing",
            "expected_category": "billing",
            "actual_sentiment": "neutral",
            "expected_sentiment": "neutral",
            "actual_route": "billing_reply",
            "expected_route": "billing_reply",
            "actual_faq_in_state": True,
            "expected_faq_in_state": True,
            "actual_faq_id": "refund_timing",
            "expected_faq_id": "refund_timing",
            "faq_id_ok": True,
            "complexity": "simple",
            "tags": ["billing", "faq_hit", "neutral"],
            "analysis_source": "llm",
            "response_source": "llm",
        }

        # 构造一条语义正确、但发生了两种降级的结果。
        fallback_result: EvaluationResult = {
            "name": "fallback_success",
            "query": "退款一般多久到账？",
            "passed": True,
            "category_ok": True,
            "sentiment_ok": True,
            "route_ok": True,
            "faq_ok": True,
            "actual_category": "billing",
            "expected_category": "billing",
            "actual_sentiment": "neutral",
            "expected_sentiment": "neutral",
            "actual_route": "billing_reply",
            "expected_route": "billing_reply",
            "actual_faq_in_state": True,
            "expected_faq_in_state": True,
            "actual_faq_id": "refund_timing",
            "expected_faq_id": "refund_timing",
            "faq_id_ok": True,
            "complexity": "simple",
            "tags": ["billing", "faq_hit", "neutral"],
            "analysis_source": "rule_fallback",
            "response_source": "faq_fallback",
        }

        summary = build_summary([model_result, fallback_result])

        # 验证总数和整体通过数。
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["passed"], 2)

        # 验证四项独立指标都统计到了两条正确结果。
        self.assertEqual(summary["category_correct"], 2)
        self.assertEqual(summary["sentiment_correct"], 2)
        self.assertEqual(summary["route_correct"], 2)
        self.assertEqual(summary["faq_correct"], 2)
        self.assertEqual(summary["faq_id_correct"], 2)
        self.assertEqual(
            summary["group_metrics"]["complexity"]["simple"],
            {"total": 2, "passed": 2, "pass_rate": 1.0},
        )
        self.assertEqual(
            summary["group_metrics"]["tags"]["billing"],
            {"total": 2, "passed": 2, "pass_rate": 1.0},
        )

        # 验证两个来源分布和两种降级次数。
        self.assertEqual(
            summary["analysis_source_counts"],
            {"llm": 1, "rule_fallback": 1},
        )
        self.assertEqual(
            summary["response_source_counts"],
            {"llm": 1, "faq_fallback": 1},
        )
        self.assertEqual(summary["rule_fallback_count"], 1)
        self.assertEqual(summary["faq_fallback_count"], 1)

    @patch("src.evaluation_runner.run_langgraph_llm_customer_service_agent")
    def test_rule_fallback_with_correct_result_is_semantically_passed(
        self,
        mock_run_agent,
    ) -> None:
        # 构造一条人工预先标注好的技术负面问题。
        case: EvaluationCase = {
            "name": "fallback_technical_negative",
            "query": "软件打开后一直崩溃，太差了！",
            "expected_category": "technical",
            "expected_sentiment": "negative",
            "expected_route": "human_handoff",
            "expected_faq_in_state": False,
            "expected_faq_id": None,
            "complexity": "medium",
            "tags": ["technical", "negative", "human_handoff"],
        }

        # 模拟完整客服 Agent 在模型分析失败后得到的最终状态。
        # 这里不请求真实 API，而是直接构造降级后的结果。
        mock_run_agent.return_value = {
            "query": case["query"],
            "category": "technical",
            "sentiment": "negative",
            "analysis_source": "rule_fallback",
            "analysis_error": "ValueError",
            "route": "human_handoff",
            "faq_id": None,
            "response": (
                "系统当前繁忙，已使用备用方式继续处理您的问题。\n"
                "您的问题已转交人工客服，请稍候。"
            ),
            "response_source": "local",
        }

        # 执行评估器，而不是直接执行真实客服工作流。
        result = evaluate_case(case)

        # 验证语义结果全部正确，所以这条样本整体通过。
        self.assertTrue(result["passed"])

        # 验证分类、情绪、路线和 FAQ 状态分别通过。
        self.assertTrue(result["category_ok"])
        self.assertTrue(result["sentiment_ok"])
        self.assertTrue(result["route_ok"])
        self.assertTrue(result["faq_ok"])

        # 验证评估器仍然记录了本次发生了规则降级。
        self.assertEqual(result["analysis_source"], "rule_fallback")

        # 验证最终回复确实来自本地 Python 回复逻辑。
        self.assertEqual(result["response_source"], "local")

        # 验证评估器只调用了一次客服 Agent。
        mock_run_agent.assert_called_once_with(case["query"])

    @patch("src.evaluation_runner.run_langgraph_llm_customer_service_agent")
    def test_faq_reply_fallback_is_recorded_separately(
        self,
        mock_run_agent,
    ) -> None:
        # 准备一条能够命中退款 FAQ 的中性账单问题。
        faq_answer = "退款申请审核通过后，原路退款通常在 3 至 5 个工作日到账。"

        case: EvaluationCase = {
            "name": "faq_reply_fallback",
            "query": "退款一般多久到账？",
            "expected_category": "billing",
            "expected_sentiment": "neutral",
            "expected_route": "billing_reply",
            "expected_faq_in_state": True,
            "expected_faq_id": "refund_timing",
            "complexity": "simple",
            "tags": ["billing", "faq_hit", "neutral"],
        }

        # 模拟分析模型成功，但回复模型超时后的最终状态。
        # FAQ 原文仍然被保留，因此最终回复仍然可靠。
        mock_run_agent.return_value = {
            "query": case["query"],
            "category": "billing",
            "sentiment": "neutral",
            "analysis_source": "llm",
            "route": "billing_reply",
            "faq_id": "refund_timing",
            "faq_answer": faq_answer,
            "response": faq_answer,
            "response_source": "faq_fallback",
            "response_error": "APITimeoutError",
        }

        # 执行评估器。
        result = evaluate_case(case)

        # 分类、情绪、路线和 FAQ 状态都正确。
        self.assertTrue(result["passed"])
        self.assertTrue(result["category_ok"])
        self.assertTrue(result["sentiment_ok"])
        self.assertTrue(result["route_ok"])
        self.assertTrue(result["faq_ok"])

        # 分析阶段来自大模型，说明分类分析本身没有失败。
        self.assertEqual(result["analysis_source"], "llm")

        # 回复阶段使用 FAQ 原文兜底。
        self.assertEqual(result["response_source"], "faq_fallback")

        # 验证评估器只运行了一次客服 Agent。
        mock_run_agent.assert_called_once_with(case["query"])

    def test_build_case_distribution_counts_independent_dimensions(self) -> None:
        # 构造三条可控样本。
        # 测试只使用本地字典，不会调用真实模型。
        cases: list[EvaluationCase] = [
            {
                "name": "distribution_billing_simple",
                "query": "退款一般多久到账？",
                "expected_category": "billing",
                "expected_sentiment": "neutral",
                "expected_route": "billing_reply",
                "expected_faq_in_state": False,
                "expected_faq_id": None,
                "complexity": "simple",
                "tags": ["billing", "faq_miss"],
            },
            {
                "name": "distribution_technical_complex",
                "query": "软件打开后一直崩溃，太差了！",
                "expected_category": "technical",
                "expected_sentiment": "negative",
                "expected_route": "human_handoff",
                "expected_faq_in_state": False,
                "expected_faq_id": None,
                "complexity": "complex",
                "tags": ["technical", "negative", "human_handoff"],
            },
            {
                "name": "distribution_billing_medium",
                "query": "支付成功但订单状态没有更新。",
                "expected_category": "billing",
                "expected_sentiment": "positive",
                "expected_route": "billing_reply",
                "expected_faq_in_state": False,
                "expected_faq_id": None,
                "complexity": "medium",
                "tags": ["billing", "positive", "boundary"],
            },
        ]

        # 调用本地分布统计函数。
        distribution = build_case_distribution(cases)

        # 样本总数应该是 3。
        self.assertEqual(distribution["total"], 3)

        # billing 出现 2 次，technical 出现 1 次。
        self.assertEqual(
            distribution["category_counts"],
            {
                "billing": 2,
                "technical": 1,
            },
        )

        # 三种情绪各出现 1 次。
        self.assertEqual(
            distribution["sentiment_counts"],
            {
                "neutral": 1,
                "negative": 1,
                "positive": 1,
            },
        )

        # billing_reply 出现 2 次，human_handoff 出现 1 次。
        self.assertEqual(
            distribution["route_counts"],
            {
                "billing_reply": 2,
                "human_handoff": 1,
            },
        )

        # 三种复杂度各出现 1 次。
        self.assertEqual(
            distribution["complexity_counts"],
            {
                "simple": 1,
                "complex": 1,
                "medium": 1,
            },
        )

        # billing 标签出现 2 次。
        self.assertEqual(
            distribution["tag_counts"]["billing"],
            2,
        )

        # faq_miss 标签出现 1 次。
        self.assertEqual(
            distribution["tag_counts"]["faq_miss"],
            1,
        )

        # human_handoff 标签出现 1 次。
        self.assertEqual(
            distribution["tag_counts"]["human_handoff"],
            1,
        )

        # 因为一条样本可以有多个标签，
        # 所有标签计数之和可以大于样本总数。
        self.assertGreater(
            sum(distribution["tag_counts"].values()),
            distribution["total"],
        )

    def test_evaluate_case_accepts_custom_agent_runner(self) -> None:
        # 使用已有的第一条评估样本。
        case = EVALUATION_CASES[0]

        # 用列表记录自定义 Agent 是否被调用。
        calls: list[str] = []

        def fake_agent(query: str) -> CustomerState:
            # 记录评估器传入的用户问题。
            calls.append(query)

            # 返回与这条样本预期一致的本地状态。
            #
            # 这里不调用真实模型，
            # 只是验证评估器是否真的使用了传入的函数。
            return {
                "query": query,
                "category": "billing",
                "sentiment": "neutral",
                "route": "billing_reply",
                "faq_id": "refund_timing",
                "faq_answer": "本地测试 FAQ 答案",
            }

        # 把 fake_agent 作为待测 Agent 传给评估器。
        result = evaluate_case(
            case,
            agent_runner=fake_agent,
        )

        # 由于 fake_agent 返回的字段符合预期，
        # 这条样本应该通过。
        self.assertTrue(result["passed"])

        # 确认评估器确实调用了自定义 Agent。
        self.assertEqual(
            calls,
            [case["query"]],
        )

    def test_build_comparison_metric_calculates_rates_and_improvement(
        self,
    ) -> None:
        # 模拟基线方案正确 44 条。
        baseline_correct = 44

        # 模拟候选方案正确 46 条。
        candidate_correct = 46

        # 总样本数为 50。
        total = 50

        # 调用本地计算函数。
        metric = _build_comparison_metric(
            baseline_correct=baseline_correct,
            candidate_correct=candidate_correct,
            total=total,
        )

        # 检查原始数量是否被正确保存。
        self.assertEqual(metric["baseline_correct"], 44)
        self.assertEqual(metric["candidate_correct"], 46)
        self.assertEqual(metric["total"], 50)

        # 44 / 50 = 0.88。
        self.assertAlmostEqual(
            metric["baseline_rate"],
            0.88,
        )

        # 46 / 50 = 0.92。
        self.assertAlmostEqual(
            metric["candidate_rate"],
            0.92,
        )

        # 0.92 - 0.88 = 0.04。
        self.assertAlmostEqual(
            metric["absolute_delta"],
            0.04,
        )

        # 0.04 / 0.88 ≈ 0.04545。
        self.assertAlmostEqual(
            metric["relative_improvement"],
            0.04 / 0.88,
        )

    def test_build_comparison_metric_handles_empty_dataset(
        self,
    ) -> None:
        # total 为 0 时，不应该抛出除零错误。
        metric = _build_comparison_metric(
            baseline_correct=0,
            candidate_correct=0,
            total=0,
        )

        # 空数据集的比例统一为 0。
        self.assertEqual(metric["baseline_rate"], 0.0)
        self.assertEqual(metric["candidate_rate"], 0.0)
        self.assertEqual(metric["absolute_delta"], 0.0)

        # 相对提升无法计算，所以应该是 None。
        self.assertIsNone(
            metric["relative_improvement"],
        )

    def test_build_comparison_compares_overall_metric(self) -> None:
        # 这个内部函数用于构造本地测试结果。
        #
        # 它不调用任何 Agent，不请求模型，
        # 只手动生成符合 EvaluationResult 结构的字典。
        def make_result(
            name: str,
            passed: bool,
        ) -> EvaluationResult:
            # passed 为 True 时，模拟五项检查全部通过。
            if passed:
                return {
                    "name": name,
                    "query": "本地比较测试问题",
                    "passed": True,
                    "category_ok": True,
                    "sentiment_ok": True,
                    "route_ok": True,
                    "faq_ok": True,
                    "actual_category": "billing",
                    "expected_category": "billing",
                    "actual_sentiment": "neutral",
                    "expected_sentiment": "neutral",
                    "actual_route": "billing_reply",
                    "expected_route": "billing_reply",
                    "actual_faq_in_state": True,
                    "expected_faq_in_state": True,
                    "actual_faq_id": "refund_timing",
                    "expected_faq_id": "refund_timing",
                    "faq_id_ok": True,
                    "complexity": "simple",
                    "tags": ["billing", "faq_hit"],
                    "analysis_source": "local",
                    "response_source": "local",
                }

            # passed 为 False 时，模拟五项检查全部失败。
            return {
                "name": name,
                "query": "本地比较测试问题",
                "passed": False,
                "category_ok": False,
                "sentiment_ok": False,
                "route_ok": False,
                "faq_ok": False,
                "actual_category": "general",
                "expected_category": "billing",
                "actual_sentiment": "negative",
                "expected_sentiment": "neutral",
                "actual_route": "human_handoff",
                "expected_route": "billing_reply",
                "actual_faq_in_state": False,
                "expected_faq_in_state": True,
                "actual_faq_id": None,
                "expected_faq_id": "refund_timing",
                "faq_id_ok": False,
                "complexity": "simple",
                "tags": ["billing", "faq_hit"],
                "analysis_source": "local",
                "response_source": "local",
            }

        # 基线方案：
        # 第一条通过，第二条失败，因此整体通过数是 1。
        baseline_results = [
            make_result("comparison_001", True),
            make_result("comparison_002", False),
        ]

        # 候选方案：
        # 同样两条样本都通过，因此整体通过数是 2。
        candidate_results = [
            make_result("comparison_001", True),
            make_result("comparison_002", True),
        ]

        # 比较两套同名、同顺序的结果。
        comparison = build_comparison(
            baseline_results,
            candidate_results,
        )

        # 两套结果都使用两条样本。
        self.assertEqual(comparison["total"], 2)

        # 读取整体通过率的比较数据。
        overall = comparison["metrics"]["overall"]

        # 基线通过 1 条，候选通过 2 条。
        self.assertEqual(overall["baseline_correct"], 1)
        self.assertEqual(overall["candidate_correct"], 2)

        # 基线通过率：1 / 2 = 0.5。
        self.assertAlmostEqual(overall["baseline_rate"], 0.5)

        # 候选通过率：2 / 2 = 1.0。
        self.assertAlmostEqual(overall["candidate_rate"], 1.0)

        # 绝对提升：1.0 - 0.5 = 0.5。
        self.assertAlmostEqual(overall["absolute_delta"], 0.5)

        # 相对提升：0.5 / 0.5 = 1.0。
        self.assertAlmostEqual(
            overall["relative_improvement"],
            1.0,
        )

    def test_build_comparison_rejects_different_sample_names(
        self,
    ) -> None:
        # 构造一条基线结果。
        baseline_result: EvaluationResult = {
            "name": "comparison_001",
            "query": "退款一般多久到账？",
            "passed": True,
            "category_ok": True,
            "sentiment_ok": True,
            "route_ok": True,
            "faq_ok": True,
            "actual_category": "billing",
            "expected_category": "billing",
            "actual_sentiment": "neutral",
            "expected_sentiment": "neutral",
            "actual_route": "billing_reply",
            "expected_route": "billing_reply",
            "actual_faq_in_state": True,
            "expected_faq_in_state": True,
            "actual_faq_id": "refund_timing",
            "expected_faq_id": "refund_timing",
            "faq_id_ok": True,
            "complexity": "simple",
            "tags": ["billing", "faq_hit"],
            "analysis_source": "local",
            "response_source": "local",
        }

        # 复制基线结果，但把候选结果的 name 改成另一个名称。
        #
        # **baseline_result 表示复制原字典中的所有键值。
        # 后面的 name 会覆盖原来的 name。
        candidate_result = {
            **baseline_result,
            "name": "comparison_002",
        }

        # 样本名称不一致时，比较函数应该拒绝执行。
        with self.assertRaises(ValueError):
            build_comparison(
                [baseline_result],
                [candidate_result],
            )

    def test_save_comparison_report_writes_three_json_files(
        self,
    ) -> None:
        # 构造一条完整的本地评估结果。
        result: EvaluationResult = {
            "name": "comparison_report_sample",
            "query": "退款一般多久到账？",
            "passed": True,
            "category_ok": True,
            "sentiment_ok": True,
            "route_ok": True,
            "faq_ok": True,
            "actual_category": "billing",
            "expected_category": "billing",
            "actual_sentiment": "neutral",
            "expected_sentiment": "neutral",
            "actual_route": "billing_reply",
            "expected_route": "billing_reply",
            "actual_faq_in_state": True,
            "expected_faq_in_state": True,
            "actual_faq_id": "refund_timing",
            "expected_faq_id": "refund_timing",
            "faq_id_ok": True,
            "complexity": "simple",
            "tags": ["billing", "faq_hit"],
            "analysis_source": "local",
            "response_source": "local",
        }

        # 使用 tests 目录下的唯一临时目录。
        temp_dir = (
            Path(__file__).resolve().parent
            / f".tmp_comparison_{uuid4().hex}"
        )

        try:
            # 使用同一条结果模拟基线和候选都成功。
            report_dir = save_comparison_report(
                baseline_results=[result],
                candidate_results=[result],
                output_root=temp_dir,
                comparison_id="test-comparison",
            )

            # 确认返回路径正确。
            self.assertEqual(
                report_dir,
                temp_dir / "test-comparison",
            )

            # 确认三个文件都存在。
            self.assertTrue(
                (report_dir / "baseline_results.json").exists()
            )
            self.assertTrue(
                (report_dir / "candidate_results.json").exists()
            )
            self.assertTrue(
                (report_dir / "comparison.json").exists()
            )

            # 读取比较汇总文件。
            comparison_payload = json.loads(
                (report_dir / "comparison.json").read_text(
                    encoding="utf-8"
                )
            )

            # 检查运行元数据。
            self.assertEqual(
                comparison_payload["comparison_id"],
                "test-comparison",
            )
            self.assertEqual(
                comparison_payload["sample_count"],
                1,
            )

            # 基线和候选都通过，因此 overall 都是 1/1。
            self.assertEqual(
                comparison_payload["comparison"]["metrics"]["overall"][
                    "baseline_rate"
                ],
                1.0,
            )
            self.assertEqual(
                comparison_payload["comparison"]["metrics"]["overall"][
                    "candidate_rate"
                ],
                1.0,
            )

        finally:
            # 测试结束后删除临时报告目录。
            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )

    def test_load_evaluation_results_reads_saved_results(
        self,
    ) -> None:
        # 构造一条完整的本地评估结果。
        # 测试中不会调用任何 Agent 或模型。
        result: EvaluationResult = {
            "name": "load_report_sample",
            "query": "退款一般多久到账？",
            "passed": True,
            "category_ok": True,
            "sentiment_ok": True,
            "route_ok": True,
            "faq_ok": True,
            "actual_category": "billing",
            "expected_category": "billing",
            "actual_sentiment": "neutral",
            "expected_sentiment": "neutral",
            "actual_route": "billing_reply",
            "expected_route": "billing_reply",
            "actual_faq_in_state": True,
            "expected_faq_in_state": True,
            "actual_faq_id": "refund_timing",
            "expected_faq_id": "refund_timing",
            "faq_id_ok": True,
            "complexity": "simple",
            "tags": ["billing", "faq_hit"],
            "analysis_source": "local",
            "response_source": "local",
        }

        # 在 tests 目录创建本次测试专用的唯一临时目录。
        temp_dir = (
            Path(__file__).resolve().parent
            / f".tmp_load_results_{uuid4().hex}"
        )

        try:
            # 先复用现有保存函数，生成真实格式的 results.json。
            report_dir = save_evaluation_report(
                [result],
                output_root=temp_dir,
                run_id="load-test",
            )

            # 再使用新函数读取同一份结果文件。
            loaded_results = load_evaluation_results(report_dir)

            # 读取结果必须与原先保存的数据完全相同。
            self.assertEqual(loaded_results, [result])

        finally:
            # 无论测试通过或失败，都删除临时目录。
            shutil.rmtree(temp_dir, ignore_errors=True)




if __name__ == "__main__":
    # 直接运行本文件时，启动 unittest 测试。
    unittest.main()
