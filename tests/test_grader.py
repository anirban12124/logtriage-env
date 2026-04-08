"""
Tests for src/grader.py — final episode grading across 10 components.

All scores are clamped to the open interval (0, 1) by _clamp01.
Never exactly 0.0 or 1.0.  The epsilon is 0.001.
"""

import pytest
from src.grader import TaskGrader, _clamp01, _EPS
from src.tasks import TASKS


# ─── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def grader():
    return TaskGrader()


def _easy_gt():
    return TASKS["task_easy"]["ground_truth"]


def _medium_gt():
    return TASKS["task_medium"]["ground_truth"]


def _hard_gt():
    return TASKS["task_hard"]["ground_truth"]


def _easy_config():
    return TASKS["task_easy"]


def _medium_config():
    return TASKS["task_medium"]


def _hard_config():
    return TASKS["task_hard"]


def _default_behavior(steps=10, report_source="submitted"):
    return {"steps_taken": steps, "report_source": report_source}


# Helper: assert a score is strictly in (0, 1)
def _assert_open01(val, label="score"):
    assert 0.0 < val < 1.0, f"{label} = {val} is not in the open interval (0, 1)"


# ─── Overall grading structure ───────────────────────────────────


class TestGraderStructure:
    """Verify grader output format and constraints."""

    def test_grade_returns_correct_keys(self, grader):
        result = grader.grade(
            {}, [], "", "", _easy_gt(), _easy_config(), _default_behavior()
        )
        assert "task_id" in result
        assert "final_score" in result
        assert "components" in result
        assert result["task_id"] == "task_easy"

    def test_final_score_strictly_between_0_and_1(self, grader):
        result = grader.grade(
            {}, [], "", "", _easy_gt(), _easy_config(), _default_behavior()
        )
        _assert_open01(result["final_score"], "final_score")
        _assert_open01(result["score"], "score")

    def test_all_10_components_present(self, grader):
        result = grader.grade(
            {}, [], "", "", _easy_gt(), _easy_config(), _default_behavior()
        )
        expected_components = {
            "annotation_precision", "annotation_recall", "annotation_quality",
            "correlation_precision", "correlation_recall", "chain_reconstruction",
            "severity_classification", "report_completeness", "report_coherence",
            "investigation_efficiency",
        }
        assert set(result["components"].keys()) == expected_components

    def test_component_scores_strictly_between_0_and_1(self, grader):
        """Every component score must be strictly in (0, 1)."""
        result = grader.grade(
            {"log_008": "error"}, [], "MEDIUM", "Test report",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        for name, comp in result["components"].items():
            assert "score" in comp, f"{name} missing 'score'"
            assert "weight" in comp, f"{name} missing 'weight'"
            assert "weighted" in comp, f"{name} missing 'weighted'"
            assert "detail" in comp, f"{name} missing 'detail'"
            _assert_open01(comp["score"], f"{name}.score")

    def test_weights_sum_to_one(self, grader):
        """Grader weights for each task should sum to ~1.0."""
        for task_id in ["task_easy", "task_medium", "task_hard"]:
            weights = TASKS[task_id]["grader_weights"]
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.001, (
                f"{task_id} weights sum to {total}, expected 1.0"
            )

    def test_no_score_is_exactly_0_or_1(self, grader):
        """Comprehensive: run multiple scenarios, verify NO score is 0.0 or 1.0."""
        scenarios = [
            # Empty agent
            ({}, [], "", "", _easy_gt(), _easy_config(), _default_behavior()),
            # Perfect annotations
            ({"log_008": "error", "log_023": "error", "log_041": "error"},
             [], "MEDIUM", "database connection refused auth-service affected postgresql port 5432",
             _easy_gt(), _easy_config(), _default_behavior(steps=5)),
            # Wrong annotations
            ({"log_001": "error"}, [], "LOW", "short",
             _easy_gt(), _easy_config(), _default_behavior(steps=15)),
            # Perfect everything (medium)
            (_medium_gt()["annotations"],
             _medium_gt()["correlations"], "HIGH",
             "payment service database pool exhausted order service queue backup api gateway 503 errors",
             _medium_gt(), _medium_config(), _default_behavior(steps=10)),
        ]
        for i, args in enumerate(scenarios):
            result = grader.grade(*args)
            _assert_open01(result["score"], f"scenario[{i}].score")
            _assert_open01(result["final_score"], f"scenario[{i}].final_score")
            for name, comp in result["components"].items():
                _assert_open01(comp["score"], f"scenario[{i}].{name}.score")


# ─── Annotation precision ───────────────────────────────────────


class TestAnnotationPrecision:
    """Component A: Annotation Precision."""

    def test_perfect_precision(self, grader):
        result = grader.grade(
            {"log_008": "error", "log_023": "error", "log_041": "error"},
            [], "MEDIUM", "Report",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        score = result["components"]["annotation_precision"]["score"]
        assert score >= 0.99  # Clamped from 1.0 → 0.999

    def test_zero_precision(self, grader):
        result = grader.grade(
            {"log_001": "error", "log_002": "error"},
            [], "MEDIUM", "Report",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        score = result["components"]["annotation_precision"]["score"]
        assert score <= 0.01  # Clamped from 0.0 → 0.001
        _assert_open01(score)

    def test_partial_precision(self, grader):
        result = grader.grade(
            {"log_008": "error", "log_001": "error"},
            [], "MEDIUM", "Report",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        assert result["components"]["annotation_precision"]["score"] == 0.5

    def test_no_annotations_with_gt(self, grader):
        result = grader.grade(
            {}, [], "MEDIUM", "Report",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        score = result["components"]["annotation_precision"]["score"]
        assert score <= 0.01
        _assert_open01(score)


# ─── Annotation recall ──────────────────────────────────────────


class TestAnnotationRecall:
    """Component B: Annotation Recall."""

    def test_perfect_recall(self, grader):
        result = grader.grade(
            {"log_008": "error", "log_023": "error", "log_041": "error"},
            [], "MEDIUM", "Report",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        score = result["components"]["annotation_recall"]["score"]
        assert score >= 0.99

    def test_zero_recall(self, grader):
        result = grader.grade(
            {}, [], "MEDIUM", "Report",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        score = result["components"]["annotation_recall"]["score"]
        assert score <= 0.01
        _assert_open01(score)

    def test_partial_recall(self, grader):
        result = grader.grade(
            {"log_008": "error"},
            [], "MEDIUM", "Report",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        score = result["components"]["annotation_recall"]["score"]
        assert abs(score - 1 / 3) < 0.01


# ─── Annotation quality ─────────────────────────────────────────


class TestAnnotationQuality:
    """Component C: Annotation Quality (category similarity)."""

    def test_perfect_quality(self, grader):
        result = grader.grade(
            {"log_008": "error", "log_023": "error", "log_041": "error"},
            [], "MEDIUM", "Report",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        score = result["components"]["annotation_quality"]["score"]
        assert score >= 0.99

    def test_related_category_gives_partial_quality(self, grader):
        result = grader.grade(
            {"log_008": "symptom"},
            [], "MEDIUM", "Report",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        quality = result["components"]["annotation_quality"]["score"]
        _assert_open01(quality)

    def test_no_correct_annotations_zero_quality(self, grader):
        result = grader.grade(
            {"log_001": "error"},
            [], "MEDIUM", "Report",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        score = result["components"]["annotation_quality"]["score"]
        assert score <= 0.01
        _assert_open01(score)


# ─── Correlation precision & recall ──────────────────────────────


class TestCorrelations:
    """Components D & E: Correlation Precision and Recall."""

    def test_perfect_correlations(self, grader):
        gt_corrs = TASKS["task_medium"]["ground_truth"]["correlations"]
        result = grader.grade(
            {"log_045": "root_cause", "log_067": "symptom"},
            gt_corrs, "HIGH", "Report",
            _medium_gt(), _medium_config(), _default_behavior()
        )
        assert result["components"]["correlation_precision"]["score"] >= 0.99
        assert result["components"]["correlation_recall"]["score"] >= 0.99

    def test_wrong_correlations(self, grader):
        result = grader.grade(
            {}, [["log_001", "log_002"]], "HIGH", "Report",
            _medium_gt(), _medium_config(), _default_behavior()
        )
        score = result["components"]["correlation_precision"]["score"]
        assert score <= 0.01
        _assert_open01(score)

    def test_no_correlations_with_gt(self, grader):
        result = grader.grade(
            {}, [], "HIGH", "Report",
            _medium_gt(), _medium_config(), _default_behavior()
        )
        score = result["components"]["correlation_recall"]["score"]
        assert score <= 0.01
        _assert_open01(score)

    def test_transitive_correlation_gives_partial_credit(self, grader):
        result = grader.grade(
            {},
            [["log_045", "log_134"]],
            "HIGH", "Report",
            _medium_gt(), _medium_config(), _default_behavior()
        )
        recall = result["components"]["correlation_recall"]["score"]
        _assert_open01(recall)
        assert recall > 0.001

    def test_easy_task_no_correlations_needed(self, grader):
        result = grader.grade(
            {}, [], "MEDIUM", "Report",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        assert result["components"]["correlation_precision"]["score"] >= 0.99
        assert result["components"]["correlation_recall"]["score"] >= 0.99


# ─── Chain reconstruction ───────────────────────────────────────


class TestChainReconstruction:
    """Component F: Chain Reconstruction."""

    def test_no_gt_chains_gives_perfect(self, grader):
        result = grader.grade(
            {}, [], "MEDIUM", "Report",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        assert result["components"]["chain_reconstruction"]["score"] >= 0.99

    def test_perfect_chain(self, grader):
        gt_corrs = TASKS["task_medium"]["ground_truth"]["correlations"]
        result = grader.grade(
            {}, gt_corrs, "HIGH", "Report",
            _medium_gt(), _medium_config(), _default_behavior()
        )
        assert result["components"]["chain_reconstruction"]["score"] > 0.8

    def test_reversed_direction_partial_credit(self, grader):
        result = grader.grade(
            {}, [["log_067", "log_045"]], "HIGH", "Report",
            _medium_gt(), _medium_config(), _default_behavior()
        )
        chain = result["components"]["chain_reconstruction"]["score"]
        _assert_open01(chain)


# ─── Severity classification ────────────────────────────────────


class TestSeverityClassification:
    """Component G: Severity Classification."""

    def test_exact_match(self, grader):
        result = grader.grade(
            {}, [], "MEDIUM", "Report",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        assert result["components"]["severity_classification"]["score"] >= 0.99

    def test_off_by_one(self, grader):
        result = grader.grade(
            {}, [], "HIGH", "Report",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        assert result["components"]["severity_classification"]["score"] == 0.5

    def test_off_by_two(self, grader):
        result = grader.grade(
            {}, [], "CRITICAL", "Report",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        assert result["components"]["severity_classification"]["score"] == 0.15

    def test_off_by_three(self, grader):
        result = grader.grade(
            {}, [], "LOW", "Report",
            _hard_gt(), _hard_config(), _default_behavior()
        )
        score = result["components"]["severity_classification"]["score"]
        assert score <= 0.01
        _assert_open01(score)

    def test_empty_severity(self, grader):
        result = grader.grade(
            {}, [], "", "Report",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        score = result["components"]["severity_classification"]["score"]
        assert score <= 0.01
        _assert_open01(score)

    def test_case_insensitive_severity(self, grader):
        result = grader.grade(
            {}, [], "medium", "Report",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        assert result["components"]["severity_classification"]["score"] >= 0.99


# ─── Report completeness (word-overlap fallback) ────────────────


class TestReportCompleteness:
    """Component H: Report Completeness (word-overlap fallback path)."""

    def test_perfect_report(self, grader):
        findings = TASKS["task_easy"]["ground_truth"]["key_findings"]
        report = ", ".join(findings) + ". Detailed analysis."
        result = grader.grade(
            {}, [], "MEDIUM", report,
            _easy_gt(), _easy_config(), _default_behavior()
        )
        completeness = result["components"]["report_completeness"]["score"]
        assert completeness > 0.5

    def test_empty_report(self, grader):
        result = grader.grade(
            {}, [], "MEDIUM", "",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        score = result["components"]["report_completeness"]["score"]
        assert score <= 0.01
        _assert_open01(score)

    def test_draft_report_penalized(self, grader):
        report = "database connection refused auth-service affected postgresql port 5432"
        submitted = grader.grade(
            {}, [], "MEDIUM", report,
            _easy_gt(), _easy_config(), _default_behavior(report_source="submitted")
        )
        draft = grader.grade(
            {}, [], "MEDIUM", report,
            _easy_gt(), _easy_config(), _default_behavior(report_source="draft")
        )
        assert draft["components"]["report_completeness"]["score"] <= \
               submitted["components"]["report_completeness"]["score"]


# ─── Report coherence ───────────────────────────────────────────


class TestReportCoherence:
    """Component I: Report Coherence."""

    def test_empty_report_zero_coherence(self, grader):
        result = grader.grade(
            {}, [], "MEDIUM", "",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        score = result["components"]["report_coherence"]["score"]
        assert score <= 0.01
        _assert_open01(score)

    def test_short_report_low_coherence(self, grader):
        result = grader.grade(
            {}, [], "MEDIUM", "Bad.",
            _easy_gt(), _easy_config(), _default_behavior()
        )
        score = result["components"]["report_coherence"]["score"]
        assert score <= 0.01
        _assert_open01(score)

    def test_well_structured_report(self, grader):
        report = (
            "Root cause: The auth-service experienced database connection failures. "
            "First, the connection pool became exhausted on port 5432. "
            "Then, auth-service began returning errors. "
            "This caused service degradation. "
            "Subsequently, all user authentication requests failed. "
            "Impact: auth-service was the sole affected service. "
            "Severity: MEDIUM. "
            "Timeline: The issue began at 09:15 and was first detected in log_008. "
            "Due to the connection pool exhaustion, the circuit breaker activated in log_023. "
            "Finally, log_041 shows complete pool exhaustion with 12 requests queued."
        )
        result = grader.grade(
            {}, [], "MEDIUM", report,
            _easy_gt(), _easy_config(), _default_behavior()
        )
        coherence = result["components"]["report_coherence"]["score"]
        assert coherence > 0.5
        _assert_open01(coherence)

    def test_draft_coherence_penalized(self, grader):
        report = (
            "Root cause: database connection pool exhaustion. "
            "First, connections were refused. Then services degraded. "
            "Impact: auth-service. Severity: MEDIUM."
        )
        submitted = grader.grade(
            {}, [], "MEDIUM", report,
            _easy_gt(), _easy_config(), _default_behavior(report_source="submitted")
        )
        draft = grader.grade(
            {}, [], "MEDIUM", report,
            _easy_gt(), _easy_config(), _default_behavior(report_source="draft")
        )
        assert draft["components"]["report_coherence"]["score"] <= \
               submitted["components"]["report_coherence"]["score"]


# ─── Investigation efficiency ────────────────────────────────────


class TestInvestigationEfficiency:
    """Component J: Investigation Efficiency."""

    def test_fast_high_quality_gives_best_efficiency(self, grader):
        gt = _easy_gt()
        ann = {lid: cat for lid, cat in gt["annotations"].items()}
        result = grader.grade(
            ann, [], "MEDIUM",
            "database connection refused auth-service affected postgresql port 5432",
            gt, _easy_config(),
            _default_behavior(steps=5)
        )
        eff = result["components"]["investigation_efficiency"]["score"]
        assert eff >= 0.5
        _assert_open01(eff)

    def test_max_steps_poor_quality_low_efficiency(self, grader):
        result = grader.grade(
            {}, [], "", "",
            _easy_gt(), _easy_config(),
            _default_behavior(steps=15)
        )
        eff = result["components"]["investigation_efficiency"]["score"]
        assert eff < 0.1
        _assert_open01(eff)


# ─── Perfect run scoring ────────────────────────────────────────


class TestPerfectRun:
    """Test that a perfect run achieves a very high score."""

    def test_perfect_easy_run(self, grader):
        gt = _easy_gt()
        report = (
            "Root cause: database connection refused on auth-service. "
            "The auth-service affected users connecting to postgresql port 5432. "
            "First, connection was refused in log_008. Then, circuit breaker opened in log_023. "
            "Finally, pool exhausted in log_041 with 12 requests queued. "
            "Impact: auth-service was the sole affected service. "
            "Severity: MEDIUM. Due to service degradation, user authentication was impacted."
        )
        result = grader.grade(
            gt["annotations"], gt["correlations"], gt["severity"],
            report, gt, _easy_config(), _default_behavior(steps=8)
        )
        _assert_open01(result["final_score"])
        assert result["final_score"] > 0.6

    def test_perfect_medium_run(self, grader):
        gt = _medium_gt()
        report = (
            "Root cause: payment service database pool exhausted. "
            "The order service queue backup reached 15000 messages. "
            "Api gateway 503 errors were returned to clients. "
            "This was a cascading failure from payment to gateway. "
            "First, payment-service connections exhausted in log_045. "
            "Then, order-service timed out in log_102. "
            "Subsequently, api-gateway returned HTTP 503 errors in log_156. "
            "Impact: all three services affected. Severity: HIGH."
        )
        result = grader.grade(
            gt["annotations"], gt["correlations"], gt["severity"],
            report, gt, _medium_config(), _default_behavior(steps=15)
        )
        _assert_open01(result["final_score"])
        assert result["final_score"] > 0.6


# ─── Clamp function tests ───────────────────────────────────────


class TestClamp01:
    """Direct tests for the _clamp01 utility."""

    def test_clamp_zero(self):
        assert _clamp01(0.0) == _EPS

    def test_clamp_one(self):
        assert _clamp01(1.0) == 1.0 - _EPS

    def test_clamp_negative(self):
        assert _clamp01(-5.0) == _EPS

    def test_clamp_above_one(self):
        assert _clamp01(2.5) == 1.0 - _EPS

    def test_clamp_mid(self):
        assert _clamp01(0.5) == 0.5

    def test_clamp_none(self):
        assert _clamp01(None) == 0.5

    def test_clamp_nan(self):
        assert _clamp01(float('nan')) == 0.5

    def test_clamp_inf(self):
        assert _clamp01(float('inf')) == 0.5
