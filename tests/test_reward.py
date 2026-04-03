"""
Tests for src/reward.py — dense per-step reward calculation.
"""

import pytest
from src.reward import RewardCalculator
from src.tasks import TASKS


# ─── Fixtures ────────────────────────────────────────────────────


def _make_calc(task_id: str = "task_easy") -> RewardCalculator:
    """Create a RewardCalculator for the given task."""
    task = TASKS[task_id]
    return RewardCalculator(task["ground_truth"], task)


def _easy_calc():
    return _make_calc("task_easy")


def _medium_calc():
    return _make_calc("task_medium")


# ─── Annotation rewards ─────────────────────────────────────────


class TestAnnotationReward:
    """Test rewards for the annotate action."""

    def test_correct_annotation_gives_positive_reward(self):
        calc = _easy_calc()
        # Track log as viewed first
        calc.track_page_view(0, ["log_008"], ["auth-service"])
        result = calc.calculate("annotate", {"log_id": "log_008", "category": "error"}, 1, {})
        assert result["value"] > 0
        assert "correct_annotation" in result["components"]

    def test_wrong_annotation_gives_penalty(self):
        calc = _easy_calc()
        result = calc.calculate("annotate", {"log_id": "log_001", "category": "error"}, 1, {})
        assert result["value"] < 0
        assert "wrong_annotation" in result["components"]

    def test_duplicate_annotation_gives_penalty(self):
        calc = _easy_calc()
        calc.track_page_view(0, ["log_008"], ["auth-service"])
        calc.calculate("annotate", {"log_id": "log_008", "category": "error"}, 1, {})
        result = calc.calculate("annotate", {"log_id": "log_008", "category": "error"}, 2, {})
        assert result["value"] < 0
        assert "duplicate_annotation" in result["components"]

    def test_early_annotation_gets_temporal_bonus(self):
        """Annotations early in the episode should get higher reward."""
        calc1 = _easy_calc()
        calc1.track_page_view(0, ["log_008"], ["auth-service"])
        early = calc1.calculate("annotate", {"log_id": "log_008", "category": "error"}, 1, {})

        calc2 = _easy_calc()
        calc2.track_page_view(0, ["log_008"], ["auth-service"])
        late = calc2.calculate("annotate", {"log_id": "log_008", "category": "error"}, 14, {})

        assert early["value"] > late["value"]

    def test_informed_bonus_for_viewed_log(self):
        """Annotating a log the agent has actually seen gives more reward."""
        calc1 = _easy_calc()
        calc1.track_page_view(0, ["log_008"], ["auth-service"])
        viewed = calc1.calculate("annotate", {"log_id": "log_008", "category": "error"}, 1, {})

        calc2 = _easy_calc()
        # Don't track any page views
        not_viewed = calc2.calculate("annotate", {"log_id": "log_008", "category": "error"}, 1, {})

        assert viewed["value"] > not_viewed["value"]

    def test_wrong_annotation_penalty_escalates(self):
        """Repeated wrong annotations should get harsher penalties."""
        calc = _easy_calc()
        # Submit many wrong annotations to drop precision
        for i in range(1, 6):
            if f"log_{i:03d}" not in TASKS["task_easy"]["ground_truth"]["annotations"]:
                calc.calculate(
                    "annotate", {"log_id": f"log_{i:03d}", "category": "error"}, i, {}
                )
        # Precision should be very low now — next wrong should get steep penalty
        result = calc.calculate(
            "annotate", {"log_id": "log_006", "category": "error"}, 6, {}
        )
        assert result["components"]["wrong_annotation"] <= -0.10

    def test_partial_category_credit(self):
        """Annotating with a related (but not exact) category should get partial credit."""
        calc = _easy_calc()
        calc.track_page_view(0, ["log_008"], ["auth-service"])
        # GT is "error", annotating as "symptom" should still get some credit
        result = calc.calculate(
            "annotate", {"log_id": "log_008", "category": "symptom"}, 1, {}
        )
        assert result["value"] > 0
        # But less than exact match
        calc2 = _easy_calc()
        calc2.track_page_view(0, ["log_008"], ["auth-service"])
        exact = calc2.calculate(
            "annotate", {"log_id": "log_008", "category": "error"}, 1, {}
        )
        assert exact["value"] > result["value"]


# ─── Milestone bonuses ───────────────────────────────────────────


class TestMilestones:
    """Test coverage milestone bonuses."""

    def test_milestone_25_percent(self):
        """Finding 25% of GT annotations should give a milestone bonus."""
        calc = _easy_calc()
        calc.track_page_view(0, ["log_008"], ["auth-service"])
        result = calc.calculate(
            "annotate", {"log_id": "log_008", "category": "error"}, 1, {}
        )
        # 1 of 3 = 33% — should trigger 25% milestone
        assert "milestone_bonus" in result["components"]

    def test_milestone_100_percent(self):
        """Finding all GT annotations should give the largest milestone bonus."""
        calc = _easy_calc()
        gt_ids = list(TASKS["task_easy"]["ground_truth"]["annotations"].keys())
        calc.track_page_view(0, gt_ids, ["auth-service"])

        last_result = None
        for i, lid in enumerate(gt_ids):
            last_result = calc.calculate(
                "annotate", {"log_id": lid, "category": "error"}, i + 1, {}
            )
        # 100% milestone should have been triggered
        assert calc._milestone_100 is True


# ─── Correlation rewards ────────────────────────────────────────


class TestCorrelationReward:
    """Test rewards for the correlate action."""

    def test_correct_correlation_gives_positive_reward(self):
        calc = _medium_calc()
        result = calc.calculate(
            "correlate",
            {"source_log_id": "log_045", "target_log_id": "log_067"},
            5, {}
        )
        assert result["value"] > 0
        assert "correct_correlation" in result["components"]

    def test_wrong_correlation_gives_penalty(self):
        calc = _easy_calc()
        result = calc.calculate(
            "correlate",
            {"source_log_id": "log_001", "target_log_id": "log_002"},
            1, {}
        )
        assert result["value"] < 0
        assert "wrong_correlation" in result["components"]

    def test_duplicate_correlation_gives_penalty(self):
        calc = _medium_calc()
        calc.calculate(
            "correlate",
            {"source_log_id": "log_045", "target_log_id": "log_067"},
            5, {}
        )
        result = calc.calculate(
            "correlate",
            {"source_log_id": "log_045", "target_log_id": "log_067"},
            6, {}
        )
        assert "duplicate_correlation" in result["components"]

    def test_chain_bonus_for_long_chains(self):
        """Building a long chain of correlations should give chain bonuses."""
        calc = _make_calc("task_hard")
        gt_corrs = TASKS["task_hard"]["ground_truth"]["correlations"]
        chain_bonus_seen = False
        for i, (src, tgt) in enumerate(gt_corrs):
            result = calc.calculate(
                "correlate",
                {"source_log_id": src, "target_log_id": tgt},
                i + 1, {}
            )
            if "chain_bonus" in result["components"]:
                chain_bonus_seen = True
        assert chain_bonus_seen, "Long chains should produce chain bonuses"


# ─── Classification rewards ─────────────────────────────────────


class TestClassificationReward:
    """Test rewards for classify_incident."""

    def test_exact_severity_match(self):
        calc = _easy_calc()
        result = calc.calculate("classify_incident", {"severity": "MEDIUM"}, 5, {})
        assert result["components"]["correct_severity"] == 0.20

    def test_off_by_one_severity(self):
        calc = _easy_calc()
        result = calc.calculate("classify_incident", {"severity": "HIGH"}, 5, {})
        assert result["components"]["close_severity"] == 0.05

    def test_off_by_two_severity(self):
        calc = _easy_calc()
        result = calc.calculate("classify_incident", {"severity": "CRITICAL"}, 5, {})
        assert result["components"]["wrong_severity"] == -0.10

    def test_invalid_severity_string(self):
        calc = _easy_calc()
        result = calc.calculate("classify_incident", {"severity": "BANANA"}, 5, {})
        assert result["components"]["invalid_severity"] == -0.05


# ─── Search rewards ──────────────────────────────────────────────


class TestSearchReward:
    """Test rewards for search actions."""

    def test_useful_search_gives_reward(self):
        calc = _easy_calc()
        env_state = {"search_hits": 10, "relevant_hits": 3}
        result = calc.calculate("search", {"pattern": "connection"}, 1, env_state)
        assert result["value"] > 0
        assert "useful_search" in result["components"]

    def test_useless_search_gives_penalty(self):
        calc = _easy_calc()
        env_state = {"search_hits": 0, "relevant_hits": 0}
        result = calc.calculate("search", {"pattern": "xyznonexistent"}, 1, env_state)
        assert result["value"] < 0
        assert "useless_search" in result["components"]

    def test_search_tracks_first_search_step(self):
        calc = _easy_calc()
        assert calc.first_search_step is None
        calc.calculate("search", {"pattern": "error"}, 3, {"search_hits": 5, "relevant_hits": 1})
        assert calc.first_search_step == 3


# ─── Report rewards ─────────────────────────────────────────────


class TestReportReward:
    """Test rewards for draft/submit report."""

    def test_first_draft_gives_reward(self):
        calc = _easy_calc()
        result = calc.calculate(
            "draft_report",
            {"summary": "database connection refused auth-service affected"},
            5, {}
        )
        assert result["value"] >= 0
        assert "draft_reward" in result["components"]

    def test_submit_with_good_coverage(self):
        calc = _easy_calc()
        summary = (
            "Root cause: database connection refused on auth-service. "
            "The auth-service affected users. Connection to postgresql port 5432 failed."
        )
        result = calc.calculate("submit_report", {"summary": summary}, 10, {})
        assert result["value"] > 0
        assert "report_reward" in result["components"]

    def test_draft_improvement_reward(self):
        calc = _easy_calc()
        # First draft
        calc.calculate("draft_report", {"summary": "database issue"}, 5, {})
        # Improved draft
        result = calc.calculate(
            "draft_report",
            {"summary": "database connection refused auth-service affected postgresql port 5432"},
            7,
            {"prev_draft_matches": 1},
        )
        assert result["value"] >= 0


# ─── Noop and navigation ────────────────────────────────────────


class TestNoopAndNavigation:
    """Test noop penalty and navigation rewards."""

    def test_noop_penalty(self):
        calc = _easy_calc()
        result = calc.calculate("noop", {}, 1, {})
        assert result["value"] < 0
        assert result["components"]["noop_penalty"] == -0.02

    def test_navigation_with_improved_density(self):
        calc = _easy_calc()
        env_state = {"old_density": 0.1, "new_density": 0.3}
        result = calc.calculate("filter_severity", {"level": "ERROR"}, 1, env_state)
        if "navigation_reward" in result["components"]:
            assert result["components"]["navigation_reward"] > 0


# ─── Anti-spam: repeat penalty ───────────────────────────────────


class TestRepeatPenalty:
    """Test that repeating the same action gets penalized."""

    def test_repeat_penalty(self):
        calc = _easy_calc()
        calc.calculate("noop", {}, 1, {})
        calc.calculate("noop", {}, 2, {})
        result = calc.calculate("noop", {}, 3, {})
        assert "repeat_penalty" in result["components"]
        assert result["components"]["repeat_penalty"] == -0.03


# ─── Reward capping ─────────────────────────────────────────────


class TestRewardCapping:
    """Test reward value capping."""

    def test_per_step_cap(self):
        calc = _easy_calc()
        # Even the best action should not exceed 0.40
        result = calc.calculate("noop", {}, 1, {})
        assert -0.15 <= result["value"] <= 0.40

    def test_cumulative_cap(self):
        calc = _easy_calc()
        # Run many rewarding actions
        for i in range(100):
            calc.calculate("noop", {}, i + 1, {})
        assert -1.0 <= calc.cumulative <= 3.0


# ─── Strategy bonus ─────────────────────────────────────────────


class TestStrategyBonus:
    """Test end-of-episode strategy bonuses."""

    def test_explored_first_bonus(self):
        """Searching before annotating should give a strategy bonus."""
        calc = _easy_calc()
        calc.calculate("search", {"pattern": "error"}, 1, {"search_hits": 5, "relevant_hits": 1})
        calc.track_page_view(0, ["log_008"], ["auth-service"])
        calc.calculate("annotate", {"log_id": "log_008", "category": "error"}, 2, {})
        # Submit to trigger strategy bonus
        result = calc.calculate(
            "submit_report", {"summary": "test report"}, 5, {}
        )
        assert "strategy_explored_first" in result["components"]

    def test_efficiency_bonus_for_early_completion(self):
        """Submitting well before max_steps should give efficiency bonus."""
        calc = _easy_calc()
        # Step 5 of 15 = 33% → under 70% threshold
        result = calc.calculate(
            "submit_report", {"summary": "test report"}, 5, {}
        )
        assert "efficiency_bonus" in result["components"]

    def test_multi_service_exploration_bonus(self):
        """Exploring all task services should give a multi-service bonus."""
        calc = _medium_calc()
        calc.services_explored = {"api-gateway", "order-service", "payment-service"}
        result = calc.calculate("submit_report", {"summary": "test"}, 10, {})
        assert "strategy_multi_service" in result["components"]


# ─── Difficulty multiplier ───────────────────────────────────────


class TestDifficultyMultiplier:
    """Test that harder tasks give higher rewards for correct actions."""

    def test_hard_gives_more_than_easy(self):
        calc_easy = _easy_calc()
        calc_hard = _make_calc("task_hard")
        calc_easy.track_page_view(0, ["log_008"], ["auth-service"])
        calc_hard.track_page_view(0, ["log_023"], ["auth-service"])

        easy_result = calc_easy.calculate(
            "annotate", {"log_id": "log_008", "category": "error"}, 1, {}
        )
        hard_result = calc_hard.calculate(
            "annotate", {"log_id": "log_023", "category": "reconnaissance"}, 1, {}
        )
        # Hard has 1.6x multiplier vs easy's 1.0x
        # Compare the specific component so milestone bonuses don't skew the comparison
        assert hard_result["components"]["correct_annotation"] > easy_result["components"]["correct_annotation"]
