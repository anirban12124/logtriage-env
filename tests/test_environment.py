"""
Tests for src/environment.py — the core LogTriageEnv environment.
"""

import pytest
from src.environment import LogTriageEnv
from src.tasks import TASKS


# ─── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def easy_env():
    """A fresh environment reset on task_easy."""
    env = LogTriageEnv()
    env.reset("task_easy")
    return env


@pytest.fixture
def medium_env():
    """A fresh environment reset on task_medium."""
    env = LogTriageEnv()
    env.reset("task_medium")
    return env


@pytest.fixture
def hard_env():
    """A fresh environment reset on task_hard."""
    env = LogTriageEnv()
    env.reset("task_hard")
    return env


# ─── Reset ───────────────────────────────────────────────────────


class TestReset:
    """Verify environment reset behavior."""

    @pytest.mark.parametrize("task_id", ["task_easy", "task_medium", "task_hard"])
    def test_reset_returns_observation(self, task_id):
        env = LogTriageEnv()
        result = env.reset(task_id)
        assert "observation" in result
        assert "info" in result

    @pytest.mark.parametrize("task_id", ["task_easy", "task_medium", "task_hard"])
    def test_reset_observation_fields(self, task_id):
        env = LogTriageEnv()
        result = env.reset(task_id)
        obs = result["observation"]
        assert obs["task_id"] == task_id
        assert obs["step_number"] == 0
        assert obs["max_steps"] == TASKS[task_id]["max_steps"]
        assert obs["annotations_count"] == 0
        assert obs["correlations_count"] == 0
        assert obs["severity_classified"] is None
        assert obs["current_report_draft"] is None

    def test_reset_info_contains_task_name(self):
        env = LogTriageEnv()
        result = env.reset("task_easy")
        assert result["info"]["task_name"] == "Database Connection Failures"

    def test_reset_invalid_task_raises(self):
        env = LogTriageEnv()
        with pytest.raises(ValueError, match="Unknown task"):
            env.reset("task_nonexistent")

    def test_reset_clears_previous_state(self):
        env = LogTriageEnv()
        env.reset("task_easy")
        env.step("annotate", {"log_id": "log_008", "category": "error"})
        # Reset should clear everything
        result = env.reset("task_easy")
        obs = result["observation"]
        assert obs["annotations_count"] == 0
        assert obs["step_number"] == 0

    def test_visible_logs_on_reset(self):
        env = LogTriageEnv()
        result = env.reset("task_easy")
        obs = result["observation"]
        assert len(obs["visible_logs"]) <= 20  # page_size
        assert obs["current_page"] == 0


# ─── Step basics ─────────────────────────────────────────────────


class TestStepBasics:
    """Basic step mechanics."""

    def test_step_increments_count(self, easy_env):
        result = easy_env.step("noop")
        assert result["observation"]["step_number"] == 1

    def test_step_returns_correct_structure(self, easy_env):
        result = easy_env.step("noop")
        assert "observation" in result
        assert "reward" in result
        assert "done" in result
        assert "info" in result

    def test_step_before_reset_raises(self):
        env = LogTriageEnv()
        with pytest.raises(ValueError, match="not initialized"):
            env.step("noop")

    def test_unknown_action_type(self, easy_env):
        result = easy_env.step("fly_to_moon")
        assert result["observation"]["last_action_success"] is False
        assert "Unknown action type" in result["observation"]["last_action_message"]

    def test_done_after_max_steps(self, easy_env):
        """Episode should end after max_steps."""
        max_steps = TASKS["task_easy"]["max_steps"]
        for i in range(max_steps):
            result = easy_env.step("noop")
        assert result["done"] is True

    def test_step_after_done_returns_zero_reward(self, easy_env):
        """Steps after done should return 0 reward."""
        max_steps = TASKS["task_easy"]["max_steps"]
        for _ in range(max_steps):
            easy_env.step("noop")
        result = easy_env.step("noop")
        assert result["done"] is True
        assert result["reward"]["value"] == 0.0


# ─── Search action ───────────────────────────────────────────────


class TestSearch:
    """Test BM25 search functionality."""

    def test_search_filters_logs(self, easy_env):
        result = easy_env.step("search", {"pattern": "connection"})
        obs = result["observation"]
        assert obs["total_log_count"] < 50  # should narrow results
        assert obs["current_page"] == 0  # resets to page 0

    def test_search_finds_gt_logs(self, easy_env):
        """Searching for 'connection' should surface auth-service DB errors."""
        result = easy_env.step("search", {"pattern": "connection refused"})
        obs = result["observation"]
        visible_ids = {log["id"] for log in obs["visible_logs"]}
        # At least one GT log should appear
        gt_ids = set(TASKS["task_easy"]["ground_truth"]["annotations"].keys())
        assert visible_ids & gt_ids, "Search should surface ground-truth logs"

    def test_empty_search_clears(self, easy_env):
        easy_env.step("search", {"pattern": "connection"})
        result = easy_env.step("search", {"pattern": ""})
        obs = result["observation"]
        assert obs["total_log_count"] == 50  # all logs restored

    def test_search_message(self, easy_env):
        result = easy_env.step("search", {"pattern": "database"})
        obs = result["observation"]
        assert "Search" in obs["last_action_message"] or "search" in obs["last_action_message"].lower()


# ─── Filter actions ──────────────────────────────────────────────


class TestFilters:
    """Test severity, service, and time range filters."""

    def test_filter_severity(self, easy_env):
        result = easy_env.step("filter_severity", {"level": "ERROR"})
        obs = result["observation"]
        for log in obs["visible_logs"]:
            assert log["severity"] == "ERROR"

    def test_filter_invalid_severity(self, easy_env):
        result = easy_env.step("filter_severity", {"level": "UNKNOWN"})
        assert result["observation"]["last_action_success"] is False

    def test_filter_service(self, medium_env):
        result = medium_env.step("filter_service", {"service": "payment-service"})
        obs = result["observation"]
        for log in obs["visible_logs"]:
            assert log["service"] == "payment-service"

    def test_filter_invalid_service(self, easy_env):
        result = easy_env.step("filter_service", {"service": "nonexistent-svc"})
        assert result["observation"]["last_action_success"] is False

    def test_filter_time_range_requires_both(self, easy_env):
        result = easy_env.step("filter_time_range", {"start": "2024-01-15T09:00:00.000Z"})
        assert result["observation"]["last_action_success"] is False

    def test_clear_filters(self, easy_env):
        easy_env.step("filter_severity", {"level": "ERROR"})
        result = easy_env.step("clear_filters")
        obs = result["observation"]
        assert obs["current_filters"] == {}
        assert obs["total_log_count"] == 50

    def test_combined_filters(self, medium_env):
        """Applying severity + service filter should narrow results."""
        medium_env.step("filter_severity", {"level": "ERROR"})
        result = medium_env.step("filter_service", {"service": "payment-service"})
        obs = result["observation"]
        for log in obs["visible_logs"]:
            assert log["severity"] == "ERROR"
            assert log["service"] == "payment-service"


# ─── Scroll action ───────────────────────────────────────────────


class TestScroll:
    """Test pagination via scroll."""

    def test_scroll_down(self, medium_env):
        result = medium_env.step("scroll", {"direction": "down"})
        assert result["observation"]["current_page"] == 1

    def test_scroll_up_at_top(self, easy_env):
        """Scrolling up at page 0 should stay at page 0."""
        result = easy_env.step("scroll", {"direction": "up"})
        assert result["observation"]["current_page"] == 0

    def test_scroll_down_at_bottom(self, easy_env):
        """Scrolling past the last page should clamp."""
        total_pages = easy_env._build_observation().total_pages
        for _ in range(total_pages + 5):
            result = easy_env.step("scroll", {"direction": "down"})
        assert result["observation"]["current_page"] == total_pages - 1


# ─── Inspect action ─────────────────────────────────────────────


class TestInspect:
    """Test log inspection."""

    def test_inspect_returns_full_log(self, easy_env):
        result = easy_env.step("inspect", {"log_id": "log_008"})
        obs = result["observation"]
        assert obs["inspected_log"] is not None
        assert obs["inspected_log"]["id"] == "log_008"
        # Inspected log should have full message (not truncated)
        assert len(obs["inspected_log"]["message"]) > 0

    def test_inspect_invalid_id(self, easy_env):
        result = easy_env.step("inspect", {"log_id": "log_999"})
        assert result["observation"]["last_action_success"] is False
        assert result["observation"]["inspected_log"] is None

    def test_inspect_clears_on_next_step(self, easy_env):
        easy_env.step("inspect", {"log_id": "log_008"})
        result = easy_env.step("noop")
        assert result["observation"]["inspected_log"] is None


# ─── Annotate action ────────────────────────────────────────────


class TestAnnotate:
    """Test log annotation."""

    def test_annotate_increments_count(self, easy_env):
        result = easy_env.step("annotate", {"log_id": "log_008", "category": "error"})
        assert result["observation"]["annotations_count"] == 1

    def test_annotate_shows_in_recent(self, easy_env):
        result = easy_env.step("annotate", {"log_id": "log_008", "category": "error"})
        recent = result["observation"]["recent_annotations"]
        assert any(a["log_id"] == "log_008" for a in recent)

    def test_annotate_missing_params(self, easy_env):
        result = easy_env.step("annotate", {"log_id": "log_008"})
        assert result["observation"]["last_action_success"] is False

    def test_annotate_invalid_log_id(self, easy_env):
        result = easy_env.step("annotate", {"log_id": "log_999", "category": "error"})
        assert result["observation"]["last_action_success"] is False

    def test_annotate_update_category(self, easy_env):
        """Re-annotating should update the category."""
        easy_env.step("annotate", {"log_id": "log_008", "category": "error"})
        result = easy_env.step("annotate", {"log_id": "log_008", "category": "root_cause"})
        # Count should still be 1 (updated, not added)
        assert result["observation"]["annotations_count"] == 1
        assert "Updated" in result["observation"]["last_action_message"]

    def test_annotations_by_category(self, easy_env):
        easy_env.step("annotate", {"log_id": "log_008", "category": "error"})
        result = easy_env.step("annotate", {"log_id": "log_023", "category": "error"})
        by_cat = result["observation"]["annotations_by_category"]
        assert by_cat.get("error") == 2


# ─── Correlate action ───────────────────────────────────────────


class TestCorrelate:
    """Test event correlation."""

    def test_correlate_increments_count(self, medium_env):
        result = medium_env.step("correlate", {
            "source_log_id": "log_045",
            "target_log_id": "log_067",
        })
        assert result["observation"]["correlations_count"] == 1

    def test_correlate_shows_in_recent(self, medium_env):
        result = medium_env.step("correlate", {
            "source_log_id": "log_045",
            "target_log_id": "log_067",
        })
        recent = result["observation"]["recent_correlations"]
        assert ["log_045", "log_067"] in recent

    def test_correlate_self_fails(self, easy_env):
        result = easy_env.step("correlate", {
            "source_log_id": "log_008",
            "target_log_id": "log_008",
        })
        assert result["observation"]["last_action_success"] is False

    def test_correlate_missing_source(self, easy_env):
        result = easy_env.step("correlate", {
            "source_log_id": "",
            "target_log_id": "log_008",
        })
        assert result["observation"]["last_action_success"] is False

    def test_correlate_invalid_source(self, easy_env):
        result = easy_env.step("correlate", {
            "source_log_id": "log_999",
            "target_log_id": "log_008",
        })
        assert result["observation"]["last_action_success"] is False

    def test_correlate_invalid_target(self, easy_env):
        result = easy_env.step("correlate", {
            "source_log_id": "log_008",
            "target_log_id": "log_999",
        })
        assert result["observation"]["last_action_success"] is False


# ─── Classify incident ──────────────────────────────────────────


class TestClassify:
    """Test severity classification."""

    @pytest.mark.parametrize("severity", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    def test_valid_severity(self, easy_env, severity):
        result = easy_env.step("classify_incident", {"severity": severity})
        assert result["observation"]["severity_classified"] == severity

    def test_invalid_severity(self, easy_env):
        result = easy_env.step("classify_incident", {"severity": "BANANA"})
        assert result["observation"]["last_action_success"] is False
        assert result["observation"]["severity_classified"] is None

    def test_case_insensitive(self, easy_env):
        result = easy_env.step("classify_incident", {"severity": "high"})
        assert result["observation"]["severity_classified"] == "HIGH"


# ─── Draft and submit report ────────────────────────────────────


class TestReport:
    """Test draft/submit report actions."""

    def test_draft_saves_report(self, easy_env):
        result = easy_env.step("draft_report", {"summary": "Test report"})
        obs = result["observation"]
        assert obs["current_report_draft"] == "Test report"
        assert obs["draft_feedback"] is not None
        assert obs["last_action_success"] is True

    def test_draft_feedback_format(self, easy_env):
        result = easy_env.step("draft_report", {"summary": "database connection refused"})
        feedback = result["observation"]["draft_feedback"]
        assert "key areas" in feedback or "Report covers" in feedback

    def test_submit_too_early_converts_to_draft(self, easy_env):
        """Submit before step 3 should auto-convert to draft."""
        result = easy_env.step("submit_report", {"summary": "Early report"})
        assert result["done"] is False
        assert "Too early" in result["observation"]["last_action_message"]

    def test_submit_after_step_3_ends_episode(self, easy_env):
        """Submit at step >= 3 should end the episode."""
        easy_env.step("noop")
        easy_env.step("noop")
        result = easy_env.step("submit_report", {"summary": "Final report"})
        assert result["done"] is True

    def test_submit_returns_grader_result(self, easy_env):
        """Submission should trigger grading."""
        easy_env.step("annotate", {"log_id": "log_008", "category": "error"})
        easy_env.step("classify_incident", {"severity": "MEDIUM"})
        result = easy_env.step("submit_report", {
            "summary": "database connection refused in auth-service on postgresql port 5432"
        })
        assert result["done"] is True
        assert "grader_result" in result["info"]
        grader = result["info"]["grader_result"]
        assert "final_score" in grader
        assert grader["final_score"] >= 0.0


# ─── Message truncation ─────────────────────────────────────────


class TestObservationTruncation:
    """Verify observation building truncates long messages."""

    def test_visible_logs_truncated_to_200(self, easy_env):
        obs = easy_env._build_observation()
        for log in obs.visible_logs:
            assert len(log.message) <= 200

    def test_inspected_log_not_truncated(self, easy_env):
        """Inspected log should show full message."""
        easy_env.step("inspect", {"log_id": "log_041"})
        obs = easy_env._build_observation()
        if obs.inspected_log:
            # GT log_041 has a long message — should NOT be truncated
            original = next(l for l in easy_env.logs if l["id"] == "log_041")
            assert obs.inspected_log.message == original["message"]


# ─── State endpoint ──────────────────────────────────────────────


class TestState:
    """Test the state() method."""

    def test_state_returns_correct_fields(self, easy_env):
        state = easy_env.state()
        assert state["task_id"] == "task_easy"
        assert state["step"] == 0
        assert state["done"] is False
        assert state["annotations"] == {}
        assert state["correlations"] == []
        assert state["severity"] == ""
        assert state["report"] == ""

    def test_state_reflects_actions(self, easy_env):
        easy_env.step("annotate", {"log_id": "log_008", "category": "error"})
        easy_env.step("classify_incident", {"severity": "HIGH"})
        state = easy_env.state()
        assert state["annotations"]["log_008"] == "error"
        assert state["severity"] == "HIGH"
        assert state["step"] == 2
