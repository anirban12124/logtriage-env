"""
Tests for determinism — verifying reproducibility of log generation
and environment behavior across runs.
"""

import json
import hashlib
import pytest
from src.log_generator import generate_logs
from src.environment import LogTriageEnv
from src.tasks import TASKS


def _hash_logs(logs: list) -> str:
    """SHA-256 hash of serialized logs for comparison."""
    raw = json.dumps(logs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


# ─── Log generation determinism ──────────────────────────────────


class TestLogGenerationDeterminism:
    """Logs must be identical across multiple calls with same task_id."""

    @pytest.mark.parametrize("task_id", ["task_easy", "task_medium", "task_hard"])
    def test_same_logs_across_runs(self, task_id):
        """10 consecutive calls should produce identical logs."""
        baseline = generate_logs(task_id)
        baseline_hash = _hash_logs(baseline)
        for i in range(10):
            logs = generate_logs(task_id)
            assert _hash_logs(logs) == baseline_hash, (
                f"Run {i+2} produced different logs for {task_id}"
            )

    @pytest.mark.parametrize("task_id", ["task_easy", "task_medium", "task_hard"])
    def test_identical_entries(self, task_id):
        """Every individual log entry should match field-by-field."""
        run1 = generate_logs(task_id)
        run2 = generate_logs(task_id)
        assert len(run1) == len(run2)
        for a, b in zip(run1, run2):
            assert a["id"] == b["id"]
            assert a["timestamp"] == b["timestamp"]
            assert a["service"] == b["service"]
            assert a["severity"] == b["severity"]
            assert a["message"] == b["message"]
            assert a.get("metadata", {}) == b.get("metadata", {})

    def test_different_tasks_produce_different_logs(self):
        """Different task_ids should produce different log datasets."""
        h_easy = _hash_logs(generate_logs("task_easy"))
        h_medium = _hash_logs(generate_logs("task_medium"))
        h_hard = _hash_logs(generate_logs("task_hard"))
        assert h_easy != h_medium
        assert h_medium != h_hard
        assert h_easy != h_hard


# ─── Environment determinism ────────────────────────────────────


class TestEnvironmentDeterminism:
    """Same action sequences should produce identical observations."""

    def test_reset_produces_same_observation(self):
        """Two resets of the same task should give identical observations."""
        env1 = LogTriageEnv()
        env2 = LogTriageEnv()
        result1 = env1.reset("task_easy")
        result2 = env2.reset("task_easy")
        assert result1["observation"] == result2["observation"]

    def test_identical_action_sequence(self):
        """Same action sequence should produce identical reward traces."""
        actions = [
            ("filter_severity", {"level": "ERROR"}),
            ("search", {"pattern": "connection"}),
            ("annotate", {"log_id": "log_008", "category": "error"}),
            ("classify_incident", {"severity": "MEDIUM"}),
        ]

        rewards1 = []
        rewards2 = []

        for run_rewards in [rewards1, rewards2]:
            env = LogTriageEnv()
            env.reset("task_easy")
            for action_type, params in actions:
                result = env.step(action_type, params)
                run_rewards.append(result["reward"])

        for r1, r2 in zip(rewards1, rewards2):
            assert r1 == r2, "Reward traces diverged"

    def test_observation_determinism_across_steps(self):
        """Observations at each step should be identical for same actions."""
        actions = [
            ("filter_severity", {"level": "ERROR"}),
            ("scroll", {"direction": "down"}),
            ("inspect", {"log_id": "log_008"}),
        ]

        obs_list1 = []
        obs_list2 = []

        for obs_list in [obs_list1, obs_list2]:
            env = LogTriageEnv()
            env.reset("task_easy")
            for action_type, params in actions:
                result = env.step(action_type, params)
                obs_list.append(result["observation"])

        for o1, o2 in zip(obs_list1, obs_list2):
            assert o1 == o2


# ─── Seeding independence ────────────────────────────────────────


class TestSeedingIndependence:
    """Generating one task should not affect another task's seed."""

    def test_generation_order_independence(self):
        """Generating easy then hard should give same result as hard alone."""
        # Generate easy first, then hard
        generate_logs("task_easy")
        hard_after_easy = generate_logs("task_hard")

        # Generate hard alone
        hard_alone = generate_logs("task_hard")

        assert _hash_logs(hard_after_easy) == _hash_logs(hard_alone)

    def test_environment_isolation(self):
        """Running one environment should not affect another."""
        env1 = LogTriageEnv()
        env1.reset("task_easy")
        env1.step("search", {"pattern": "error"})

        # Now create a fresh env for the same task
        env2 = LogTriageEnv()
        result = env2.reset("task_easy")
        # Should start fresh, not be affected by env1
        assert result["observation"]["step_number"] == 0
        assert result["observation"]["annotations_count"] == 0
