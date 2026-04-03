"""
Tests for src/log_generator.py — deterministic log generation.
"""

import pytest
from src.log_generator import generate_logs
from src.tasks import TASKS


# ─── Basic generation ────────────────────────────────────────────


class TestLogGeneration:
    """Verify generate_logs produces correct output for each task."""

    @pytest.mark.parametrize("task_id", ["task_easy", "task_medium", "task_hard"])
    def test_correct_log_count(self, task_id):
        """Generated logs must match the configured log_count."""
        logs = generate_logs(task_id)
        expected = TASKS[task_id]["log_count"]
        assert len(logs) == expected

    @pytest.mark.parametrize("task_id", ["task_easy", "task_medium", "task_hard"])
    def test_log_ids_are_sequential(self, task_id):
        """log IDs should be log_001, log_002, ..., log_N."""
        logs = generate_logs(task_id)
        for i, log in enumerate(logs, start=1):
            assert log["id"] == f"log_{i:03d}"

    @pytest.mark.parametrize("task_id", ["task_easy", "task_medium", "task_hard"])
    def test_required_fields_present(self, task_id):
        """Every log entry must have id, timestamp, service, severity, message."""
        logs = generate_logs(task_id)
        required = {"id", "timestamp", "service", "severity", "message"}
        for log in logs:
            assert required.issubset(log.keys()), f"Missing fields in {log['id']}"

    @pytest.mark.parametrize("task_id", ["task_easy", "task_medium", "task_hard"])
    def test_timestamps_are_monotonic(self, task_id):
        """Timestamps should be non-decreasing (time moves forward)."""
        logs = generate_logs(task_id)
        timestamps = [log["timestamp"] for log in logs]
        assert timestamps == sorted(timestamps), "Timestamps are not monotonically increasing"

    @pytest.mark.parametrize("task_id", ["task_easy", "task_medium", "task_hard"])
    def test_services_are_valid(self, task_id):
        """Every log's service should be from the task's configured list."""
        logs = generate_logs(task_id)
        valid_services = set(TASKS[task_id]["services"])
        for log in logs:
            assert log["service"] in valid_services, (
                f"{log['id']} has invalid service '{log['service']}'"
            )

    @pytest.mark.parametrize("task_id", ["task_easy", "task_medium", "task_hard"])
    def test_severities_are_valid(self, task_id):
        """Every severity should be one of the standard levels."""
        valid_severities = {"DEBUG", "INFO", "WARN", "ERROR", "FATAL"}
        logs = generate_logs(task_id)
        for log in logs:
            assert log["severity"] in valid_severities, (
                f"{log['id']} has invalid severity '{log['severity']}'"
            )


# ─── Ground truth placement ──────────────────────────────────────


class TestGroundTruthLogs:
    """Verify ground-truth log entries appear at the correct positions."""

    @pytest.mark.parametrize("task_id", ["task_easy", "task_medium", "task_hard"])
    def test_ground_truth_ids_exist(self, task_id):
        """All ground-truth log IDs from the task config must be present."""
        logs = generate_logs(task_id)
        log_ids = {log["id"] for log in logs}
        gt_ids = set(TASKS[task_id]["ground_truth"]["annotations"].keys())
        assert gt_ids.issubset(log_ids), f"Missing GT IDs: {gt_ids - log_ids}"

    def test_easy_gt_positions(self):
        """Easy task GT logs should be at positions 8, 23, 41."""
        logs = generate_logs("task_easy")
        id_map = {log["id"]: log for log in logs}
        for log_id in ["log_008", "log_023", "log_041"]:
            assert log_id in id_map
            assert id_map[log_id]["service"] == "auth-service"
            assert id_map[log_id]["severity"] == "ERROR"

    def test_medium_gt_contains_multiple_services(self):
        """Medium task GT logs span payment-service, order-service, api-gateway."""
        logs = generate_logs("task_medium")
        id_map = {log["id"]: log for log in logs}
        gt_ann = TASKS["task_medium"]["ground_truth"]["annotations"]
        gt_services = {id_map[lid]["service"] for lid in gt_ann if lid in id_map}
        assert len(gt_services) >= 2, "Medium GT should span multiple services"

    def test_hard_gt_contains_attack_indicators(self):
        """Hard task GT logs should contain security-related messages."""
        logs = generate_logs("task_hard")
        id_map = {log["id"]: log for log in logs}
        # log_078 is brute force
        assert "failed login" in id_map["log_078"]["message"].lower()
        # log_145 is credential compromise
        assert "svc-deploy" in id_map["log_145"]["message"].lower()
        # log_267 is data exfiltration
        assert "export" in id_map["log_267"]["message"].lower()

    def test_hard_gt_metadata_has_source_ip(self):
        """Hard task attack logs should have the attacker IP in metadata."""
        logs = generate_logs("task_hard")
        id_map = {log["id"]: log for log in logs}
        attacker_ip = "198.51.100.23"
        for lid in ["log_023", "log_078", "log_091", "log_145"]:
            meta = id_map[lid].get("metadata", {})
            assert meta.get("source_ip") == attacker_ip, (
                f"{lid} should have attacker IP in metadata"
            )


# ─── Noise and distractors ───────────────────────────────────────


class TestNoiseAndDistractors:
    """Verify noise/distractor log properties."""

    def test_easy_has_no_distractors(self):
        """Easy task should have only background noise (no distractors)."""
        logs = generate_logs("task_easy")
        gt_ids = set(TASKS["task_easy"]["ground_truth"]["annotations"].keys())
        non_gt = [l for l in logs if l["id"] not in gt_ids]
        # All non-GT logs should be background noise (mostly INFO/DEBUG)
        assert len(non_gt) == 50 - len(gt_ids)

    def test_medium_has_distractors(self):
        """Medium task should have some WARN/ERROR distractors that are NOT GT."""
        logs = generate_logs("task_medium")
        gt_ids = set(TASKS["task_medium"]["ground_truth"]["annotations"].keys())
        non_gt_errors = [
            l for l in logs
            if l["id"] not in gt_ids and l["severity"] in ("ERROR", "WARN")
        ]
        # Medium has 5 distractors
        assert len(non_gt_errors) >= 5, "Medium task should have distractor errors"

    def test_hard_has_more_distractors(self):
        """Hard task should have more distractors than medium."""
        logs = generate_logs("task_hard")
        gt_ids = set(TASKS["task_hard"]["ground_truth"]["annotations"].keys())
        non_gt_errors = [
            l for l in logs
            if l["id"] not in gt_ids and l["severity"] in ("ERROR", "WARN")
        ]
        # Hard has 15 distractors (some bg WARN are also possible)
        assert len(non_gt_errors) >= 15, "Hard task should have many distractor errors"

    def test_background_messages_are_non_empty(self):
        """All log messages should be non-empty strings."""
        logs = generate_logs("task_easy")
        for log in logs:
            assert isinstance(log["message"], str)
            assert len(log["message"]) > 0, f"{log['id']} has empty message"


# ─── Timestamp format ────────────────────────────────────────────


class TestTimestampFormat:
    """Verify timestamp formatting."""

    @pytest.mark.parametrize("task_id", ["task_easy", "task_medium", "task_hard"])
    def test_timestamp_format(self, task_id):
        """Timestamps should match ISO8601-like format: 2024-01-15THH:MM:SS.mmmZ."""
        import re
        logs = generate_logs(task_id)
        pattern = r"2024-01-15T\d{2}:\d{2}:\d{2}\.\d{3}Z"
        for log in logs:
            assert re.match(pattern, log["timestamp"]), (
                f"{log['id']} timestamp '{log['timestamp']}' doesn't match format"
            )

    def test_easy_starts_around_9am(self):
        """Easy task logs should start around 09:00."""
        logs = generate_logs("task_easy")
        first_ts = logs[0]["timestamp"]
        assert first_ts.startswith("2024-01-15T09:")

    def test_hard_starts_around_2am(self):
        """Hard task logs should start around 02:00 (security scenario)."""
        logs = generate_logs("task_hard")
        first_ts = logs[0]["timestamp"]
        assert first_ts.startswith("2024-01-15T02:")
