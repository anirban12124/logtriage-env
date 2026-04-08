"""
Test: Verify ALL scores emitted by the grading pipeline are strictly in (0, 1).
This simulates the validator's check.
"""
import sys
import math

# Ensure imports work
sys.path.insert(0, ".")

from src.reward import RewardCalculator
from src.grader import TaskGrader
from src.tasks import TASKS

EPS = 0.001

def assert_in_range(value, label):
    """Assert value is strictly in (0, 1) — not 0.0, not 1.0."""
    assert isinstance(value, (int, float)), f"{label}: expected numeric, got {type(value)}"
    assert not math.isnan(value), f"{label}: NaN detected"
    assert not math.isinf(value), f"{label}: inf detected"
    assert value > 0.0, f"{label}: {value} <= 0.0 (must be > 0)"
    assert value < 1.0, f"{label}: {value} >= 1.0 (must be < 1)"


def check_reward_output(reward_dict, context):
    """Check all values in a reward dict are in (0, 1)."""
    assert_in_range(reward_dict["value"], f"{context}.value")
    assert_in_range(reward_dict["cumulative"], f"{context}.cumulative")
    for k, v in reward_dict.get("components", {}).items():
        assert_in_range(v, f"{context}.components.{k}")


def check_grader_output(grader_result, context):
    """Check all scores in grader output are in (0, 1)."""
    assert_in_range(grader_result["score"], f"{context}.score")
    assert_in_range(grader_result["final_score"], f"{context}.final_score")
    for comp_name, comp_data in grader_result.get("components", {}).items():
        if isinstance(comp_data, dict):
            assert_in_range(comp_data["score"], f"{context}.components.{comp_name}.score")
            assert_in_range(comp_data["weight"], f"{context}.components.{comp_name}.weight")
            assert_in_range(comp_data["weighted"], f"{context}.components.{comp_name}.weighted")


def test_reward_calculator():
    """Test that RewardCalculator never returns values outside (0, 1)."""
    print("=" * 60)
    print("TEST: RewardCalculator output ranges")
    print("=" * 60)
    
    for task_id, task_config in TASKS.items():
        gt = task_config["ground_truth"]
        calc = RewardCalculator(gt, task_config)
        
        # Simulate various action sequences
        actions = [
            ("search", {"pattern": "error"}, {"search_hits": 5, "relevant_hits": 2}),
            ("filter_severity", {"level": "ERROR"}, {"old_density": 0, "new_density": 0.3}),
            ("scroll", {"direction": "down"}, {"old_density": 0.3, "new_density": 0.1}),
            ("annotate", {"log_id": "FAKE_ID", "category": "error"}, {}),  # wrong annotation
            ("noop", {}, {}),
            ("classify_incident", {"severity": "LOW"}, {}),  # possibly wrong
            ("submit_report", {"summary": "test report"}, {"episode_done": True}),
        ]
        
        for step_i, (action_type, params, env_state) in enumerate(actions, 1):
            result = calc.calculate(action_type, params, step_i, env_state)
            check_reward_output(result, f"{task_id}.step{step_i}({action_type})")
        
        print(f"  ✅ {task_id}: all reward values in (0, 1)")
    
    # Edge case: test with correct annotations
    for task_id, task_config in TASKS.items():
        gt = task_config["ground_truth"]
        calc = RewardCalculator(gt, task_config)
        
        # Annotate all correct log IDs
        step = 0
        for log_id, category in gt["annotations"].items():
            step += 1
            result = calc.calculate("annotate", {"log_id": log_id, "category": category}, step, {})
            check_reward_output(result, f"{task_id}_correct.step{step}(annotate)")
        
        # Correlate all correct pairs
        for pair in gt.get("correlations", []):
            step += 1
            result = calc.calculate(
                "correlate",
                {"source_log_id": pair[0], "target_log_id": pair[1]},
                step, {}
            )
            check_reward_output(result, f"{task_id}_correct.step{step}(correlate)")
        
        # Classify correct severity
        step += 1
        result = calc.calculate(
            "classify_incident",
            {"severity": gt["severity"]},
            step, {}
        )
        check_reward_output(result, f"{task_id}_correct.step{step}(classify)")
        
        # Submit report
        step += 1
        result = calc.calculate(
            "submit_report",
            {"summary": " ".join(gt["key_findings"])},
            step, {"episode_done": True}
        )
        check_reward_output(result, f"{task_id}_correct.step{step}(submit)")
        
        print(f"  ✅ {task_id}: correct-path reward values all in (0, 1)")


def test_grader():
    """Test that TaskGrader never returns scores outside (0, 1)."""
    print()
    print("=" * 60)
    print("TEST: TaskGrader output ranges")
    print("=" * 60)
    
    grader = TaskGrader()
    
    for task_id, task_config in TASKS.items():
        gt = task_config["ground_truth"]
        
        # Case 1: Empty submission (worst case)
        result = grader.grade(
            agent_annotations={},
            agent_correlations=[],
            agent_severity="",
            agent_report="",
            ground_truth=gt,
            task_config=task_config,
            behavior={"steps_taken": task_config["max_steps"], "report_source": "none"},
        )
        check_grader_output(result, f"{task_id}.empty")
        print(f"  ✅ {task_id} empty: score={result['score']} -- all in (0, 1)")
        
        # Case 2: Perfect submission
        result = grader.grade(
            agent_annotations=dict(gt["annotations"]),
            agent_correlations=[list(c) for c in gt.get("correlations", [])],
            agent_severity=gt["severity"],
            agent_report=" ".join(gt["key_findings"]) + " root cause impact timeline affected",
            ground_truth=gt,
            task_config=task_config,
            behavior={"steps_taken": 3, "report_source": "submitted"},
        )
        check_grader_output(result, f"{task_id}.perfect")
        print(f"  ✅ {task_id} perfect: score={result['score']} -- all in (0, 1)")
        
        # Case 3: Partial/wrong submission
        result = grader.grade(
            agent_annotations={"fake_log_1": "error", "fake_log_2": "warning"},
            agent_correlations=[["fake_1", "fake_2"]],
            agent_severity="LOW",
            agent_report="something happened",
            ground_truth=gt,
            task_config=task_config,
            behavior={"steps_taken": task_config["max_steps"], "report_source": "draft"},
        )
        check_grader_output(result, f"{task_id}.wrong")
        print(f"  ✅ {task_id} wrong: score={result['score']} -- all in (0, 1)")


if __name__ == "__main__":
    try:
        test_reward_calculator()
        test_grader()
        print()
        print("=" * 60)
        print("ALL TESTS PASSED ✅")
        print("All scores are strictly in (0, 1)")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ ASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
