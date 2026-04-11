from src.models import (
    Observation, Action, ActionType, Reward, LogEntry,
    StepResult, ResetResult, GraderResult,
)
from src.tasks import TASKS
from src.log_generator import generate_logs
from src.reward import RewardCalculator
from src.grader import TaskGrader

from typing import List, Dict, Optional, Any
from rank_bm25 import BM25Okapi
import math


class LogTriageEnv:
    def __init__(self):
        self.task_config: Optional[dict] = None
        self.grader = TaskGrader()
        self.reward_calc: Optional[RewardCalculator] = None
        self._reset_internal()

    def _reset_internal(self):
        self.logs: List[dict] = []
        self.filtered_logs: List[dict] = []
        self.current_page: int = 0
        self.page_size: int = 20
        self.step_count: int = 0
        self.done: bool = False

        self.agent_annotations: Dict[str, str] = {}
        self.agent_correlations: List[List[str]] = []
        self.agent_severity: str = ""
        self.agent_report: str = ""
        self.report_source: str = "none"  # "none", "draft", "submitted"
        self.prev_draft_matches: int = 0

        self.active_filters: Dict[str, str] = {}
        self.search_pattern: str = ""
        self.inspected_log: Optional[dict] = None

        self.last_action_success: bool = True
        self.last_action_message: str = "OK"
        self.draft_feedback: Optional[str] = None

        # BM25 search index (built at reset)
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_corpus_tokens: List[List[str]] = []
        self._search_ranked_ids: Optional[List[str]] = None

    # ─── Score Clamping Helpers ─────────────────────────────────

    @staticmethod
    def _clamp_score(v, eps=0.001):
        """Clamp value to strictly (0, 1) — safe against NaN/inf/None."""
        if v is None or not isinstance(v, (int, float)):
            return float(0.5)
        v = float(v)  # ensure pure Python float
        if math.isnan(v) or math.isinf(v):
            return float(0.5)
        # Clamp to [0, 1] first
        v = max(0.0, min(1.0, v))
        # Then enforce strict open interval
        if v <= 0.0:
            v = eps
        if v >= 1.0:
            v = 1.0 - eps
        return float(v)

    @staticmethod
    def _clamp_reward(reward_obj, eps=0.001):
        """Clamp all numeric score-like values inside a reward dict or float."""
        SCORE_KEYS = {
            "value", "cumulative", "score", "task_score", "final_score",
            "annotation_precision", "annotation_recall", "annotation_quality",
            "correlation_precision", "correlation_recall",
            "chain_reconstruction", "severity_classification",
            "report_completeness", "report_coherence",
            "investigation_efficiency",
        }

        def safe(v):
            if v is None or not isinstance(v, (int, float)):
                return float(0.5)
            v = float(v)  # ensure pure Python float
            if math.isnan(v) or math.isinf(v):
                return float(0.5)
            v = max(0.0, min(1.0, v))
            if v <= 0.0:
                v = eps
            if v >= 1.0:
                v = 1.0 - eps
            return float(v)

        if isinstance(reward_obj, (int, float)):
            return safe(reward_obj)

        if isinstance(reward_obj, dict):
            clamped = {}
            for k, v in reward_obj.items():
                if isinstance(v, (int, float)) and k in SCORE_KEYS:
                    clamped[k] = safe(v)
                elif isinstance(v, dict):
                    # Recursively clamp nested dicts
                    inner = {}
                    for ik, iv in v.items():
                        if isinstance(iv, (int, float)) and ik in SCORE_KEYS:
                            inner[ik] = safe(iv)
                        elif isinstance(iv, dict):
                            inner[ik] = LogTriageEnv._clamp_reward(iv, eps)
                        else:
                            inner[ik] = iv
                    clamped[k] = inner
                else:
                    clamped[k] = v
            return clamped

        return reward_obj

    # ─── BM25 ──────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowering tokenizer for BM25."""
        return text.lower().split()

    def _build_bm25_index(self):
        """Build BM25 index over all log messages + service + metadata."""
        corpus_tokens = []
        for log in self.logs:
            parts = [
                log.get("message", ""),
                log.get("service", ""),
                log.get("severity", ""),
            ]
            for v in log.get("metadata", {}).values():
                if isinstance(v, str):
                    parts.append(v)
            doc_text = " ".join(parts)
            corpus_tokens.append(self._tokenize(doc_text))

        self._bm25_corpus_tokens = corpus_tokens
        self._bm25_index = BM25Okapi(corpus_tokens)

    # ─── Reset ─────────────────────────────────────────────────

    def reset(self, task_id: str) -> dict:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task: {task_id}. Available: {list(TASKS.keys())}")

        self.task_config = TASKS[task_id]
        self._reset_internal()
        self.logs = generate_logs(task_id)
        self.filtered_logs = self.logs.copy()
        self.reward_calc = RewardCalculator(
            self.task_config["ground_truth"], self.task_config
        )

        # Build BM25 index for ranked search
        self._build_bm25_index()

        # Track initial page view
        self._track_page_view()

        obs = self._build_observation()
        return {
            "observation": obs.model_dump(),
            "info": {
                "task_name": self.task_config["name"],
                "max_steps": self.task_config["max_steps"],
            },
        }

    # ─── Step ──────────────────────────────────────────────────

    def step(self, action_type: str, params: dict = None) -> dict:
        if params is None:
            params = {}

        if self.task_config is None:
            raise ValueError("Environment not initialized. Call reset() first.")

        # ── Already-done early return ──
        if self.done:
            obs = self._build_observation()
            clamped_cumulative = self._clamp_score(
                self.reward_calc.cumulative if self.reward_calc else 0.0
            )
            return {
                "observation": obs.model_dump(),
                "reward": {
                    "value": self._clamp_score(0.0),
                    "components": {},
                    "cumulative": clamped_cumulative,
                },
                "done": True,
                "info": {
                    "message": "Episode already complete.",
                    "score": clamped_cumulative,
                    "task_score": clamped_cumulative,
                },
            }

        self.step_count += 1
        self.last_action_success = True
        self.last_action_message = "OK"
        self.inspected_log = None
        self.draft_feedback = None

        # Build pre-action state for density calculation
        old_density = self._relevant_density()

        # Process action
        env_state = {}
        try:
            env_state = self._process_action(action_type, params)
        except Exception as e:
            self.last_action_success = False
            self.last_action_message = str(e)

        # Post-action density
        new_density = self._relevant_density()
        env_state["old_density"] = old_density
        env_state["new_density"] = new_density

        # Track page view after action
        self._track_page_view()

        # Check episode end
        if action_type == "submit_report" and self.step_count >= 3:
            self.done = True
            self.report_source = "submitted"
            env_state["episode_done"] = True
        elif self.step_count >= self.task_config["max_steps"]:
            self.done = True
            env_state["episode_done"] = True
            # Use draft as fallback if no submission
            if self.report_source == "none" and self.agent_report:
                self.report_source = "draft"

        # Calculate reward — CLAMP IT
        reward_result = self.reward_calc.calculate(
            action_type, params, self.step_count, env_state
        )
        reward_result = self._clamp_reward(reward_result)

        # Grade if done
        info: Dict[str, Any] = {}
        if self.done:
            behavior = {
                "steps_taken": self.step_count,
                "report_source": self.report_source,
            }
            grader_result = self.grader.grade(
                self.agent_annotations,
                self.agent_correlations,
                self.agent_severity,
                self.agent_report,
                self.task_config["ground_truth"],
                self.task_config,
                behavior,
            )
            info["grader_result"] = grader_result

            # Expose clamped score at EVERY level the validator might check
            task_score = self._clamp_score(grader_result.get("score", 0.5))
            print(f"GRADE: task_score = {task_score} (type={type(task_score).__name__})")
            info["score"] = task_score
            info["task_score"] = task_score
            info["final_score"] = task_score

            # Also ensure reward value matches the task score on final step
            if isinstance(reward_result, dict):
                reward_result["value"] = task_score
                reward_result["score"] = task_score
                reward_result["task_score"] = task_score
            else:
                reward_result = {
                    "value": task_score,
                    "score": task_score,
                    "task_score": task_score,
                }
        else:
            # Non-done steps: ensure info has a clamped placeholder score
            info["score"] = self._clamp_score(0.0)
            info["task_score"] = self._clamp_score(0.0)

        obs = self._build_observation()
        return {
            "observation": obs.model_dump(),
            "reward": reward_result,
            "done": self.done,
            "info": info,
            "score": info.get("score", self._clamp_score(0.0)),
        }

    # ─── State ─────────────────────────────────────────────────

    def state(self) -> dict:
        return {
            "task_id": self.task_config["id"] if self.task_config else None,
            "step": self.step_count,
            "done": self.done,
            "annotations": self.agent_annotations,
            "correlations": self.agent_correlations,
            "severity": self.agent_severity,
            "report": self.agent_report,
            "report_source": self.report_source,
            "filters": self.active_filters,
        }

    # ─── Action Processing ──────────────────────────────────────

    def _process_action(self, action_type: str, params: dict) -> dict:
        env_state = {}

        if action_type == "search":
            env_state = self._do_search(params.get("pattern", ""))

        elif action_type == "filter_severity":
            self._do_filter_severity(params.get("level", ""))

        elif action_type == "filter_service":
            self._do_filter_service(params.get("service", ""))

        elif action_type == "filter_time_range":
            self._do_filter_time_range(
                params.get("start", ""), params.get("end", "")
            )

        elif action_type == "clear_filters":
            self.active_filters = {}
            self.search_pattern = ""
            self.filtered_logs = self.logs.copy()
            self.current_page = 0

        elif action_type == "scroll":
            direction = params.get("direction", "down")
            total_pages = max(1, (len(self.filtered_logs) + self.page_size - 1)
                             // self.page_size)
            if direction == "down":
                self.current_page = min(self.current_page + 1, total_pages - 1)
            elif direction == "up":
                self.current_page = max(self.current_page - 1, 0)

        elif action_type == "inspect":
            env_state = self._do_inspect(params.get("log_id", ""))

        elif action_type == "annotate":
            self._do_annotate(
                params.get("log_id", ""),
                params.get("category", ""),
            )

        elif action_type == "correlate":
            self._do_correlate(
                params.get("source_log_id", ""),
                params.get("target_log_id", ""),
            )

        elif action_type == "classify_incident":
            self._do_classify(params.get("severity", ""))

        elif action_type == "draft_report":
            env_state = self._do_draft_report(params.get("summary", ""))

        elif action_type == "submit_report":
            env_state = self._do_submit_report(params.get("summary", ""))

        elif action_type == "noop":
            pass

        else:
            self.last_action_success = False
            self.last_action_message = (
                f"Unknown action type: '{action_type}'. "
                f"Valid types: search, filter_severity, filter_service, "
                f"filter_time_range, clear_filters, scroll, inspect, "
                f"annotate, correlate, classify_incident, draft_report, "
                f"submit_report, noop"
            )

        return env_state

    def _do_search(self, pattern: str) -> dict:
        if not pattern:
            self.search_pattern = ""
            self._search_ranked_ids = None
            self.active_filters.pop("search", None)
            self._apply_filters()
            self.last_action_message = "Search cleared. Showing all logs."
            return {"search_hits": len(self.filtered_logs), "relevant_hits": 0}

        self.search_pattern = pattern.lower()
        self.active_filters["search"] = pattern

        # BM25 ranking
        if self._bm25_index is not None:
            query_tokens = self._tokenize(pattern)
            scores = self._bm25_index.get_scores(query_tokens)

            scored = sorted(
                zip(self.logs, scores),
                key=lambda x: x[1],
                reverse=True,
            )

            self._search_ranked_ids = [
                log["id"] for log, score in scored if score > 0.0
            ]
        else:
            self._search_ranked_ids = None

        self._apply_filters()
        self.current_page = 0

        gt_ann = self.task_config["ground_truth"]["annotations"]
        gt_ids = set(gt_ann.keys())
        found_gt = set()
        for log in self.filtered_logs:
            if log["id"] in gt_ids:
                found_gt.add(log["id"])

        self.last_action_message = f"Search '{pattern}': {len(self.filtered_logs)} results (BM25 ranked)."
        return {
            "search_hits": len(self.filtered_logs),
            "relevant_hits": len(found_gt),
        }

    def _do_filter_severity(self, level: str):
        valid = {"DEBUG", "INFO", "WARN", "ERROR", "FATAL"}
        level_upper = level.upper()
        if level_upper not in valid:
            self.last_action_success = False
            self.last_action_message = (
                f"Invalid severity '{level}'. Use: {', '.join(sorted(valid))}"
            )
            return
        self.active_filters["severity"] = level_upper
        self._apply_filters()
        self.current_page = 0

    def _do_filter_service(self, service: str):
        available = self.task_config["services"]
        if service not in available:
            self.last_action_success = False
            self.last_action_message = (
                f"Service '{service}' not found. Available: {', '.join(available)}"
            )
            return
        self.active_filters["service"] = service
        self._apply_filters()
        self.current_page = 0

    def _do_filter_time_range(self, start: str, end: str):
        if not start or not end:
            self.last_action_success = False
            self.last_action_message = "Both 'start' and 'end' timestamps required."
            return
        self.active_filters["time_start"] = start
        self.active_filters["time_end"] = end
        self._apply_filters()
        self.current_page = 0

    def _do_inspect(self, log_id: str) -> dict:
        log = self._find_log(log_id)
        if log is None:
            self.last_action_success = False
            self.last_action_message = f"Log ID '{log_id}' not found."
            return {}
        self.inspected_log = log
        self.last_action_message = f"Inspecting {log_id}."
        return {}

    def _do_annotate(self, log_id: str, category: str):
        if not log_id or not category:
            self.last_action_success = False
            self.last_action_message = "Both 'log_id' and 'category' are required."
            return

        log = self._find_log(log_id)
        if log is None:
            self.last_action_success = False
            self.last_action_message = f"Log ID '{log_id}' not found."
            return

        old_cat = self.agent_annotations.get(log_id)
        self.agent_annotations[log_id] = category
        if old_cat:
            self.last_action_message = (
                f"Updated annotation for {log_id}: {old_cat} → {category}"
            )
        else:
            self.last_action_message = f"Annotated {log_id} as '{category}'."

    def _do_correlate(self, source: str, target: str):
        if not source or not target:
            self.last_action_success = False
            self.last_action_message = "Both 'source_log_id' and 'target_log_id' required."
            return

        if source == target:
            self.last_action_success = False
            self.last_action_message = "Cannot correlate a log with itself."
            return

        if self._find_log(source) is None:
            self.last_action_success = False
            self.last_action_message = f"Source log '{source}' not found."
            return

        if self._find_log(target) is None:
            self.last_action_success = False
            self.last_action_message = f"Target log '{target}' not found."
            return

        self.agent_correlations.append([source, target])
        self.last_action_message = f"Correlated {source} → {target}."

    def _do_classify(self, severity: str):
        valid = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        sev_clean = severity.strip().upper()
        if sev_clean not in valid:
            self.last_action_success = False
            self.last_action_message = (
                f"Invalid severity '{severity}'. Use: {', '.join(sorted(valid))}"
            )
            return
        self.agent_severity = sev_clean
        self.last_action_message = f"Incident classified as {sev_clean}."

    def _do_draft_report(self, summary: str) -> dict:
        self.agent_report = summary
        if self.report_source == "none":
            self.report_source = "draft"

        gt_findings = self.task_config["ground_truth"]["key_findings"]
        matches = sum(1 for kf in gt_findings if kf.lower() in summary.lower())
        total = len(gt_findings)

        self.draft_feedback = f"Report covers approximately {matches}/{total} key areas."
        self.last_action_message = "Draft saved."

        env_state = {"prev_draft_matches": self.prev_draft_matches}
        self.prev_draft_matches = matches
        return env_state

    def _do_submit_report(self, summary: str) -> dict:
        if self.step_count < 3:
            result = self._do_draft_report(summary)
            self.last_action_message = (
                f"Too early for final submission (step {self.step_count}/3 min). "
                f"Saved as draft. Submit again after step 3."
            )
            return result

        self.agent_report = summary
        self.report_source = "submitted"
        self.last_action_message = "Report submitted. Episode complete."
        return {}

    # ─── Helpers ────────────────────────────────────────────────

    def _find_log(self, log_id: str) -> Optional[dict]:
        for log in self.logs:
            if log["id"] == log_id:
                return log
        return None

    def _apply_filters(self):
        result = self.logs.copy()

        if "severity" in self.active_filters:
            sev = self.active_filters["severity"]
            result = [l for l in result if l["severity"] == sev]

        if "service" in self.active_filters:
            svc = self.active_filters["service"]
            result = [l for l in result if l["service"] == svc]

        if "time_start" in self.active_filters and "time_end" in self.active_filters:
            start = self.active_filters["time_start"]
            end = self.active_filters["time_end"]
            result = [l for l in result if start <= l["timestamp"] <= end]

        # Search: use BM25 ranked order if available, else naive substring
        if self.search_pattern:
            if self._search_ranked_ids is not None:
                ranked_set = set(self._search_ranked_ids)
                filtered_ids = set(l["id"] for l in result)
                id_to_log = {l["id"]: l for l in result}
                result = [
                    id_to_log[lid] for lid in self._search_ranked_ids
                    if lid in filtered_ids and lid in ranked_set
                ]
            else:
                pattern = self.search_pattern
                result = [l for l in result if pattern in l["message"].lower()
                          or pattern in l["service"].lower()
                          or any(pattern in v.lower()
                                  for v in l.get("metadata", {}).values()
                                  if isinstance(v, str))]

        self.filtered_logs = result

    def _relevant_density(self) -> float:
        visible = self._get_visible_logs()
        if not visible:
            return 0.0
        gt_ids = set(self.task_config["ground_truth"]["annotations"].keys())
        relevant = sum(1 for l in visible if l["id"] in gt_ids)
        return relevant / len(visible)

    def _get_visible_logs(self) -> list:
        start = self.current_page * self.page_size
        end = start + self.page_size
        return self.filtered_logs[start:end]

    def _track_page_view(self):
        if self.reward_calc is None:
            return
        visible = self._get_visible_logs()
        log_ids = [l["id"] for l in visible]
        services = list(set(l["service"] for l in visible))
        self.reward_calc.track_page_view(self.current_page, log_ids, services)

    def _count_severities(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for log in self.logs:
            sev = log["severity"]
            counts[sev] = counts.get(sev, 0) + 1
        return counts

    def _build_observation(self) -> Observation:
        visible = self._get_visible_logs()
        total_pages = max(1, (len(self.filtered_logs) + self.page_size - 1)
                         // self.page_size)

        visible_entries = []
        for log in visible:
            msg = log["message"]
            if len(msg) > 200:
                msg = msg[:197] + "..."
            visible_entries.append(LogEntry(
                id=log["id"],
                timestamp=log["timestamp"],
                service=log["service"],
                severity=log["severity"],
                message=msg,
                metadata=log.get("metadata", {}),
            ))

        inspected = None
        if self.inspected_log:
            inspected = LogEntry(
                id=self.inspected_log["id"],
                timestamp=self.inspected_log["timestamp"],
                service=self.inspected_log["service"],
                severity=self.inspected_log["severity"],
                message=self.inspected_log["message"],
                metadata=self.inspected_log.get("metadata", {}),
            )

        ann_items = list(self.agent_annotations.items())
        recent_ann = [{"log_id": k, "category": v} for k, v in ann_items[-5:]]

        ann_by_cat: Dict[str, int] = {}
        for cat in self.agent_annotations.values():
            ann_by_cat[cat] = ann_by_cat.get(cat, 0) + 1

        recent_corr = self.agent_correlations[-3:]

        return Observation(
            task_id=self.task_config["id"],
            goal=self.task_config["goal"],
            step_number=self.step_count,
            max_steps=self.task_config["max_steps"],
            visible_logs=visible_entries,
            total_log_count=len(self.filtered_logs),
            current_page=self.current_page,
            total_pages=total_pages,
            severity_counts=self._count_severities(),
            available_services=self.task_config["services"],
            current_filters=self.active_filters,
            annotations_count=len(self.agent_annotations),
            recent_annotations=recent_ann,
            annotations_by_category=ann_by_cat,
            correlations_count=len(self.agent_correlations),
            recent_correlations=recent_corr,
            severity_classified=self.agent_severity or None,
            current_report_draft=self.agent_report or None,
            inspected_log=inspected,
            last_action_success=self.last_action_success,
            last_action_message=self.last_action_message,
            draft_feedback=self.draft_feedback,
        )