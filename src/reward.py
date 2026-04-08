from typing import Dict, List, Set, Tuple, Optional
from src.tasks import get_category_similarity


class RewardCalculator:
    def __init__(self, ground_truth: dict, task_config: dict):
        self.gt = ground_truth
        self.config = task_config
        self.difficulty = task_config["difficulty"]
        self.max_steps = task_config["max_steps"]

        self.cumulative = 0.0
        self.prev_actions: List[str] = []
        self.found_annotations: Set[str] = set()
        self.found_correlations: Set[Tuple[str, str]] = set()
        self.total_annotations_submitted = 0
        self.correct_annotations_submitted = 0
        self.viewed_pages: Set[int] = set()
        self.viewed_log_ids: Set[str] = set()
        self.services_explored: Set[str] = set()
        self.has_searched_before_annotating = False
        self.first_annotation_step: Optional[int] = None
        self.first_search_step: Optional[int] = None
        self.severity_submitted = False
        self.draft_submitted = False

        # Coverage milestones
        self._milestone_25 = False
        self._milestone_50 = False
        self._milestone_75 = False
        self._milestone_100 = False

        # Difficulty multiplier
        self._diff_mult = {"easy": 1.0, "medium": 1.3, "hard": 1.6}.get(
            self.difficulty, 1.0
        )

    def _running_precision(self) -> float:
        if self.total_annotations_submitted == 0:
            return 1.0
        return self.correct_annotations_submitted / self.total_annotations_submitted

    def _coverage(self) -> float:
        gt_count = len(self.gt.get("annotations", {}))
        if gt_count == 0:
            return 1.0
        return len(self.found_annotations) / gt_count

    def _coverage_multiplier(self) -> float:
        c = self._coverage()
        if c < 0.5:
            return 1.2
        elif c <= 0.8:
            return 1.0
        else:
            return 0.8

    def _temporal_multiplier(self, step: int) -> float:
        ratio = step / self.max_steps
        if ratio < 0.3:
            return 1.3
        elif ratio < 0.7:
            return 1.0
        else:
            return 0.7

    def _informed_bonus(self, log_id: str) -> float:
        if log_id in self.viewed_log_ids:
            return 1.1
        return 0.5

    def _check_milestones(self) -> float:
        bonus = 0.0
        c = self._coverage()
        if c >= 0.25 and not self._milestone_25:
            self._milestone_25 = True
            bonus += 0.05
        if c >= 0.50 and not self._milestone_50:
            self._milestone_50 = True
            bonus += 0.08
        if c >= 0.75 and not self._milestone_75:
            self._milestone_75 = True
            bonus += 0.10
        if c >= 1.0 and not self._milestone_100:
            self._milestone_100 = True
            bonus += 0.15
        return bonus

    def _chain_length(self, correlations: Set[Tuple[str, str]]) -> int:
        if not correlations:
            return 0
        graph: Dict[str, List[str]] = {}
        for s, t in correlations:
            graph.setdefault(s, []).append(t)

        def dfs(node: str, visited: set) -> int:
            best = 0
            for nxt in graph.get(node, []):
                if nxt not in visited:
                    visited.add(nxt)
                    best = max(best, 1 + dfs(nxt, visited))
                    visited.remove(nxt)
            return best

        max_chain = 0
        for start in graph:
            max_chain = max(max_chain, dfs(start, {start}))
        return max_chain + 1  # nodes, not edges

    def _chain_bonus(self) -> float:
        length = self._chain_length(self.found_correlations)
        if length >= 6:
            return 0.20
        elif length >= 5:
            return 0.15
        elif length >= 4:
            return 0.10
        elif length >= 3:
            return 0.05
        return 0.0

    def _repeat_penalty(self, action_sig: str) -> float:
        if action_sig in self.prev_actions[-3:]:
            return -0.03
        return 0.0

    def track_page_view(self, page: int, visible_log_ids: List[str],
                        service_in_view: List[str]):
        self.viewed_pages.add(page)
        self.viewed_log_ids.update(visible_log_ids)
        self.services_explored.update(service_in_view)

    def calculate(
        self,
        action_type: str,
        params: dict,
        step: int,
        env_state: dict,
    ) -> dict:
        components = {}
        total = 0.0

        gt_ann = self.gt.get("annotations", {})
        gt_corr = self.gt.get("correlations", [])
        gt_sev = self.gt.get("severity", "")
        gt_findings = self.gt.get("key_findings", [])

        action_sig = f"{action_type}:{params}"
        repeat = self._repeat_penalty(action_sig)
        if repeat < 0:
            components["repeat_penalty"] = repeat
            total += repeat

        # ─── ANNOTATE ───
        if action_type == "annotate":
            log_id = params.get("log_id", "")
            category = params.get("category", "")
            self.total_annotations_submitted += 1

            if self.first_annotation_step is None:
                self.first_annotation_step = step

            if log_id in gt_ann:
                if log_id not in self.found_annotations:
                    self.found_annotations.add(log_id)
                    self.correct_annotations_submitted += 1

                    cat_sim = get_category_similarity(category, gt_ann[log_id])
                    base = 0.10
                    reward = (
                        base
                        * cat_sim
                        * self._temporal_multiplier(step)
                        * self._diff_mult
                        * self._coverage_multiplier()
                        * self._informed_bonus(log_id)
                    )
                    reward = min(reward, 0.20)
                    components["correct_annotation"] = reward
                    total += reward

                    milestone = self._check_milestones()
                    if milestone > 0:
                        components["milestone_bonus"] = milestone
                        total += milestone
                else:
                    components["duplicate_annotation"] = -0.03
                    total -= 0.03
            else:
                precision = self._running_precision()
                if precision < 0.1:
                    pen = -0.15
                elif precision < 0.3:
                    pen = -0.10
                else:
                    pen = -0.05
                components["wrong_annotation"] = pen
                total += pen

        # ─── CORRELATE ───
        elif action_type == "correlate":
            source = params.get("source_log_id", "")
            target = params.get("target_log_id", "")
            pair = (source, target)
            gt_pairs = {(c[0], c[1]) for c in gt_corr}

            if pair in gt_pairs and pair not in self.found_correlations:
                self.found_correlations.add(pair)
                reward = 0.15 * self._temporal_multiplier(step) * self._diff_mult
                reward = min(reward, 0.25)
                components["correct_correlation"] = reward
                total += reward

                chain_b = self._chain_bonus()
                old_chain = self._chain_length(self.found_correlations - {pair})
                old_bonus = 0.0
                if old_chain >= 6:
                    old_bonus = 0.20
                elif old_chain >= 5:
                    old_bonus = 0.15
                elif old_chain >= 4:
                    old_bonus = 0.10
                elif old_chain >= 3:
                    old_bonus = 0.05
                new_chain_bonus = chain_b - old_bonus
                if new_chain_bonus > 0:
                    components["chain_bonus"] = new_chain_bonus
                    total += new_chain_bonus

            elif pair in self.found_correlations:
                components["duplicate_correlation"] = -0.03
                total -= 0.03
            else:
                reversed_pair = (target, source)
                if reversed_pair in gt_pairs:
                    # Right logs, wrong direction — softer penalty to signal "fix the order"
                    components["reversed_correlation"] = -0.02
                    total -= 0.02
                else:
                    components["wrong_correlation"] = -0.05
                    total -= 0.05

        # ─── SEARCH ───
        elif action_type == "search":
            if self.first_search_step is None:
                self.first_search_step = step

            search_hits = env_state.get("search_hits", 0)
            relevant_hits = env_state.get("relevant_hits", 0)

            if search_hits > 0 and relevant_hits > 0:
                precision = relevant_hits / search_hits
                gt_remaining = len(gt_ann) - len(self.found_annotations)
                recall_gain = relevant_hits / max(gt_remaining, 1) if gt_remaining > 0 else 0
                reward = 0.02 * (1 + 3 * precision) * (1 + 2 * recall_gain)
                reward = min(reward, 0.08)
                components["useful_search"] = reward
                total += reward
            elif search_hits == 0:
                components["useless_search"] = -0.01
                total -= 0.01

        # ─── CLASSIFY ───
        elif action_type == "classify_incident":
            severity = params.get("severity", "").upper()
            levels = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
            agent_lvl = levels.get(severity, 0)
            gt_lvl = levels.get(gt_sev.upper(), 0)

            if agent_lvl == 0:
                components["invalid_severity"] = -0.05
                total -= 0.05
            else:
                distance = abs(agent_lvl - gt_lvl)
                if distance == 0:
                    components["correct_severity"] = 0.20
                    total += 0.20
                elif distance == 1:
                    components["close_severity"] = 0.05
                    total += 0.05
                else:
                    components["wrong_severity"] = -0.10
                    total -= 0.10
            self.severity_submitted = True

        # ─── DRAFT REPORT ───
        elif action_type == "draft_report":
            summary = params.get("summary", "").lower()
            matches = sum(1 for kf in gt_findings if kf.lower() in summary)
            if not self.draft_submitted:
                reward = 0.02 * matches
                reward = min(reward, 0.05)
                components["draft_reward"] = reward
                total += reward
                self.draft_submitted = True
            else:
                old_matches = env_state.get("prev_draft_matches", 0)
                new_findings = max(0, matches - old_matches)
                reward = 0.02 * new_findings
                reward = min(reward, 0.05)
                if reward > 0:
                    components["draft_improvement"] = reward
                    total += reward

        # ─── SUBMIT REPORT ───
        elif action_type == "submit_report":
            summary = params.get("summary", "").lower()
            matches = sum(1 for kf in gt_findings if kf.lower() in summary)
            coverage = matches / max(len(gt_findings), 1)

            if coverage >= 0.7:
                reward = 0.30
            elif coverage >= 0.3:
                reward = 0.30 * coverage
            else:
                reward = 0.05

            reward *= self._temporal_multiplier(step) * self._diff_mult
            reward = min(reward, 0.35)
            components["report_reward"] = reward
            total += reward

        # ─── NOOP ───
        elif action_type == "noop":
            components["noop_penalty"] = -0.02
            total -= 0.02

        # ─── NAVIGATION (filter/scroll) ───
        elif action_type in ("filter_severity", "filter_service",
                             "filter_time_range", "scroll", "clear_filters"):
            old_density = env_state.get("old_density", 0)
            new_density = env_state.get("new_density", 0)
            if new_density > old_density:
                reward = 0.015 * (new_density - old_density)
                reward = min(reward, 0.03)
                components["navigation_reward"] = reward
                total += reward

        # ─── STRATEGY BONUS ───
        if action_type == "submit_report" or (
            step >= self.max_steps and env_state.get("episode_done", False)
        ):
            strategy = 0.0
            if (self.first_search_step is not None and
                    self.first_annotation_step is not None and
                    self.first_search_step < self.first_annotation_step):
                strategy += 0.02
                components["strategy_explored_first"] = 0.02

            svc_explored = len(self.services_explored)
            expected = len(self.config.get("services", []))
            if expected > 1 and svc_explored >= expected:
                bonus = min(0.02 * (svc_explored - 1), 0.06)
                strategy += bonus
                components["strategy_multi_service"] = bonus

            if step < self.max_steps * 0.7:
                components["efficiency_bonus"] = 0.10
                strategy += 0.10

            total += strategy

        # ─── CAPS ───
        total = max(-0.15, min(0.40, total))

        self.cumulative += total
        self.cumulative = max(-1.0, min(3.0, self.cumulative))

        self.prev_actions.append(action_sig)

        # Clamp ALL returned values to strict (0, 1) for validator compliance
        _EPS = 0.001
        def _clamp(v):
            return max(_EPS, min(1.0 - _EPS, float(v)))

        return {
            "value": round(_clamp(total), 4),
            "components": {k: round(_clamp(v), 4) for k, v in components.items()},
            "cumulative": round(_clamp(self.cumulative), 4),
        }