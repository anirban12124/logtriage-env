import re
from typing import Dict, List, Set, Tuple
from src.tasks import get_category_similarity


class TaskGrader:

    def grade(
        self,
        agent_annotations: Dict[str, str],
        agent_correlations: List[List[str]],
        agent_severity: str,
        agent_report: str,
        ground_truth: dict,
        task_config: dict,
        behavior: dict,
    ) -> dict:

        gt_ann = ground_truth.get("annotations", {})
        gt_corr = ground_truth.get("correlations", [])
        gt_sev = ground_truth.get("severity", "")
        gt_findings = ground_truth.get("key_findings", [])
        weights = task_config.get("grader_weights", {})
        difficulty = task_config.get("difficulty", "easy")
        max_steps = task_config.get("max_steps", 15)

        scores = {}

        # ─── A. Annotation Precision ───
        if agent_annotations:
            correct = sum(1 for lid in agent_annotations if lid in gt_ann)
            scores["annotation_precision"] = correct / len(agent_annotations)
        else:
            scores["annotation_precision"] = 0.0 if gt_ann else 1.0

        # ─── B. Annotation Recall ───
        if gt_ann:
            found = sum(1 for lid in gt_ann if lid in agent_annotations)
            scores["annotation_recall"] = found / len(gt_ann)
        else:
            scores["annotation_recall"] = 1.0

        # ─── C. Annotation Quality ───
        correct_ids = [lid for lid in agent_annotations if lid in gt_ann]
        if correct_ids:
            qualities = []
            for lid in correct_ids:
                sim = get_category_similarity(
                    agent_annotations[lid], gt_ann[lid]
                )
                if sim >= 1.0:
                    qualities.append(1.0)
                elif sim > 0.80:
                    qualities.append(0.85)
                elif sim > 0.60:
                    qualities.append(0.60)
                elif sim > 0.40:
                    qualities.append(0.30)
                else:
                    qualities.append(0.0)
            scores["annotation_quality"] = sum(qualities) / len(qualities)
        else:
            scores["annotation_quality"] = 0.0

        # ─── D. Correlation Precision ───
        agent_pairs: Set[Tuple[str, str]] = set()
        for pair in agent_correlations:
            if len(pair) == 2:
                agent_pairs.add((pair[0], pair[1]))

        gt_pairs: Set[Tuple[str, str]] = set()
        for pair in gt_corr:
            gt_pairs.add((pair[0], pair[1]))

        if agent_pairs:
            correct_corr = len(agent_pairs & gt_pairs)
            scores["correlation_precision"] = correct_corr / len(agent_pairs)
        else:
            if not gt_pairs:
                scores["correlation_precision"] = 1.0
            else:
                scores["correlation_precision"] = 0.0

        # ─── E. Correlation Recall (with transitive credit) ───
        if gt_pairs:
            direct_found = len(gt_pairs & agent_pairs)

            # Transitive matching
            gt_graph: Dict[str, Set[str]] = {}
            for s, t in gt_pairs:
                gt_graph.setdefault(s, set()).add(t)

            transitive_credit = 0.0
            for a_s, a_t in agent_pairs:
                if (a_s, a_t) in gt_pairs:
                    continue  # already counted as direct
                # Check if there's a path from a_s to a_t in GT graph
                visited = set()
                queue = [a_s]
                found_path = False
                while queue:
                    node = queue.pop(0)
                    if node == a_t:
                        found_path = True
                        break
                    if node in visited:
                        continue
                    visited.add(node)
                    for nxt in gt_graph.get(node, set()):
                        queue.append(nxt)
                if found_path:
                    transitive_credit += 0.5

            recall = min(1.0, (direct_found + transitive_credit) / len(gt_pairs))
            scores["correlation_recall"] = recall
        else:
            scores["correlation_recall"] = 1.0

        # ─── F. Chain Reconstruction ───
        if not gt_pairs:
            scores["chain_reconstruction"] = 1.0
        else:
            # Build graphs
            def build_graph(pairs):
                g = {}
                nodes = set()
                for s, t in pairs:
                    g.setdefault(s, []).append(t)
                    nodes.add(s)
                    nodes.add(t)
                return g, nodes

            def longest_path(graph, nodes):
                def dfs(node, visited):
                    best = 0
                    for nxt in graph.get(node, []):
                        if nxt not in visited:
                            visited.add(nxt)
                            best = max(best, 1 + dfs(nxt, visited))
                            visited.remove(nxt)
                    return best

                max_len = 0
                for n in nodes:
                    max_len = max(max_len, dfs(n, {n}))
                return max_len + 1  # count nodes, not edges

            gt_graph_struct, gt_nodes = build_graph(gt_pairs)
            agent_graph_struct, agent_nodes = build_graph(agent_pairs)

            # Path score
            gt_longest = longest_path(gt_graph_struct, gt_nodes)
            agent_matching_longest = longest_path(agent_graph_struct,
                                                   agent_nodes & gt_nodes)
            path_score = agent_matching_longest / max(gt_longest, 1)
            path_score = min(1.0, path_score)

            # Coverage score
            coverage = len(agent_nodes & gt_nodes) / max(len(gt_nodes), 1)

            # Direction score
            direction_scores = []
            for a_s, a_t in agent_pairs:
                if (a_s, a_t) in gt_pairs:
                    direction_scores.append(1.0)
                elif (a_t, a_s) in gt_pairs:
                    direction_scores.append(0.5)
            direction = (sum(direction_scores) / max(len(direction_scores), 1)
                        if direction_scores else 0.0)

            scores["chain_reconstruction"] = (
                0.4 * path_score + 0.4 * coverage + 0.2 * direction
            )

        # ─── G. Severity Classification ───
        levels = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        agent_sev_clean = agent_severity.strip().upper().strip("!.,;")
        agent_lvl = levels.get(agent_sev_clean, 0)
        gt_lvl = levels.get(gt_sev.upper(), 0)

        if agent_lvl == 0:
            scores["severity_classification"] = 0.0
        else:
            distance = abs(agent_lvl - gt_lvl)
            if distance == 0:
                scores["severity_classification"] = 1.0
            elif distance == 1:
                scores["severity_classification"] = 0.5
            elif distance == 2:
                scores["severity_classification"] = 0.15
            else:
                scores["severity_classification"] = 0.0

        # ─── H. Report Completeness ───
        report_source = behavior.get("report_source", "none")
        if agent_report:
            report_lower = agent_report.lower()
            finding_scores = []
            for kf in gt_findings:
                kf_lower = kf.lower()
                # Exact substring match
                if kf_lower in report_lower:
                    finding_scores.append(1.0)
                else:
                    # Partial word matching
                    kf_words = set(kf_lower.split())
                    report_words = set(report_lower.split())
                    overlap = len(kf_words & report_words) / max(len(kf_words), 1)
                    if overlap >= 0.7:
                        finding_scores.append(0.7)
                    elif overlap >= 0.4:
                        finding_scores.append(0.4)
                    else:
                        finding_scores.append(0.0)

            completeness = sum(finding_scores) / max(len(finding_scores), 1)

            # Draft penalty
            if report_source == "draft":
                completeness *= 0.80

            scores["report_completeness"] = completeness
        else:
            scores["report_completeness"] = 0.0

        # ─── I. Report Coherence ───
        if agent_report and len(agent_report) > 10:
            # Length adequacy
            length = len(agent_report)
            length_targets = {"easy": (50, 300), "medium": (100, 600),
                             "hard": (150, 1500)}
            min_len, max_len = length_targets.get(difficulty, (50, 500))

            if length < min_len * 0.5:
                length_score = 0.0
            elif length < min_len:
                length_score = 0.3
            elif length <= max_len:
                length_score = 1.0
            elif length <= max_len * 2:
                length_score = 0.7
            else:
                length_score = 0.5

            # Temporal ordering words
            temporal_words = [
                "first", "then", "subsequently", "followed by", "after",
                "finally", "initially", "began", "leading to", "next",
                "eventually", "before", "during", "timeline",
            ]
            temporal_count = sum(
                1 for tw in temporal_words if tw in agent_report.lower()
            )
            temporal_expected = {"easy": 1, "medium": 2, "hard": 3}.get(
                difficulty, 2
            )
            temporal_score = min(1.0, temporal_count / max(temporal_expected, 1))

            # Causal language
            causal_words = [
                "caused", "resulted in", "led to", "because", "due to",
                "triggered", "propagated", "as a result", "consequently",
                "root cause", "impact", "affected",
            ]
            sentences = [s.strip() for s in re.split(r'[.!?\n]', agent_report)
                        if s.strip()]
            causal_count = sum(
                1 for s in sentences
                if any(cw in s.lower() for cw in causal_words)
            )
            causal_density = causal_count / max(len(sentences), 1)
            causal_score = min(1.0, causal_density / 0.3)

            # Structural markers
            structural = [
                "root cause", "impact", "timeline", "affected",
                "summary", "recommendation", "severity", "findings",
                ":", "-", "*", "1.", "2.",
            ]
            struct_count = sum(
                1 for sm in structural if sm in agent_report.lower()
            )
            struct_score = min(1.0, struct_count / 3)

            # Specificity
            specific_patterns = [
                r'log_\d{3}', r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
                r'\d{2}:\d{2}', r'port \d+', r'HTTP \d{3}',
                r'[a-z]+-service',
            ]
            specific_count = sum(
                1 for pat in specific_patterns
                if re.search(pat, agent_report, re.IGNORECASE)
            )
            expected_specific = {"easy": 3, "medium": 5, "hard": 8}.get(
                difficulty, 4
            )
            specific_score = min(1.0, specific_count / max(expected_specific, 1))

            coherence = (
                0.25 * length_score +
                0.25 * temporal_score +
                0.20 * causal_score +
                0.15 * struct_score +
                0.15 * specific_score
            )

            if report_source == "draft":
                coherence *= 0.80

            scores["report_coherence"] = coherence
        else:
            scores["report_coherence"] = 0.0

        # ─── J. Investigation Efficiency ───
        steps_taken = behavior.get("steps_taken", max_steps)
        step_ratio = steps_taken / max_steps

        quality_score = 0.0
        quality_weights = {k: v for k, v in weights.items()
                         if k != "investigation_efficiency"}
        total_q_weight = sum(quality_weights.values())
        if total_q_weight > 0:
            for k, w in quality_weights.items():
                quality_score += scores.get(k, 0.0) * (w / total_q_weight)

        if quality_score >= 0.7:
            if step_ratio <= 0.4:
                scores["investigation_efficiency"] = 1.0
            elif step_ratio <= 0.6:
                scores["investigation_efficiency"] = 0.85
            elif step_ratio <= 0.8:
                scores["investigation_efficiency"] = 0.65
            else:
                scores["investigation_efficiency"] = 0.50
        elif quality_score >= 0.4:
            if step_ratio <= 0.5:
                scores["investigation_efficiency"] = 0.70
            elif step_ratio <= 0.7:
                scores["investigation_efficiency"] = 0.60
            else:
                scores["investigation_efficiency"] = 0.50
        else:
            scores["investigation_efficiency"] = 0.2 * (1 - step_ratio)

        # ─── Final weighted score ───
        final = 0.0
        components = {}
        for key, weight in weights.items():
            score = scores.get(key, 0.0)
            score = max(0.0, min(1.0, score))
            weighted = score * weight
            final += weighted
            components[key] = {
                "score": round(score, 4),
                "weight": round(weight, 4),
                "weighted": round(weighted, 4),
                "detail": self._detail(key, scores, gt_ann, gt_corr,
                                       agent_annotations, agent_correlations,
                                       behavior),
            }

        return {
            "task_id": task_config["id"],
            "final_score": round(final, 4),
            "components": components,
        }

    def _detail(self, key, scores, gt_ann, gt_corr, agent_ann, agent_corr,
                behavior):
        if key == "annotation_precision":
            correct = sum(1 for lid in agent_ann if lid in gt_ann)
            return f"{correct} correct of {len(agent_ann)} submitted"
        elif key == "annotation_recall":
            found = sum(1 for lid in gt_ann if lid in agent_ann)
            return f"{found} of {len(gt_ann)} ground truth found"
        elif key == "annotation_quality":
            return f"avg category similarity {scores.get(key, 0):.2f}"
        elif key == "correlation_precision":
            return f"{scores.get(key, 0):.2f}"
        elif key == "correlation_recall":
            return f"{scores.get(key, 0):.2f}"
        elif key == "chain_reconstruction":
            return f"chain score {scores.get(key, 0):.2f}"
        elif key == "severity_classification":
            return f"score {scores.get(key, 0):.2f}"
        elif key == "report_completeness":
            src = behavior.get("report_source", "none")
            return f"completeness {scores.get(key, 0):.2f} (source: {src})"
        elif key == "report_coherence":
            return f"coherence {scores.get(key, 0):.2f}"
        elif key == "investigation_efficiency":
            steps = behavior.get("steps_taken", 0)
            return f"steps {steps}, efficiency {scores.get(key, 0):.2f}"
        return ""