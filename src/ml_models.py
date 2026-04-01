"""
ml_models.py — Shared ML model loader and embedding utilities.

Uses the pre-trained all-MiniLM-L6-v2 sentence transformer (no fine-tuning).
All embeddings are precomputed at startup and reused across sessions.

Usage:
    from src.ml_models import get_models
    models = get_models()
    sim = models.report_similarity(report_text, gt_finding_descriptions)
"""

import os
import numpy as np
from typing import List, Optional

_MODELS_INSTANCE = None


class MLModels:
    """
    Wraps the pre-trained MiniLM sentence transformer and precomputed
    embedding arrays.  Everything is CPU-only for determinism.
    """

    def __init__(self, model_path: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        # Load from local bundled path if provided (Docker), else download
        if model_path and os.path.isdir(model_path):
            print(f"[ml_models] Loading MiniLM from local path: {model_path}")
            self._model = SentenceTransformer(model_path, device="cpu")
        else:
            print("[ml_models] Downloading all-MiniLM-L6-v2 …")
            self._model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        # Precompute category embeddings and store them
        from src.tasks import CATEGORY_DESCRIPTIONS
        self._category_names = list(CATEGORY_DESCRIPTIONS.keys())
        self._category_descs = list(CATEGORY_DESCRIPTIONS.values())
        self._category_embeddings = self._embed(self._category_descs)  # (12, 384)

        # Precompute per-task finding embeddings
        from src.tasks import TASKS
        self._finding_embeddings = {}
        for task_id, task_cfg in TASKS.items():
            descs = task_cfg["ground_truth"].get("key_finding_descriptions", [])
            if descs:
                self._finding_embeddings[task_id] = self._embed(descs)  # (n, 384)

        print(f"[ml_models] Ready — "
              f"{len(self._category_names)} categories, "
              f"{len(self._finding_embeddings)} task finding sets")

    # ── Internal helpers ────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts → float32 numpy array, normalized."""
        vecs = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        return vecs

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two already-normalized vectors."""
        return float(np.clip(np.dot(a, b), 0.0, 1.0))

    # ── Public API ──────────────────────────────────────────────────

    def category_similarity(self, cat_a: str, cat_b: str) -> float:
        """
        Semantic similarity between two category names.
        Returns 1.0 for identical, 0.0–1.0 for related.
        Falls back to the static similarity map if a category is unknown.
        """
        if cat_a == cat_b:
            return 1.0

        def embed_if_needed(cat):
            if cat in self._category_names:
                idx = self._category_names.index(cat)
                return self._category_embeddings[idx]
            # Unknown category: embed its text on-the-fly
            return self._embed([cat])[0]

        va = embed_if_needed(cat_a)
        vb = embed_if_needed(cat_b)
        return self._cosine(va, vb)

    def report_completeness(
        self,
        report_text: str,
        task_id: str,
        report_source: str = "submitted",
    ) -> float:
        """
        Grader H: semantic coverage of GT key findings in the agent report.

        For each GT finding description, finds the most similar sentence
        in the report, maps it to a partial-credit score, then averages.

        Applies 20% draft penalty if report_source == 'draft'.
        """
        if not report_text or not report_text.strip():
            return 0.0

        finding_embs = self._finding_embeddings.get(task_id)
        if finding_embs is None or len(finding_embs) == 0:
            # Fallback to word-overlap if no embeddings available
            return 0.0

        # Split report into sentences (simple rule-based, deterministic)
        import re
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', report_text) if s.strip()]
        if not sentences:
            return 0.0

        # Embed report sentences
        sent_embs = self._embed(sentences)  # (n_sentences, 384)

        # For each GT finding, find best-matching sentence
        covered_scores = []
        for f_emb in finding_embs:
            sims = sent_embs @ f_emb  # cosine similarities (normalized vecs → dot product)
            best_sim = float(np.max(sims))

            if best_sim > 0.80:
                covered_scores.append(1.0)
            elif best_sim > 0.65:
                covered_scores.append(0.7)
            elif best_sim > 0.50:
                covered_scores.append(0.4)
            else:
                covered_scores.append(0.0)

        completeness = sum(covered_scores) / len(covered_scores)

        if report_source == "draft":
            completeness *= 0.80

        return round(completeness, 4)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Public access to embed arbitrary texts."""
        return self._embed(texts)


# ── Singleton accessor ───────────────────────────────────────────────

def get_models(model_path: Optional[str] = None) -> MLModels:
    """
    Returns the singleton MLModels instance.
    First call loads the model; subsequent calls return cached instance.
    """
    global _MODELS_INSTANCE
    if _MODELS_INSTANCE is None:
        _MODELS_INSTANCE = MLModels(model_path=model_path)
    return _MODELS_INSTANCE


# ── Build-time precompute helper ─────────────────────────────────────

def precompute_all_embeddings(
    model_path: Optional[str] = None,
    output_dir: str = "data/embeddings",
):
    """
    Precomputes and saves:
      - category_embeddings.npy        (12, 384)
      - category_similarity_matrix.npy (12, 12)
      - {easy,medium,hard}_finding_embeddings.npy
    Called from Dockerfile during image build.
    """
    os.makedirs(output_dir, exist_ok=True)
    models = get_models(model_path=model_path)

    # Category embeddings
    cat_emb_path = os.path.join(output_dir, "category_embeddings.npy")
    np.save(cat_emb_path, models._category_embeddings)
    print("Saved category embeddings -> " + cat_emb_path +
          " shape=" + str(models._category_embeddings.shape))

    # Category similarity matrix
    sim_matrix = models._category_embeddings @ models._category_embeddings.T
    sim_matrix = np.clip(sim_matrix, 0.0, 1.0).astype(np.float32)
    sim_path = os.path.join(output_dir, "category_similarity_matrix.npy")
    np.save(sim_path, sim_matrix)
    print("Saved category similarity matrix -> " + sim_path +
          " shape=" + str(sim_matrix.shape))

    # Per-task finding embeddings
    task_names = {"task_easy": "easy", "task_medium": "medium", "task_hard": "hard"}
    for task_id, short_name in task_names.items():
        embs = models._finding_embeddings.get(task_id)
        if embs is not None:
            path = os.path.join(output_dir, f"{short_name}_finding_embeddings.npy")
            np.save(path, embs)
            print("Saved " + short_name + " finding embeddings -> " + path + " shape=" + str(embs.shape))

    print("\nAll embeddings precomputed and saved.")


if __name__ == "__main__":
    # Run directly to precompute: python -m src.ml_models
    precompute_all_embeddings()
