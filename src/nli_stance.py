"""NLI-based stance detection using roberta-large-mnli.

Splits long articles into overlapping token chunks (chunk_size=400, stride=200)
to handle RoBERTa's 512-token hard limit.  Aggregates chunk-level scores via
mean(entailment_prob - contradiction_prob) to produce a single article-level
stance score in [-1, +1].
"""

import json
import logging
from typing import List, Dict

import numpy as np

logger = logging.getLogger(__name__)


def _score_to_label(score: float) -> str:
    """Map a [-1, +1] stance score to a stance label string."""
    if score > 0.6:
        return "strongly_agree"
    elif score > 0.2:
        return "agree"
    elif score >= -0.2:
        return "neutral"
    elif score >= -0.6:
        return "disagree"
    else:
        return "strongly_disagree"


class NLIStanceAnalyzer:
    """Stance analyzer backed by roberta-large-mnli (NLI).

    Label IDs from the model:  CONTRADICTION=0, NEUTRAL=1, ENTAILMENT=2.
    Per-chunk score = entailment_prob - contradiction_prob  → [-1, +1].
    Article score   = mean(chunk scores).
    Confidence      = mean(max(entailment, contradiction) per chunk).
    """

    MODEL_NAME = "roberta-large-mnli"
    CHUNK_SIZE = 400    # tokens per chunk (leaves room for hypothesis + specials)
    CHUNK_STRIDE = 200  # overlap between chunks

    def __init__(self):
        self._tokenizer = None
        self._model = None
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        logger.info(f"Loading NLI model: {self.MODEL_NAME} …")
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self._model.to(self.device)
        self._model.eval()
        logger.info(f"NLI model loaded on {self.device}")

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk_tokens(self, token_ids: List[int], hypothesis_len: int) -> List[List[int]]:
        """Split token_ids into overlapping windows.

        Each window fits within 512 tokens when combined with:
          [CLS] hypothesis [SEP] [SEP] chunk [SEP]  →  3 special + hypothesis_len + chunk_len ≤ 512
        """
        max_chunk = 512 - hypothesis_len - 3
        max_chunk = max(1, max_chunk)

        stride = min(self.CHUNK_STRIDE, max_chunk)
        step = min(self.CHUNK_SIZE, max_chunk)

        chunks = []
        start = 0
        while start < len(token_ids):
            end = min(start + step, len(token_ids))
            chunks.append(token_ids[start:end])
            if end == len(token_ids):
                break
            start += stride

        return chunks if chunks else [token_ids[:max_chunk]]

    # ------------------------------------------------------------------
    # Single-article prediction
    # ------------------------------------------------------------------

    def predict(self, premise: str, hypothesis: str) -> Dict:
        """Run NLI stance detection for one (premise, hypothesis) pair.

        Returns a dict with:
            stance_score  : float in [-1, +1]
            stance_label  : str
            confidence    : float in [0, 1]
            reasoning     : str (diagnostic string)
        """
        import torch

        self._load_model()

        # Tokenise full premise and hypothesis separately (no truncation yet)
        hyp_ids = self._tokenizer.encode(
            hypothesis, add_special_tokens=False, truncation=False
        )
        prem_ids = self._tokenizer.encode(
            premise, add_special_tokens=False, truncation=False
        )

        chunks = self._chunk_tokens(prem_ids, len(hyp_ids))

        chunk_scores = []
        chunk_confidences = []
        entailment_means = []
        neutral_means = []
        contradiction_means = []

        with torch.no_grad():
            for chunk_ids in chunks:
                # Build token-type inputs manually so we can pass raw lists
                inputs = self._tokenizer(
                    hypothesis,
                    self._tokenizer.decode(chunk_ids),
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=False,
                ).to(self.device)

                logits = self._model(**inputs).logits[0]
                probs = torch.softmax(logits, dim=0).cpu().numpy()

                # CONTRADICTION=0, NEUTRAL=1, ENTAILMENT=2
                contradiction_prob = float(probs[0])
                neutral_prob = float(probs[1])
                entailment_prob = float(probs[2])

                chunk_scores.append(entailment_prob - contradiction_prob)
                chunk_confidences.append(max(entailment_prob, contradiction_prob))
                entailment_means.append(entailment_prob)
                neutral_means.append(neutral_prob)
                contradiction_means.append(contradiction_prob)

        stance_score = float(np.mean(chunk_scores))
        confidence = float(np.mean(chunk_confidences))
        stance_label = _score_to_label(stance_score)

        n = len(chunks)
        reasoning = (
            f"entailment={np.mean(entailment_means):.3f} "
            f"neutral={np.mean(neutral_means):.3f} "
            f"contradiction={np.mean(contradiction_means):.3f} "
            f"(mean over {n} chunk{'s' if n != 1 else ''})"
        )

        return {
            "stance_score": stance_score,
            "stance_label": stance_label,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    # ------------------------------------------------------------------
    # Batch prediction (multiple premises, one hypothesis)
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        premises: List[str],
        hypothesis: str,
        batch_size: int = 8,
    ) -> List[Dict]:
        """Run stance detection for multiple premises against one hypothesis.

        Args:
            premises:   List of article texts (title + content).
            hypothesis: The claim text.
            batch_size: Number of *chunks* to process in one GPU forward pass.
                        (Not the same as number of articles.)

        Returns:
            List of result dicts (same shape as predict()), one per premise.
        """
        self._load_model()

        import torch

        # Tokenise hypothesis once
        hyp_ids = self._tokenizer.encode(
            hypothesis, add_special_tokens=False, truncation=False
        )

        results = []

        for premise in premises:
            prem_ids = self._tokenizer.encode(
                premise, add_special_tokens=False, truncation=False
            )
            chunks = self._chunk_tokens(prem_ids, len(hyp_ids))

            all_chunk_scores = []
            all_chunk_confidences = []
            all_entailment = []
            all_neutral = []
            all_contradiction = []

            # Process chunks in mini-batches
            for start in range(0, len(chunks), batch_size):
                mini_batch = chunks[start: start + batch_size]
                chunk_texts = [self._tokenizer.decode(c) for c in mini_batch]

                inputs = self._tokenizer(
                    [hypothesis] * len(chunk_texts),
                    chunk_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    logits = self._model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()

                for prob_row in probs:
                    contradiction_prob = float(prob_row[0])
                    neutral_prob = float(prob_row[1])
                    entailment_prob = float(prob_row[2])

                    all_chunk_scores.append(entailment_prob - contradiction_prob)
                    all_chunk_confidences.append(max(entailment_prob, contradiction_prob))
                    all_entailment.append(entailment_prob)
                    all_neutral.append(neutral_prob)
                    all_contradiction.append(contradiction_prob)

            stance_score = float(np.mean(all_chunk_scores))
            confidence = float(np.mean(all_chunk_confidences))
            stance_label = _score_to_label(stance_score)

            n = len(chunks)
            reasoning = (
                f"entailment={np.mean(all_entailment):.3f} "
                f"neutral={np.mean(all_neutral):.3f} "
                f"contradiction={np.mean(all_contradiction):.3f} "
                f"(mean over {n} chunk{'s' if n != 1 else ''})"
            )

            results.append({
                "stance_score": stance_score,
                "stance_label": stance_label,
                "confidence": confidence,
                "reasoning": reasoning,
            })

        return results
