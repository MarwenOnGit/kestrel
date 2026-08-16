"""Signal classification (Doc 05 §1).

Two implementations behind one interface:

  * StubClassifier - simulator only. Derives a prediction from ground truth via
    a confusion model, so the rest of the pipeline can be developed and tested
    long before a trained network exists.
  * TensorRTClassifier - the real one. Placeholder until Step 4 of the plan.

Both obey the rule that matters: below MIN_CONFIDENCE the answer is "unknown",
and unknown is never authorized for capture.
"""
from __future__ import annotations

import numpy as np

from kestrel.common.messages import Detection

MIN_CONFIDENCE = 0.85

CLASSES = ["fm_voice", "am_voice", "dmr", "tetra", "p25", "lte", "nr5g",
           "lora", "wifi", "bluetooth", "ocusync", "satcom_terminal", "unknown"]


class StubClassifier:
    """Simulates a classifier with realistic, SNR-dependent behaviour.

    Accuracy degrades at low SNR exactly as a real model's does - this matters,
    because a stub that is always right would let downstream bugs hide.
    """

    model_version = "stub-0.1"

    def __init__(self, seed: int = 0, accuracy_at_high_snr: float = 0.95,
                 confusable: dict[str, list[str]] | None = None):
        self.rng = np.random.default_rng(seed)
        self.acc = accuracy_at_high_snr
        self.confusable = confusable or {
            "dmr": ["tetra", "p25"], "tetra": ["dmr", "p25"], "p25": ["dmr", "tetra"],
            "fm_voice": ["am_voice"], "am_voice": ["fm_voice"],
            "lte": ["nr5g"], "nr5g": ["lte"],
            "wifi": ["ocusync"], "ocusync": ["wifi"],
        }

    def classify(self, det: Detection, iq: np.ndarray | None = None
                 ) -> tuple[str, float]:
        truth = det.truth_class
        if truth is None:
            return "unknown", 0.0

        # P(correct) ramps with SNR: poor below 10 dB, near-asymptotic above 20.
        snr = det.snr_db
        p_correct = self.acc / (1.0 + np.exp(-(snr - 10.0) / 3.0))

        if self.rng.random() < p_correct:
            predicted = truth
            confidence = float(np.clip(self.rng.normal(0.93, 0.04), 0.5, 0.999))
        else:
            alts = self.confusable.get(truth, []) + ["unknown"]
            predicted = str(self.rng.choice(alts))
            confidence = float(np.clip(self.rng.normal(0.70, 0.12), 0.2, 0.95))

        if confidence < MIN_CONFIDENCE:
            predicted = "unknown"
        return predicted, confidence


class TensorRTClassifier:
    """Real inference path. Implemented in Step 4 of the plan (Doc 11)."""

    model_version = "unimplemented"

    def __init__(self, engine_path: str):
        raise NotImplementedError(
            "Step 4 of docs/11-phase1-implementation-plan.md. "
            "Use StubClassifier until a trained model exists.")


def apply_classification(det: Detection, classifier,
                         iq: np.ndarray | None = None) -> Detection:
    """Classify in place. `iq` is the narrowband snippet for this detection;
    the stub ignores it, a real model cannot work without it."""
    cls, conf = classifier.classify(det, iq)
    det.signal_class = cls
    det.class_confidence = conf
    det.model_version = getattr(classifier, "model_version", "unknown")
    return det
