import numpy as np


class MotorModel:
    """
    Motor Module — SVM + KNN Classifier (deterministic simulation).

    Evaluates motor dexterity using finger-tapping test metrics.
    Clinical references (adapted from neurological tapping assessments):

      • Normal adults          → ≥ 5.0 taps/sec  (25+ taps in 5 s)
      • Mild impairment        → 3.0 – 5.0 taps/sec
      • Moderate impairment    → 1.5 – 3.0 taps/sec
      • Severe impairment      → < 1.5 taps/sec

    Input features (from preprocessing):
      • tapping_speed  — taps per second
      • tap_count      — raw count
      • regularity     — (placeholder) assumed from speed

    The model outputs a "poor dexterity" probability.
    """

    # Speed boundaries (taps per second)
    NORMAL_SPEED = 5.0
    MILD_SPEED = 3.0
    MODERATE_SPEED = 1.5

    def __init__(self):
        pass

    def preprocess(self, tap_count: int, duration_seconds: float) -> dict:
        """
        Normalize Tapping Speed, Scale Scores  (per block diagram).
        """
        duration_seconds = max(duration_seconds, 0.1)  # avoid /0
        speed = tap_count / duration_seconds

        return {
            "tap_count": int(tap_count),
            "duration": round(duration_seconds, 2),
            "tapping_speed": round(speed, 3),
        }

    def predict(self, features: dict) -> dict:
        """
        Deterministic classification based on tapping speed.
        Uses a piecewise-linear mapping to produce a continuous probability.
        """
        speed = features["tapping_speed"]

        if speed >= self.NORMAL_SPEED:
            # Normal range — very low risk
            poor_prob = 0.05 + 0.10 * max(0, 1 - (speed - self.NORMAL_SPEED) / 3.0)
        elif speed >= self.MILD_SPEED:
            # Mild impairment zone:  linearly from 0.15 → 0.45
            t = (self.NORMAL_SPEED - speed) / (self.NORMAL_SPEED - self.MILD_SPEED)
            poor_prob = 0.15 + 0.30 * t
        elif speed >= self.MODERATE_SPEED:
            # Moderate impairment zone:  linearly from 0.45 → 0.75
            t = (self.MILD_SPEED - speed) / (self.MILD_SPEED - self.MODERATE_SPEED)
            poor_prob = 0.45 + 0.30 * t
        else:
            # Severe impairment
            poor_prob = 0.75 + 0.20 * min(1.0, (self.MODERATE_SPEED - speed) / self.MODERATE_SPEED)

        poor_prob = float(np.clip(poor_prob, 0.0, 0.98))

        # Determine dexterity level label
        if poor_prob < 0.25:
            level = "Normal"
        elif poor_prob < 0.50:
            level = "Mild Impairment"
        elif poor_prob < 0.75:
            level = "Moderate Impairment"
        else:
            level = "Severe Impairment"

        return {
            "poor_dexterity_probability": round(poor_prob, 3),
            "normal_dexterity_probability": round(1.0 - poor_prob, 3),
            "dexterity_level": level,
            "tapping_speed": features["tapping_speed"],
        }
