import numpy as np
import math


class FusionModel:
    """
    Naive Bayes Fusion Module (per block diagram).

    Combines probabilities from three independent modalities using a
    Naive Bayes-inspired approach:

        P(High | V,M,A) ∝ P(V|High) · P(M|High) · P(A|High) · P(High)

    Since we already have per-module "risk" probabilities (not raw
    likelihoods conditioned on class), we use a log-odds fusion:

        1. Convert each module probability to log-odds
        2. Apply per-module weights (visual evidence is strongest,
           motor is secondary, acoustic supplements)
        3. Sum the weighted log-odds  (+prior log-odds)
        4. Convert back to probability
        5. Map to 3-level stratification

    Weights reflect clinical significance per the literature:
        Visual (inflammation)  : 0.45  — most direct indicator
        Motor  (dexterity)     : 0.30  — functional impairment
        Acoustic (crepitus)    : 0.25  — supplementary evidence

    Risk thresholds:
        fused_prob < 0.35   → Low Risk
        0.35 ≤ prob < 0.65  → Moderate Risk
        prob ≥ 0.65         → High Risk
    """

    WEIGHTS = {
        "visual": 0.45,
        "motor": 0.30,
        "acoustic": 0.25,
    }

    # Prior probability of arthritis (base rate ~ 20% in target population)
    PRIOR = 0.20

    # Final stratification thresholds
    THRESHOLD_LOW = 0.35
    THRESHOLD_HIGH = 0.65

    def __init__(self):
        pass

    def predict(self, visual_probs: dict, motor_probs: dict, acoustic_probs: dict) -> dict:
        """
        Fuse three modality probabilities into a single risk stratification.

        P_rob = P_c + P_cn  (per diagram — combined probability)
        """
        inflamed = visual_probs.get("inflamed_probability", 0.15)
        poor_dex = motor_probs.get("poor_dexterity_probability", 0.15)
        crepitus = acoustic_probs.get("crepitus_probability", 0.15)

        # Clamp to avoid log(0)
        eps = 1e-6
        probs = {
            "visual": np.clip(inflamed, eps, 1 - eps),
            "motor": np.clip(poor_dex, eps, 1 - eps),
            "acoustic": np.clip(crepitus, eps, 1 - eps),
        }

        # --- Log-odds fusion ---
        prior_log_odds = self._log_odds(self.PRIOR)

        weighted_log_odds = prior_log_odds
        for key, prob in probs.items():
            w = self.WEIGHTS[key]
            # Each module contributes its log-odds scaled by its weight
            module_log_odds = self._log_odds(prob)
            weighted_log_odds += w * module_log_odds

        # Convert back to probability
        fused_prob = self._inv_log_odds(weighted_log_odds)
        fused_prob = float(np.clip(fused_prob, 0.0, 1.0))

        # --- 3-Level Stratification ---
        if fused_prob < self.THRESHOLD_LOW:
            risk_level = "Low Risk"
        elif fused_prob < self.THRESHOLD_HIGH:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        return {
            "risk_score": round(fused_prob, 3),
            "risk_level": risk_level,
            "fusion_method": "Naive Bayes (log-odds fusion)",
            "weights": self.WEIGHTS,
            "prior": self.PRIOR,
            "fusion_details": {
                "visual_inflamed_contribution": round(inflamed, 3),
                "motor_poor_dexterity_contribution": round(poor_dex, 3),
                "acoustic_crepitus_contribution": round(crepitus, 3),
            },
        }

    @staticmethod
    def _log_odds(p: float) -> float:
        """Convert probability to log-odds."""
        return float(math.log(p / (1.0 - p)))

    @staticmethod
    def _inv_log_odds(lo: float) -> float:
        """Convert log-odds back to probability."""
        return float(1.0 / (1.0 + math.exp(-lo)))
