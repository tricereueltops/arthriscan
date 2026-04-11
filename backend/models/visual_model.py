import numpy as np
import cv2


class VisualModel:
    """
    Visual Module — MobileNetV2 CNN (simulated).

    In production this would run a fine-tuned MobileNetV2.  Here we
    replicate what the CNN *learns* to look for: redness, warmth,
    swelling indicators in the image of a hand joint.

    Feature set extracted from the image:
      • Redness ratio   — mean(R) / mean(B)  (inflamed joints are redder)
      • Warmth index    — mean(R) / mean(G)  (R-dominant = warm tones)
      • Color variance  — std of the HSV-S channel (swollen areas are
                          more uniformly flushed → lower saturation variance)
      • Edge density    — proportion of Canny edge pixels (swelling
                          smooths skin texture → fewer edges)
    """

    # Clinically-inspired thresholds
    REDNESS_THRESHOLD = 1.15       # R/B ratio above this suggests redness
    WARMTH_THRESHOLD = 1.10        # R/G ratio
    SATURATION_VAR_THRESHOLD = 40  # below this → uniform flush → inflammation
    EDGE_DENSITY_THRESHOLD = 0.08  # below this → smooth/swollen skin

    def __init__(self):
        pass

    # ---- preprocessing ----
    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """
        Hue Augmentation, Standardise, Resize  (per block diagram).
        Returns the preprocessed BGR image as a numpy array.
        """
        # Decode raw bytes → BGR image
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        # Resize to 224×224 (MobileNetV2 input size)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        # Standardise pixel values to 0-1 range
        img = img.astype(np.float32) / 255.0

        return img

    # ---- prediction ----
    def predict(self, img: np.ndarray) -> dict:
        """
        Analyse the preprocessed image and return inflammation probability.
        """
        # Split channels (BGR in OpenCV)
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        mean_r = np.mean(r)
        mean_g = np.mean(g)
        mean_b = np.mean(b)

        # Feature 1: Redness ratio  (higher = more inflamed)
        redness = mean_r / (mean_b + 1e-6)

        # Feature 2: Warmth index
        warmth = mean_r / (mean_g + 1e-6)

        # Feature 3: Saturation variance (need HSV)
        img_uint8 = (img * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV)
        sat_var = float(np.std(hsv[:, :, 1]))

        # Feature 4: Edge density
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / edges.size

        # ---- Scoring (sigmoid-weighted combination) ----
        score = 0.0

        # Redness contribution (0-0.35)
        redness_signal = (redness - self.REDNESS_THRESHOLD) / 0.5
        score += 0.35 * self._sigmoid(redness_signal)

        # Warmth contribution (0-0.25)
        warmth_signal = (warmth - self.WARMTH_THRESHOLD) / 0.4
        score += 0.25 * self._sigmoid(warmth_signal)

        # Saturation variance contribution (0-0.20)
        # LOW variance = inflamed (uniform flush)
        sat_signal = (self.SATURATION_VAR_THRESHOLD - sat_var) / 20
        score += 0.20 * self._sigmoid(sat_signal)

        # Edge density contribution (0-0.20)
        # LOW edges = swelling (smooth skin)
        edge_signal = (self.EDGE_DENSITY_THRESHOLD - edge_density) / 0.04
        score += 0.20 * self._sigmoid(edge_signal)

        inflamed_prob = float(np.clip(score, 0.0, 1.0))

        return {
            "inflamed_probability": round(inflamed_prob, 3),
            "non_inflamed_probability": round(1.0 - inflamed_prob, 3),
            "features": {
                "redness_ratio": round(float(redness), 3),
                "warmth_index": round(float(warmth), 3),
                "saturation_variance": round(sat_var, 2),
                "edge_density": round(edge_density, 4),
            },
        }

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))

    # ---- fallback when no real image provided ----
    def predict_default(self) -> dict:
        """Return a neutral baseline when no image is available."""
        return {
            "inflamed_probability": 0.15,
            "non_inflamed_probability": 0.85,
            "features": {
                "redness_ratio": 1.0,
                "warmth_index": 1.0,
                "saturation_variance": 50.0,
                "edge_density": 0.12,
            },
        }
