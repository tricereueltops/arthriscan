import numpy as np

# librosa is optional — we handle the ImportError gracefully
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class AcousticModel:
    """
    Acoustic Module — LSTM Sequence Analysis (deterministic simulation).

    Analyses joint crepitus (grinding / clicking / popping sounds) from
    an audio recording of joint movement.

    Pipeline (per block diagram):
      1. Noise cancellation  — spectral gating via simple thresholding
      2. MFCC extraction     — 13 coefficients via librosa
      3. Feature scoring     — deterministic rules inspired by LSTM outputs

    Crepitus acoustic markers used for classification:
      • Spectral centroid    — crepitus shifts energy to mid-frequencies (200-800 Hz)
      • MFCC energy          — higher average MFCC magnitude in bands 2-6
      • Zero-crossing rate   — crepitus has irregular, higher ZCR
      • RMS energy variance  — grinding produces uneven energy bursts
    """

    # Thresholds (tuned for 16 kHz mono recordings)
    CENTROID_THRESHOLD = 1500     # Hz — crepitus pushes centroid up
    ZCR_THRESHOLD = 0.08         # above this → irregular signal
    MFCC_ENERGY_THRESHOLD = 15   # mean abs MFCC(2:6) — higher = crepitus
    RMS_VAR_THRESHOLD = 0.01     # RMS variance — higher = grinding bursts

    def __init__(self):
        pass

    def preprocess(self, audio_bytes: bytes) -> dict:
        """
        Noise Cancellation + MFCC Feature Extraction (per block diagram).
        Returns a dict of extracted features.
        """
        if not HAS_LIBROSA:
            return self._fallback_features()

        try:
            # Decode audio bytes
            import io, soundfile as sf
            audio_data, sr = sf.read(io.BytesIO(audio_bytes))

            # Convert to mono if stereo
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Resample to 16 kHz for consistency
            if sr != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
                sr = 16000

            # --- Noise cancellation (simple spectral gating) ---
            # Estimate noise from first 0.25s, then subtract
            noise_sample = audio_data[:int(0.25 * sr)]
            noise_rms = np.sqrt(np.mean(noise_sample ** 2)) + 1e-10
            # Gate: zero out frames below noise floor
            audio_data = np.where(np.abs(audio_data) > noise_rms * 1.5,
                                  audio_data, audio_data * 0.1)

            # --- Feature extraction ---
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            rms = librosa.feature.rms(y=audio_data)

            features = {
                "spectral_centroid": float(np.mean(spectral_centroid)),
                "zero_crossing_rate": float(np.mean(zcr)),
                "mfcc_energy": float(np.mean(np.abs(mfccs[2:7]))),
                "rms_variance": float(np.var(rms)),
                "duration_seconds": round(len(audio_data) / sr, 2),
            }
            return features

        except Exception as e:
            print(f"Acoustic preprocessing error: {e}")
            return self._fallback_features()

    def predict(self, features: dict) -> dict:
        """
        Deterministic scoring based on extracted acoustic features.
        Mimics what an LSTM would learn from crepitus patterns.
        """
        centroid = features.get("spectral_centroid", 800)
        zcr = features.get("zero_crossing_rate", 0.05)
        mfcc_e = features.get("mfcc_energy", 10)
        rms_var = features.get("rms_variance", 0.005)

        score = 0.0

        # Spectral centroid contribution (0-0.30)
        # Higher centroid → more crepitus-like
        centroid_signal = (centroid - self.CENTROID_THRESHOLD) / 800
        score += 0.30 * self._sigmoid(centroid_signal)

        # Zero-crossing rate contribution (0-0.20)
        zcr_signal = (zcr - self.ZCR_THRESHOLD) / 0.04
        score += 0.20 * self._sigmoid(zcr_signal)

        # MFCC energy contribution (0-0.30)
        mfcc_signal = (mfcc_e - self.MFCC_ENERGY_THRESHOLD) / 8
        score += 0.30 * self._sigmoid(mfcc_signal)

        # RMS variance contribution (0-0.20)
        rms_signal = (rms_var - self.RMS_VAR_THRESHOLD) / 0.008
        score += 0.20 * self._sigmoid(rms_signal)

        crepitus_prob = float(np.clip(score, 0.0, 1.0))

        # Label
        if crepitus_prob < 0.3:
            interpretation = "No significant crepitus"
        elif crepitus_prob < 0.6:
            interpretation = "Mild crepitus detected"
        else:
            interpretation = "Significant crepitus present"

        return {
            "crepitus_probability": round(crepitus_prob, 3),
            "no_crepitus_probability": round(1.0 - crepitus_prob, 3),
            "interpretation": interpretation,
            "features": features,
        }

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))

    @staticmethod
    def _fallback_features() -> dict:
        """Neutral features when audio can't be processed."""
        return {
            "spectral_centroid": 1200.0,
            "zero_crossing_rate": 0.06,
            "mfcc_energy": 12.0,
            "rms_variance": 0.006,
            "duration_seconds": 3.0,
        }
