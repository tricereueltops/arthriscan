from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.visual_model import VisualModel
from models.motor_model import MotorModel
from models.acoustic_model import AcousticModel
from models.fusion_model import FusionModel

app = FastAPI(
    title="ArthriScan API",
    description="Multi-Modal Arthritis Risk Stratification System",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate models once at startup
visual_model = VisualModel()
motor_model = MotorModel()
acoustic_model = AcousticModel()
fusion_model = FusionModel()


@app.post("/api/analyze")
async def analyze_data(
    motor_taps: int = Form(0),
    motor_duration: float = Form(5.0),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
):
    """
    Accepts multipart form data with:
      - motor_taps      (int)   — number of taps from the tapping test
      - motor_duration   (float) — test duration in seconds
      - image            (file)  — optional JPEG/PNG of hand joint
      - audio            (file)  — optional WAV/WebM of joint sound
    """
    print(f"[ArthriScan] Received: taps={motor_taps}, dur={motor_duration}, "
          f"image={'yes' if image else 'no'}, audio={'yes' if audio else 'no'}")

    # ===== 1. VISUAL MODULE =====
    if image and image.filename:
        image_bytes = await image.read()
        try:
            processed_img = visual_model.preprocess(image_bytes)
            visual_probs = visual_model.predict(processed_img)
        except Exception as e:
            print(f"[Visual] Error processing image: {e}")
            visual_probs = visual_model.predict_default()
    else:
        visual_probs = visual_model.predict_default()

    # ===== 2. MOTOR MODULE =====
    motor_features = motor_model.preprocess(motor_taps, motor_duration)
    motor_probs = motor_model.predict(motor_features)

    # ===== 3. ACOUSTIC MODULE =====
    if audio and audio.filename:
        audio_bytes = await audio.read()
        try:
            acoustic_features = acoustic_model.preprocess(audio_bytes)
            acoustic_probs = acoustic_model.predict(acoustic_features)
        except Exception as e:
            print(f"[Acoustic] Error processing audio: {e}")
            fallback = acoustic_model._fallback_features()
            acoustic_probs = acoustic_model.predict(fallback)
    else:
        fallback = acoustic_model._fallback_features()
        acoustic_probs = acoustic_model.predict(fallback)

    # ===== 4. FUSION (Naive Bayes) =====
    overall = fusion_model.predict(visual_probs, motor_probs, acoustic_probs)

    return {
        "visual_results": visual_probs,
        "motor_results": motor_probs,
        "acoustic_results": acoustic_probs,
        "overall_risk": overall,
    }


@app.get("/")
def root():
    return {"status": "ok", "message": "ArthriScan API is running"}
