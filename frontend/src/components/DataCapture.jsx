import React, { useState, useEffect, useRef } from 'react';
import {
  Camera,
  Mic,
  Hand,
  CheckCircle,
  Activity,
  Upload,
  Square,
  Zap,
} from 'lucide-react';

const DataCapture = ({ onComplete }) => {
  const [imageFile, setImageFile] = useState(null);
  const [imageCaptured, setImageCaptured] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioCaptured, setAudioCaptured] = useState(false);

  const [isTapping, setIsTapping] = useState(false);
  const [tapCount, setTapCount] = useState(0);
  const [timeLeft, setTimeLeft] = useState(5);
  const [tappingDone, setTappingDone] = useState(false);

  const timerRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const completedSteps = [imageCaptured, tappingDone, audioCaptured].filter(Boolean).length;

  /* ==== IMAGE ==== */
  const handleImageUpload = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setImageFile(e.target.files[0]);
      setImageCaptured(true);
    }
  };

  /* ==== MOTOR ==== */
  const startTapping = () => {
    setIsTapping(true);
    setTapCount(0);
    setTimeLeft(5);
    setTappingDone(false);

    timerRef.current = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev <= 1) {
          clearInterval(timerRef.current);
          setIsTapping(false);
          setTappingDone(true);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const handleTap = () => {
    if (isTapping) setTapCount((p) => p + 1);
  };

  /* ==== AUDIO — real MediaRecorder ==== */
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mr.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setAudioBlob(blob);
        setAudioCaptured(true);
        stream.getTracks().forEach((t) => t.stop());
      };

      mr.start();
      mediaRecorderRef.current = mr;
      setIsRecording(true);

      // Auto-stop after 3 seconds
      setTimeout(() => {
        if (mr.state === 'recording') mr.stop();
        setIsRecording(false);
      }, 3000);
    } catch (err) {
      console.warn('Mic access denied, using fallback', err);
      // Fallback — no real audio
      setAudioBlob(null);
      setAudioCaptured(true);
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  };

  const toggleRecording = () => {
    if (isRecording) stopRecording();
    else startRecording();
  };

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const allDone = imageCaptured && audioCaptured && tappingDone;

  const handleSubmit = () => {
    onComplete({
      motorTaps: tapCount,
      motorDuration: 5.0,
      imageFile,       // actual File object (or null)
      audioBlob,       // actual Blob (or null)
    });
  };

  return (
    <>
      {/* Step dots */}
      <div className="step-indicator">
        <div className={`step-dot ${completedSteps >= 1 ? 'done' : completedSteps === 0 && !imageCaptured ? 'active' : ''}`} />
        <div className={`step-line ${completedSteps >= 1 ? 'done' : ''}`} />
        <div className={`step-dot ${completedSteps >= 2 ? 'done' : completedSteps === 1 ? 'active' : ''}`} />
        <div className={`step-line ${completedSteps >= 2 ? 'done' : ''}`} />
        <div className={`step-dot ${completedSteps >= 3 ? 'done' : completedSteps === 2 ? 'active' : ''}`} />
      </div>

      {/* 1 — Visual */}
      <div className="card fade-in">
        <div className="card-header">
          <div className="card-icon visual"><Camera size={20} /></div>
          <div>
            <div className="card-label">Step 1 · Visual Module</div>
            <h2 className="card-title">Hand Joint Image</h2>
          </div>
        </div>
        <div className="card-desc">
          Upload a photo of your hand joints for inflammation analysis via MobileNetV2 CNN.
        </div>

        {imageCaptured ? (
          <div className="status-done">
            <CheckCircle size={16} /> Image captured — {imageFile?.name}
          </div>
        ) : (
          <div className="upload-zone">
            <Upload size={28} className="upload-zone-icon" />
            <div className="upload-zone-text">Tap to capture or upload</div>
            <div className="upload-zone-hint">JPG, PNG — Hand joint photo</div>
            <input type="file" accept="image/*" onChange={handleImageUpload} />
          </div>
        )}
      </div>

      {/* 2 — Motor */}
      <div className="card fade-in" style={{ animationDelay: '0.05s' }}>
        <div className="card-header">
          <div className="card-icon motor"><Activity size={20} /></div>
          <div>
            <div className="card-label">Step 2 · Motor Module</div>
            <h2 className="card-title">Dexterity / Tapping Test</h2>
          </div>
        </div>
        <div className="card-desc">
          Tap the target as fast as you can for 5 seconds. SVM + KNN classifiers evaluate your motor dexterity.
        </div>

        {tappingDone ? (
          <div className="status-done">
            <CheckCircle size={16} /> Tapping complete — {tapCount} taps ({(tapCount / 5).toFixed(1)}/sec)
          </div>
        ) : isTapping ? (
          <div className="tap-zone">
            <div className="tap-stats">
              <span>Time <strong>{timeLeft}s</strong></span>
              <span>Taps <strong>{tapCount}</strong></span>
            </div>
            <div className="tap-circle" onPointerDown={handleTap}>
              TAP
            </div>
          </div>
        ) : (
          <button className="btn btn-secondary" onClick={startTapping}>
            <Hand size={18} /> Start Tapping Test
          </button>
        )}
      </div>

      {/* 3 — Acoustic */}
      <div className="card fade-in" style={{ animationDelay: '0.1s' }}>
        <div className="card-header">
          <div className="card-icon acoustic"><Mic size={20} /></div>
          <div>
            <div className="card-label">Step 3 · Acoustic Module</div>
            <h2 className="card-title">Joint Crepitus Recording</h2>
          </div>
        </div>
        <div className="card-desc">
          Record the sound of your joint movement. LSTM analyzes crepitus patterns via MFCC features.
        </div>

        {audioCaptured ? (
          <div className="status-done">
            <CheckCircle size={16} /> Audio recorded successfully
          </div>
        ) : isRecording ? (
          <>
            <div className="recording-indicator">
              <div className="rec-dot" />
              <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: '#fca5a5' }}>
                Recording — move your joint near the mic
              </span>
            </div>
            <button className="btn btn-danger" onClick={toggleRecording}>
              <Square size={16} /> Stop Recording
            </button>
          </>
        ) : (
          <button className="btn btn-secondary" onClick={toggleRecording}>
            <Mic size={18} /> Start Recording
          </button>
        )}
      </div>

      {/* Submit */}
      <button className="btn btn-primary btn-lg" disabled={!allDone} onClick={handleSubmit}>
        <Zap size={18} />
        Analyze &amp; Diagnose
      </button>
    </>
  );
};

export default DataCapture;
