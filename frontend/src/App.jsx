import React, { useState } from 'react';
import './App.css';
import DataCapture from './components/DataCapture';
import ResultsDisplay from './components/ResultsDisplay';
import { Scan } from 'lucide-react';

const LOADING_STEPS = [
  'Preprocessing visual data…',
  'Running MobileNetV2 CNN…',
  'Evaluating motor dexterity…',
  'Analyzing acoustic features…',
  'Running Naive Bayes fusion…',
];

function App() {
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [currentStep, setCurrentStep] = useState(0);

  const handleAnalysis = async (data) => {
    setIsLoading(true);
    setError('');
    setCurrentStep(0);

    // Animate through loading steps
    const stepInterval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev >= LOADING_STEPS.length - 1) {
          clearInterval(stepInterval);
          return prev;
        }
        return prev + 1;
      });
    }, 500);

    try {
      const formData = new FormData();
      formData.append('motor_taps', data.motorTaps);
      formData.append('motor_duration', data.motorDuration);

      if (data.imageFile) {
        formData.append('image', data.imageFile);
      }
      if (data.audioBlob) {
        formData.append('audio', data.audioBlob, 'recording.webm');
      }

      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Analysis failed');

      const resultData = await response.json();

      // Let the loading animation finish
      setTimeout(() => {
        clearInterval(stepInterval);
        setResults(resultData);
        setIsLoading(false);
      }, Math.max(0, LOADING_STEPS.length * 500 - 200));

    } catch (err) {
      clearInterval(stepInterval);
      console.error(err);
      setError('Cannot reach backend. Is the Python server running on :8000?');
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setResults(null);
    setCurrentStep(0);
  };

  return (
    <div className="app-shell">
      {/* Header */}
      <header className="header">
        <div className="header-logo">
          <div className="header-icon">
            <Scan size={20} color="white" />
          </div>
          <h1>ArthriScan</h1>
        </div>
        <p>Multi-Modal Arthritis Risk Stratification</p>
      </header>

      {/* Main Content */}
      <main className="content">
        {error && (
          <div className="error-banner fade-in">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5"/>
              <path d="M8 5v3.5M8 10.5h.007" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="loading-screen fade-in">
            <div className="loading-ring" />
            <div className="loading-text">Analyzing Modalities</div>
            <div className="loading-steps">
              {LOADING_STEPS.map((step, i) => (
                <div
                  key={i}
                  className={`loading-step ${i === currentStep ? 'active' : ''} ${i < currentStep ? 'done' : ''}`}
                >
                  <div className="loading-step-dot" />
                  {step}
                </div>
              ))}
            </div>
          </div>
        ) : results ? (
          <ResultsDisplay results={results} onReset={handleReset} />
        ) : (
          <DataCapture onComplete={handleAnalysis} />
        )}
      </main>
    </div>
  );
}

export default App;
