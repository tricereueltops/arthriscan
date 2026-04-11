import React from 'react';
import { ArrowLeft, Eye, Activity, AudioLines, Layers, Info } from 'lucide-react';

const ResultsDisplay = ({ results, onReset }) => {
  const { overall_risk, visual_results, motor_results, acoustic_results } = results;

  const riskKey = overall_risk.risk_level.split(' ')[0].toLowerCase();

  const visualPct = (overall_risk.fusion_details.visual_inflamed_contribution * 100).toFixed(1);
  const motorPct = (overall_risk.fusion_details.motor_poor_dexterity_contribution * 100).toFixed(1);
  const acousticPct = (overall_risk.fusion_details.acoustic_crepitus_contribution * 100).toFixed(1);

  return (
    <>
      {/* Risk Hero Card */}
      <div className={`risk-hero ${riskKey} fade-in`}>
        <div className="risk-hero-label">Naive Bayes Fusion Result</div>
        <div className="risk-hero-level">{overall_risk.risk_level}</div>
        <div className="risk-hero-score">
          Fused Probability: {(overall_risk.risk_score * 100).toFixed(1)}%
        </div>
      </div>

      {/* Fusion Breakdown Bars */}
      <div className="card fade-in" style={{ animationDelay: '0.08s' }}>
        <div className="card-header">
          <div className="card-icon fusion"><Layers size={20} /></div>
          <div>
            <div className="card-label">Log-Odds Fusion</div>
            <h3 className="card-title">Modality Contributions</h3>
          </div>
        </div>
        <div className="card-desc">
          Each modality's &ldquo;risk&rdquo; probability weighted via Naive Bayes (Visual 0.45, Motor 0.30, Acoustic 0.25).
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.25rem' }}>
          <div className="bar-row">
            <span className="bar-label">Visual</span>
            <div className="bar-track">
              <div className="bar-fill visual" style={{ width: `${visualPct}%` }} />
            </div>
            <span className="bar-value">{visualPct}%</span>
          </div>
          <div className="bar-row">
            <span className="bar-label">Motor</span>
            <div className="bar-track">
              <div className="bar-fill motor" style={{ width: `${motorPct}%` }} />
            </div>
            <span className="bar-value">{motorPct}%</span>
          </div>
          <div className="bar-row">
            <span className="bar-label">Acoustic</span>
            <div className="bar-track">
              <div className="bar-fill acoustic" style={{ width: `${acousticPct}%` }} />
            </div>
            <span className="bar-value">{acousticPct}%</span>
          </div>
        </div>
      </div>

      {/* Module Details */}
      <div className="card fade-in" style={{ animationDelay: '0.16s' }}>
        <div className="card-header">
          <div className="card-icon visual"><Eye size={20} /></div>
          <div>
            <div className="card-label">Module Outputs</div>
            <h3 className="card-title">Individual Analysis</h3>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {/* Visual */}
          <div className="module-chip">
            <div className="module-chip-left">
              <div className="module-chip-dot" style={{ background: '#60a5fa' }} />
              <div>
                <div className="module-chip-name">Inflamed</div>
                <div className="module-chip-model">MobileNetV2 CNN</div>
              </div>
            </div>
            <div className="module-chip-value">
              {(visual_results.inflamed_probability * 100).toFixed(1)}%
            </div>
          </div>

          {/* Motor */}
          <div className="module-chip">
            <div className="module-chip-left">
              <div className="module-chip-dot" style={{ background: '#f472b6' }} />
              <div>
                <div className="module-chip-name">
                  {motor_results.dexterity_level || 'Poor Dexterity'}
                </div>
                <div className="module-chip-model">
                  SVM + KNN · {motor_results.tapping_speed != null
                    ? `${motor_results.tapping_speed} taps/sec`
                    : 'Classifier'}
                </div>
              </div>
            </div>
            <div className="module-chip-value">
              {(motor_results.poor_dexterity_probability * 100).toFixed(1)}%
            </div>
          </div>

          {/* Acoustic */}
          <div className="module-chip">
            <div className="module-chip-left">
              <div className="module-chip-dot" style={{ background: '#c084fc' }} />
              <div>
                <div className="module-chip-name">
                  {acoustic_results.interpretation || 'Crepitus'}
                </div>
                <div className="module-chip-model">LSTM Sequence Analysis</div>
              </div>
            </div>
            <div className="module-chip-value">
              {(acoustic_results.crepitus_probability * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>

      {/* Extracted Features (if available) */}
      {(visual_results.features || acoustic_results.features) && (
        <div className="card fade-in" style={{ animationDelay: '0.24s' }}>
          <div className="card-header">
            <div className="card-icon fusion"><Info size={20} /></div>
            <div>
              <div className="card-label">Raw Features</div>
              <h3 className="card-title">Extracted Biomarkers</h3>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.8125rem' }}>
            {visual_results.features && (
              <>
                <div className="module-chip">
                  <span className="module-chip-name">Redness Ratio (R/B)</span>
                  <span className="module-chip-value" style={{ fontSize: '0.8125rem' }}>
                    {visual_results.features.redness_ratio}
                  </span>
                </div>
                <div className="module-chip">
                  <span className="module-chip-name">Warmth Index (R/G)</span>
                  <span className="module-chip-value" style={{ fontSize: '0.8125rem' }}>
                    {visual_results.features.warmth_index}
                  </span>
                </div>
                <div className="module-chip">
                  <span className="module-chip-name">Edge Density</span>
                  <span className="module-chip-value" style={{ fontSize: '0.8125rem' }}>
                    {visual_results.features.edge_density}
                  </span>
                </div>
              </>
            )}
            {acoustic_results.features && (
              <>
                <div className="module-chip">
                  <span className="module-chip-name">Spectral Centroid</span>
                  <span className="module-chip-value" style={{ fontSize: '0.8125rem' }}>
                    {typeof acoustic_results.features.spectral_centroid === 'number'
                      ? acoustic_results.features.spectral_centroid.toFixed(1) + ' Hz'
                      : acoustic_results.features.spectral_centroid}
                  </span>
                </div>
                <div className="module-chip">
                  <span className="module-chip-name">Zero-Crossing Rate</span>
                  <span className="module-chip-value" style={{ fontSize: '0.8125rem' }}>
                    {typeof acoustic_results.features.zero_crossing_rate === 'number'
                      ? acoustic_results.features.zero_crossing_rate.toFixed(4)
                      : acoustic_results.features.zero_crossing_rate}
                  </span>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Reset */}
      <button className="btn btn-ghost" onClick={onReset} style={{ marginTop: '0.5rem' }}>
        <ArrowLeft size={16} /> Perform New Assessment
      </button>
    </>
  );
};

export default ResultsDisplay;
