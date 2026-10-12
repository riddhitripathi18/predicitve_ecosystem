import { marked } from 'marked';
// v-fix-2 — risk stats recomputed client-side from probability records
import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, Clock, Database, FileText, Settings2, Shield, 
  AlertTriangle, CheckCircle2, RefreshCw, Plus, Trash2, 
  Download, Upload, AlertCircle, Thermometer, ShieldAlert, Cpu,
  DollarSign, TrendingUp, TrendingDown, Bot, MessageSquare, Send, Sparkles, Calculator
} from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, 
  Legend, ResponsiveContainer, ReferenceLine 
} from 'recharts';
import Gauge from './components/Gauge';

const API_BASE = '/api';

const TYPE_MAP = {
  "L (Low)": "L",
  "M (Medium)": "M",
  "H (High)": "H"
};


function MarkdownMessage({ text }) {
  if (!text) return null;
  const html = marked.parse(text, { breaks: true, gfm: true });
  return <div className="markdown-content" dangerouslySetInnerHTML={{ __html: html }} />;
}

export default function App() {
  // Navigation & Config States
  const [activeTab, setActiveTab] = useState('simulator');

  // Multi-Asset Digital Twin States
  const [activeTwin, setActiveTwin] = useState('milling'); // 'milling', 'hx', 'bearing'

  // Heat Exchanger States
  const [hxUnit, setHxUnit] = useState('E02');
  const [hxTubeIn, setHxTubeIn] = useState(150.0);
  const [hxTubeOut, setHxTubeOut] = useState(180.0);
  const [hxTubeFlow, setHxTubeFlow] = useState(50.0);
  const [hxShellIn, setHxShellIn] = useState(200.0);
  const [hxShellOut, setHxShellOut] = useState(170.0);
  const [hxShellFlow, setHxShellFlow] = useState(40.0);
  const [hxHours, setHxHours] = useState(1000.0);
  const [hxResult, setHxResult] = useState(null);
  const [hxLoading, setHxLoading] = useState(false);

  // Bearing States
  const [bearingId, setBearingId] = useState('Bearing1');
  const [bearingRms, setBearingRms] = useState(0.35);
  const [bearingKurt, setBearingKurt] = useState(3.4);
  const [bearingP2P, setBearingP2P] = useState(1.6);
  const [bearingRpm, setBearingRpm] = useState(2000);
  const [bearingHours, setBearingHours] = useState(120);
  const [bearingResult, setBearingResult] = useState(null);
  const [bearingLoading, setBearingLoading] = useState(false);

  const [opMode, setOpMode] = useState('Balanced');
  const [config, setConfig] = useState(null);
  const [activeThresh, setActiveThresh] = useState(0.35);
  const [apiError, setApiError] = useState(null);
  const [simLoading, setSimLoading] = useState(true);

  // ── TAB 1: SIMULATOR STATES ─────────────────────────────────
  const [simType, setSimType] = useState('L (Low)');
  const [simAir, setSimAir] = useState(300.0);
  const [simProc, setSimProc] = useState(310.0);
  const [simRpm, setSimRpm] = useState(1500);
  const [simTorque, setSimTorque] = useState(40.0);
  const [simWear, setSimWear] = useState(108);
  const [simResult, setSimResult] = useState(null);

  // ── TAB: FINANCIAL ROI STATES ────────────────────────────────
  const [downtimeCostHr, setDowntimeCostHr] = useState(2500);
  const [unplannedHours, setUnplannedHours] = useState(12);
  const [plannedHours, setPlannedHours] = useState(2);
  const [replacementPartCost, setReplacementPartCost] = useState(4500);
  const [plannedMaintCost, setPlannedMaintCost] = useState(800);
  const [lostUnitsHr, setLostUnitsHr] = useState(50);
  const [profitMarginUnit, setProfitMarginUnit] = useState(35);
  const [finResult, setFinResult] = useState(null);
  const [finLoading, setFinLoading] = useState(false);

  // ── TAB: LOCAL AI ASSISTANT STATES ───────────────────────────
  const [chatMessages, setChatMessages] = useState([
    {
      sender: 'assistant',
      text: '👋 Hello! I am your Local Predictive Maintenance AI Assistant powered by local telemetry diagnostics.\nAsk me anything about machine failure risk, financial loss, tool wear, heat dissipation, or maintenance actions!'
    }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);

  // ── TAB 4: BATCH ANALYZER STATES ────────────────────────────
  const [batchData, setBatchData] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [useBuiltin, setUseBuiltin] = useState(false);
  const fileInputRef = useRef(null);


  // LOAD INITIAL CONFIG & THRESHOLDS
  // ─────────────────────────────────────────────
  useEffect(() => {
    fetch(`${API_BASE}/config`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      })
      .then(data => {
        setConfig(data);
        if (data.models_loaded) {
          setActiveThresh(data.thresholds.xgb_f1);
        }
        setApiError(null);
      })
      .catch(err => {
        console.error("Error loading config:", err);
        setApiError("Could not connect to FastAPI backend API. Please make sure the backend is running (run: python main.py).");
      });
  }, []);

  // Sync threshold with operational mode preset
  useEffect(() => {
    if (!config) return;
    if (opMode === 'Balanced') {
      setActiveThresh(config.thresholds.xgb_f1);
    } else {
      setActiveThresh(config.thresholds.xgb_safety);
    }
  }, [opMode, config]);

  // ─────────────────────────────────────────────
  
  // Heat Exchanger Simulation Effect
  useEffect(() => {
    setHxLoading(true);
    const timer = setTimeout(() => {
      fetch(`${API_BASE}/simulate/hx`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          exchanger_id: hxUnit,
          tube_t_in: hxTubeIn,
          tube_t_out: hxTubeOut,
          tube_flow: hxTubeFlow,
          shell_t_in: hxShellIn,
          shell_t_out: hxShellOut,
          shell_flow: hxShellFlow,
          time_hr: hxHours
        })
      })
      .then(res => res.json())
      .then(data => { setHxResult(data); setHxLoading(false); })
      .catch(err => { console.error('HX error:', err); setHxLoading(false); });
    }, 200);
    return () => clearTimeout(timer);
  }, [hxUnit, hxTubeIn, hxTubeOut, hxTubeFlow, hxShellIn, hxShellOut, hxShellFlow, hxHours]);

  // Bearing Simulation Effect
  useEffect(() => {
    setBearingLoading(true);
    const timer = setTimeout(() => {
      fetch(`${API_BASE}/simulate/bearing`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bearing_id: bearingId,
          rms: bearingRms,
          kurtosis: bearingKurt,
          peak: bearingRms * 2.8,
          peak_to_peak: bearingP2P,
          crest_factor: 3.2,
          hours_elapsed: bearingHours,
          rpm: bearingRpm
        })
      })
      .then(res => res.json())
      .then(data => { setBearingResult(data); setBearingLoading(false); })
      .catch(err => { console.error('Bearing error:', err); setBearingLoading(false); });
    }, 200);
    return () => clearTimeout(timer);
  }, [bearingId, bearingRms, bearingKurt, bearingP2P, bearingHours, bearingRpm]);

  // RUN SIMULATOR inference
  // ─────────────────────────────────────────────
  useEffect(() => {
    setSimLoading(true);
    const timer = setTimeout(() => {
      fetch(`${API_BASE}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          air_temp: simAir,
          proc_temp: simProc,
          rpm: simRpm,
          torque: simTorque,
          tool_wear: simWear,
          type_str: simType,
          active_thresh: activeThresh
        })
      })
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      })
      .then(data => {
        setSimResult(data);
        setSimLoading(false);
        setApiError(null);
      })
      .catch(err => {
        console.error("Error running simulation:", err);
        setSimLoading(false);
        setApiError("Could not connect to FastAPI backend API. Please make sure the backend is running (run: python main.py).");
      });
    }, 250); // debounce API calls
    return () => clearTimeout(timer);
  }, [simAir, simProc, simRpm, simTorque, simWear, simType, activeThresh]);

  // ── FINANCIAL IMPACT CALCULATION EFFECT ──
  const fetchFinancialImpact = () => {
    const prob = simResult?.prediction?.probabilities?.XGBoost || 0.15;
    setFinLoading(true);
    fetch(`${API_BASE}/financial-impact`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        failure_probability: prob,
        downtime_cost_per_hour: parseFloat(downtimeCostHr),
        unplanned_downtime_hours: parseFloat(unplannedHours),
        planned_downtime_hours: parseFloat(plannedHours),
        replacement_part_cost: parseFloat(replacementPartCost),
        planned_maintenance_cost: parseFloat(plannedMaintCost),
        lost_units_per_hour: parseFloat(lostUnitsHr),
        profit_margin_per_unit: parseFloat(profitMarginUnit)
      })
    })
    .then(res => res.json())
    .then(data => {
      setFinResult(data);
      setFinLoading(false);
    })
    .catch(err => {
      console.error("Financial impact API error:", err);
      setFinLoading(false);
    });
  };

  useEffect(() => {
    fetchFinancialImpact();
  }, [simResult, downtimeCostHr, unplannedHours, plannedHours, replacementPartCost, plannedMaintCost, lostUnitsHr, profitMarginUnit]);

  // ── LOCAL LLM QUERY HANDLER ──
  const handleSendChat = (promptText) => {
    const query = promptText || chatInput;
    if (!query.trim() || chatLoading) return;

    const userMsg = { sender: 'user', text: query };
    setChatMessages(prev => [...prev, userMsg]);
    if (!promptText) setChatInput('');
    setChatLoading(true);

    const telemetryCtx = {
      rpm: simRpm,
      torque: simTorque,
      air_temp: simAir,
      proc_temp: simProc,
      tool_wear: simWear,
      type_str: simType,
      xgb_prob: simResult?.prediction?.probabilities?.XGBoost || 0.15
    };

    fetch(`${API_BASE}/llm-query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: query,
        context: telemetryCtx
      })
    })
    .then(res => res.json())
    .then(data => {
      setChatMessages(prev => [...prev, {
        sender: 'assistant',
        source: data.source,
        text: data.response
      }]);
      setChatLoading(false);
    })
    .catch(err => {
      console.error("LLM Query error:", err);
      setChatMessages(prev => [...prev, {
        sender: 'assistant',
        text: '❌ Could not connect to local AI assistant. Please check backend status.'
      }]);
      setChatLoading(false);
    });
  };




  // ─────────────────────────────────────────────
  // LOAD HEAT EXCHANGERS
  
  // ─────────────────────────────────────────────
  // BATCH processing
  // ─────────────────────────────────────────────
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    processBatchFile(file);
  };

  // Recompute risk counts from raw probability records (client-side truth)
  const computeStatsFromRecords = (records, alarmThresh) => {
    let critical = 0, warning = 0, safe = 0, alarms = 0;
    records.forEach(r => {
      const p = r["Failure Probability (%)"];
      if (p > 70)       critical++;
      else if (p >= 35) warning++;
      else              safe++;
      if (r["Predicted Failure"] === 1) alarms++;
    });
    return { total: records.length, critical, warning, safe, alarms };
  };

  const processBatchFile = (file) => {
    setBatchLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('active_thresh', activeThresh.toString());

    fetch(`${API_BASE}/batch`, {
      method: 'POST',
      body: formData
    })
    .then(async (res) => {
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({ detail: "Failed to process batch file" }));
        throw new Error(errJson.detail || `HTTP ${res.status}: Failed processing batch file`);
      }
      return res.json();
    })
    .then(data => {
      // Override backend stats with client-side recomputed values
      if (data.records && data.records.length > 0) {
        data.stats = computeStatsFromRecords(data.records, activeThresh);
      }
      setBatchData(data);
      setBatchLoading(false);
    })
    .catch(err => {
      console.error("Batch processing error:", err);
      alert(`Batch processing error: ${err.message || err}`);
      setBatchLoading(false);
    });
  };

  // Load builtin data simulation
  useEffect(() => {
    if (useBuiltin) {
      // Simulate file download & trigger API
      setBatchLoading(true);
      fetch('/ai4i2020.csv')
        .then(res => {
          if (!res.ok) throw new Error("Builtin CSV missing in public assets");
          return res.blob();
        })
        .then(blob => {
          const file = new File([blob], 'ai4i2020.csv', { type: 'text/csv' });
          processBatchFile(file);
        })
        .catch(err => {
          setUseBuiltin(false);
          setBatchLoading(false);
          alert(`Built-in simulation error: ${err.message || err}`);
        });
    }
  }, [useBuiltin]);



  return (
    <div className="layout-with-sidebar">
      {/* ── SIDEBAR PANEL ── */}
      <div className="sidebar-panel">
        <h3 style={{ marginBottom: '8px', color: '#fff', fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Cpu className="pulse-green" size={20} /> Calibration Center
        </h3>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '18px' }}>
          Calibrate alert thresholds for the facility's risk posture.
        </p>
        <hr style={{ border: '0', borderTop: '1px solid rgba(255,255,255,0.06)', margin: '14px 0' }} />
        
        <div className="form-group">
          <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '10px' }}>
            Operational Posture
          </label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '0.9rem' }}>
              <input 
                type="radio" 
                name="opMode" 
                value="Balanced" 
                checked={opMode === 'Balanced'} 
                onChange={() => setOpMode('Balanced')}
              />
              ⚖️ Balanced Mode (Optimal F1)
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '0.9rem' }}>
              <input 
                type="radio" 
                name="opMode" 
                value="Safety" 
                checked={opMode === 'Safety'} 
                onChange={() => setOpMode('Safety')}
              />
              🛡️ Safety-First (90% Recall)
            </label>
          </div>
        </div>

        <div className={`preset-banner ${opMode === 'Balanced' ? 'balanced' : 'safety'}`}>
          <span style={{ fontSize: '0.7rem', fontWeight: 600, textTransform: 'uppercase' }}>
            {opMode} Focus Preset
          </span>
          <div style={{ fontSize: '0.9rem', marginTop: '4px', fontWeight: 600 }}>
            Alarm Limit: {(activeThresh * 100).toFixed(0)}%
          </div>
        </div>

        <hr style={{ border: '0', borderTop: '1px solid rgba(255,255,255,0.06)', margin: '20px 0' }} />
        <h4 style={{ color: '#fff', fontSize: '0.9rem', marginBottom: '8px' }}>📡 System Status</h4>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <span>Primary Model: RandomForest (Constrained)</span>
          <span>Scaler Matrix: standard_scaler</span>
          <span>Sub-models: 5 Multi-Output Units</span>
        </div>
        <hr style={{ border: '0', borderTop: '1px solid rgba(255,255,255,0.06)', margin: '20px 0' }} />
        <span style={{ color: 'var(--text-dark)', fontSize: '0.75rem' }}>Confidential. Internal plant monitoring systems only.</span>
      </div>

      {/* ── MAIN LAYOUT ── */}
      <div className="main-content">
        <header className="app-header">
          <div className="header-title">
            <h1>🏭 Predictive Maintenance & Process Optimisation</h1>
            <p>
              <span className="pulse-indicator"></span> 
              STATUS: ONLINE &nbsp;|&nbsp; 
              ENGINE: RandomForest (Constrained) &nbsp;|&nbsp; 
              ACTIVE THRESHOLD: {(activeThresh * 100).toFixed(0)}%
            </p>
          </div>
        </header>

        {/* ── TABS SELECTOR ── */}
        <div className="tabs-container">
          <button className={`tab-button ${activeTab === 'simulator' ? 'active' : ''}`} onClick={() => setActiveTab('simulator')}>
            <Activity size={16} /> Digital Twin Simulator
          </button>

          <button className={`tab-button ${activeTab === 'batch' ? 'active' : ''}`} onClick={() => setActiveTab('batch')}>
            <FileText size={16} /> Fleet Batch Analyzer
          </button>

          <button className={`tab-button ${activeTab === 'financial' ? 'active' : ''}`} onClick={() => setActiveTab('financial')}>
            <DollarSign size={16} /> Financial ROI & Loss/Profit
          </button>

          <button className={`tab-button ${activeTab === 'assistant' ? 'active' : ''}`} onClick={() => setActiveTab('assistant')}>
            <Bot size={16} /> Local AI Assistant
          </button>
        </div>


        {/* Connection Error Message */}
        {apiError && (
          <div className="glass-card" style={{ borderLeft: '4px solid var(--color-critical)', padding: '24px', margin: '20px 0' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '14px' }}>
              <AlertTriangle size={32} color="var(--color-critical)" />
              <h3 style={{ color: '#fff', fontSize: '1.4rem' }}>Connection Error</h3>
            </div>
            <p style={{ color: 'var(--text-main)', fontSize: '1rem', marginBottom: '16px', lineHeight: '1.5' }}>
              {apiError}
            </p>
            <div style={{ background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '6px', fontSize: '0.85rem', fontFamily: 'monospace' }}>
              <strong>Troubleshooting tips:</strong>
              <ul style={{ paddingLeft: '20px', marginTop: '6px' }}>
                <li>Confirm that the FastAPI server is running by typing <code>python main.py</code> in the terminal.</li>
                <li>Make sure the backend port <code>8000</code> is open and reachable.</li>
                <li>Check the console logs of the browser or backend for any specific runtime exceptions.</li>
              </ul>
            </div>
            <button 
              className="tab-button active" 
              style={{ marginTop: '20px', padding: '8px 20px', cursor: 'pointer', display: 'flex', gap: '6px', alignItems: 'center' }} 
              onClick={() => {
                setApiError(null);
                setSimLoading(true);
                fetch(`${API_BASE}/config`)
                  .then(res => {
                    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
                    return res.json();
                  })
                  .then(data => {
                    setConfig(data);
                    if (data.models_loaded) {
                      setActiveThresh(data.thresholds.xgb_f1);
                    }
                  })
                  .catch(err => {
                    console.error(err);
                    setApiError("Could not connect to FastAPI backend API. Please make sure the backend is running (run: python main.py).");
                  });
              }}
            >
              <RefreshCw size={14} /> Retry Connection
            </button>
          </div>
        )}

        {/* Loading Spinner */}
        {activeTab === 'simulator' && !apiError && simLoading && !simResult && (
          <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '80px 20px', textAlign: 'center' }}>
            <RefreshCw className="pulse-green" size={48} style={{ animation: 'spin 2s linear infinite', marginBottom: '20px' }} />
            <h3 style={{ color: '#fff', marginBottom: '8px' }}>Initializing Simulator...</h3>
            <p style={{ color: 'var(--text-muted)' }}>Retrieving initial telemetry and model evaluations from the ML server.</p>
            <style>{`
              @keyframes spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
              }
            `}</style>
          </div>
        )}

        {/* ─────────────────────────────────────────────
            TAB 1: DIGITAL TWIN SIMULATOR
           ───────────────────────────────────────────── */}
        {activeTab === 'simulator' && !apiError && simResult && (
          <div>

            {/* Twin Switcher Sub-Navbar */}
            <div className="twin-subnav">
              <button className={`twin-subnav-btn ${activeTwin === 'milling' ? 'active' : ''}`} onClick={() => setActiveTwin('milling')}>
                ⚙️ CNC Milling Machine
              </button>
              <button className={`twin-subnav-btn ${activeTwin === 'hx' ? 'active' : ''}`} onClick={() => setActiveTwin('hx')}>
                🌡️ Shell & Tube Heat Exchanger
              </button>
              <button className={`twin-subnav-btn ${activeTwin === 'bearing' ? 'active' : ''}`} onClick={() => setActiveTwin('bearing')}>
                🔄 IMS Rotating Bearing Rig
              </button>
            </div>

            {/* ------------------------------------------------------------- */}
            {/* TWIN 2: HEAT EXCHANGER CONTROLS & GAUGES */}
            {/* ------------------------------------------------------------- */}
            {activeTwin === 'hx' && (
              <div>
                <div className="section-header">🌡️ Heat Exchanger Fouling & Thermal Performance Twin</div>
                <div className="grid-2">
                  <div className="glass-card">
                    <h4>🛠️ Process Stream Controls</h4>
                    <div className="form-group">
                      <label>Exchanger Unit in Preheat Train</label>
                      <select className="form-select" value={hxUnit} onChange={(e) => setHxUnit(e.target.value)}>
                        <option value="E01">E01 — Heavy Naphtha Shell Pass</option>
                        <option value="E02">E02 — Kerosene Shell Pass (High Fouling Risk)</option>
                        <option value="E03">E03 — Light Diesel Shell Pass</option>
                        <option value="E04">E04 — Light Vacuum Gas Oil (LVGO)</option>
                        <option value="E05">E05 — Heavy Diesel Shell Pass</option>
                      </select>
                    </div>

                    <div className="form-group">
                      <label>Crude Tube Inlet Temperature (°C)</label>
                      <div className="slider-container">
                        <input type="range" className="slider-input" min="100" max="220" step="1" value={hxTubeIn} onChange={(e) => setHxTubeIn(parseFloat(e.target.value))} />
                        <span className="slider-value">{hxTubeIn}°C</span>
                      </div>
                    </div>

                    <div className="form-group">
                      <label>Crude Tube Outlet Temperature (°C)</label>
                      <div className="slider-container">
                        <input type="range" className="slider-input" min="120" max="260" step="1" value={hxTubeOut} onChange={(e) => setHxTubeOut(parseFloat(e.target.value))} />
                        <span className="slider-value">{hxTubeOut}°C</span>
                      </div>
                    </div>

                    <div className="form-group">
                      <label>Crude Mass Flow Rate (kg/s)</label>
                      <div className="slider-container">
                        <input type="range" className="slider-input" min="10" max="80" step="1" value={hxTubeFlow} onChange={(e) => setHxTubeFlow(parseFloat(e.target.value))} />
                        <span className="slider-value">{hxTubeFlow} kg/s</span>
                      </div>
                    </div>

                    <div className="form-group">
                      <label>Shell Product Inlet Temperature (°C)</label>
                      <div className="slider-container">
                        <input type="range" className="slider-input" min="150" max="300" step="1" value={hxShellIn} onChange={(e) => setHxShellIn(parseFloat(e.target.value))} />
                        <span className="slider-value">{hxShellIn}°C</span>
                      </div>
                    </div>

                    <div className="form-group">
                      <label>Operational Runtime (Simulation Hours)</label>
                      <div className="slider-container">
                        <input type="range" className="slider-input" min="100" max="2000" step="25" value={hxHours} onChange={(e) => setHxHours(parseFloat(e.target.value))} />
                        <span className="slider-value">{hxHours} hrs</span>
                      </div>
                    </div>

                    <hr style={{ border: '0', borderTop: '1px solid rgba(255,255,255,0.06)', margin: '16px 0' }} />
                    <h5 style={{ marginBottom: '10px', fontSize: '0.85rem', color: '#fff' }}>⚡ Operational Stress Presets</h5>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <button className="btn btn-primary" style={{ flexGrow: 1 }} onClick={() => { setHxUnit('E01'); setHxTubeIn(120); setHxTubeOut(155); setHxHours(250); }}>
                        ✅ Clean Baseline
                      </button>
                      <button className="btn btn-danger" style={{ flexGrow: 1 }} onClick={() => { setHxUnit('E02'); setHxTubeIn(145); setHxTubeOut(162); setHxHours(1850); }}>
                        🚨 Severe Fouling (E02)
                      </button>
                    </div>
                  </div>

                  {/* HX Assessment Results Card */}
                  <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <h4 style={{ margin: 0 }}>📊 TEMA Fouling Assessment</h4>
                        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Tubular Exchanger Manufacturers Association standard</span>
                      </div>
                      <span className={`status-pill ${hxResult?.risk_level === 'CRITICAL' ? 'critical' : (hxResult?.risk_level === 'WARNING' ? 'warning' : 'safe')}`}>
                        {hxResult?.status || 'OPERATING NORMAL'}
                      </span>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px 0' }}>
                      <Gauge 
                        value={hxResult?.fouling_pct || 15} 
                        size={210} 
                        title="TEMA Fouling Resistance (%)"
                        threshold={70}
                      />
                    </div>

                    <div className="kpi-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>
                      <div className="kpi-card">
                        <span className="kpi-title">Thermal Heat Duty</span>
                        <span className="kpi-value" style={{ color: '#38bdf8' }}>{hxResult?.thermal_metrics?.duty_kW || 3150} kW</span>
                        <span className="kpi-sub">Tube ΔT: {hxResult?.thermal_metrics?.delta_t_tube_C || 30}°C</span>
                      </div>
                      <div className="kpi-card">
                        <span className="kpi-title">Fouling Resistance (Rf)</span>
                        <span className="kpi-value" style={{ color: hxResult?.risk_level === 'CRITICAL' ? 'var(--color-critical)' : '#10b981' }}>
                          {hxResult?.fouling_resistance_m2KW || 0.0003}
                        </span>
                        <span className="kpi-sub">TEMA Limit: 0.0005 m²K/W</span>
                      </div>
                      <div className="kpi-card">
                        <span className="kpi-title">Energy Penalty</span>
                        <span className="kpi-value" style={{ color: 'var(--color-warning)' }}>
                          ${hxResult?.energy_penalty_dollars_per_day || 0}
                        </span>
                        <span className="kpi-sub">Per day in furnace fuel</span>
                      </div>
                      <div className="kpi-card">
                        <span className="kpi-title">Fouling Saturation</span>
                        <span className="kpi-value" style={{ color: '#fff' }}>{hxResult?.fouling_pct || 0}%</span>
                        <span className="kpi-sub">Of maximum design margin</span>
                      </div>
                    </div>

                    <div style={{ background: 'rgba(15, 23, 42, 0.75)', border: '1px solid var(--card-border)', borderRadius: '10px', padding: '14px' }}>
                      <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '4px' }}>RECOMMENDED ACTION</div>
                      <div style={{ color: '#fff', fontSize: '0.9rem' }}>{hxResult?.recommendation || 'Continue standard condition monitoring.'}</div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ------------------------------------------------------------- */}
            {/* TWIN 3: BEARING VIBRATION CONTROLS & GAUGES */}
            {/* ------------------------------------------------------------- */}
            {activeTwin === 'bearing' && (
              <div>
                <div className="section-header">🔄 IMS Bearing Vibration & Remaining Useful Life (RUL) Twin</div>
                <div className="grid-2">
                  <div className="glass-card">
                    <h4>🛠️ Vibration Sensor Inputs</h4>
                    <div className="form-group">
                      <label>Bearing Accelerometer Channel</label>
                      <select className="form-select" value={bearingId} onChange={(e) => setBearingId(e.target.value)}>
                        <option value="Bearing1">Bearing 1 — Drive End (Test Rig #2)</option>
                        <option value="Bearing2">Bearing 2 — Non-Drive End</option>
                        <option value="Bearing3">Bearing 3 — Intermediate Support</option>
                        <option value="Bearing4">Bearing 4 — Radial Load Support</option>
                      </select>
                    </div>

                    <div className="form-group">
                      <label>Vibration RMS Acceleration (g)</label>
                      <div className="slider-container">
                        <input type="range" className="slider-input" min="0.05" max="2.8" step="0.01" value={bearingRms} onChange={(e) => setBearingRms(parseFloat(e.target.value))} />
                        <span className="slider-value">{bearingRms.toFixed(2)} g</span>
                      </div>
                    </div>

                    <div className="form-group">
                      <label>Signal Kurtosis (Impulsiveness Index)</label>
                      <div className="slider-container">
                        <input type="range" className="slider-input" min="1.5" max="15.0" step="0.1" value={bearingKurt} onChange={(e) => setBearingKurt(parseFloat(e.target.value))} />
                        <span className="slider-value">{bearingKurt.toFixed(1)}</span>
                      </div>
                    </div>

                    <div className="form-group">
                      <label>Peak-to-Peak Acceleration (g)</label>
                      <div className="slider-container">
                        <input type="range" className="slider-input" min="0.2" max="6.0" step="0.1" value={bearingP2P} onChange={(e) => setBearingP2P(parseFloat(e.target.value))} />
                        <span className="slider-value">{bearingP2P.toFixed(1)} g</span>
                      </div>
                    </div>

                    <div className="form-group">
                      <label>Spindle Rotational Speed (RPM)</label>
                      <div className="slider-container">
                        <input type="range" className="slider-input" min="1000" max="3000" step="50" value={bearingRpm} onChange={(e) => setBearingRpm(parseInt(e.target.value))} />
                        <span className="slider-value">{bearingRpm} RPM</span>
                      </div>
                    </div>

                    <div className="form-group">
                      <label>Elapsed Run-to-Failure Hours</label>
                      <div className="slider-container">
                        <input type="range" className="slider-input" min="10" max="200" step="2" value={bearingHours} onChange={(e) => setBearingHours(parseInt(e.target.value))} />
                        <span className="slider-value">{bearingHours} hrs</span>
                      </div>
                    </div>

                    <hr style={{ border: '0', borderTop: '1px solid rgba(255,255,255,0.06)', margin: '16px 0' }} />
                    <h5 style={{ marginBottom: '10px', fontSize: '0.85rem', color: '#fff' }}>⚡ Vibration Failure Presets</h5>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <button className="btn btn-primary" style={{ flexGrow: 1 }} onClick={() => { setBearingRms(0.22); setBearingKurt(3.0); setBearingP2P(0.8); }}>
                        ✅ ISO Zone A (New)
                      </button>
                      <button className="btn btn-danger" style={{ flexGrow: 1 }} onClick={() => { setBearingRms(1.95); setBearingKurt(8.4); setBearingP2P(4.5); }}>
                        🚨 Outer Race Spall (Zone D)
                      </button>
                    </div>
                  </div>

                  {/* Bearing Results Card */}
                  <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <h4 style={{ margin: 0 }}>📊 ISO 10816-3 & RUL Status</h4>
                        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Class II medium rotating machinery standard</span>
                      </div>
                      <span className={`status-pill ${bearingResult?.risk_level === 'CRITICAL' ? 'critical' : (bearingResult?.risk_level === 'WARNING' ? 'warning' : 'safe')}`}>
                        {bearingResult?.iso_zone}: {bearingResult?.iso_status}
                      </span>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px 0' }}>
                      <Gauge 
                        value={((bearingResult?.probability || 0.18) * 100).toFixed(1)} 
                        size={210} 
                        title="Bearing Failure Probability (%)"
                        threshold={40}
                      />
                    </div>

                    <div className="kpi-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>
                      <div className="kpi-card">
                        <span className="kpi-title">Predicted RUL</span>
                        <span className="kpi-value" style={{ color: bearingResult?.predicted_rul_hours < 20 ? 'var(--color-critical)' : '#38bdf8' }}>
                          {bearingResult?.predicted_rul_hours || 85} hrs
                        </span>
                        <span className="kpi-sub">Remaining Useful Life</span>
                      </div>
                      <div className="kpi-card">
                        <span className="kpi-title">Outer Race Defect (BPFO)</span>
                        <span className="kpi-value" style={{ color: '#fff' }}>
                          {bearingResult?.fault_frequencies?.['BPFO (Outer Race)'] || 119.5} Hz
                        </span>
                        <span className="kpi-sub">Inner Race (BPFI): {bearingResult?.fault_frequencies?.['BPFI (Inner Race)'] || 180.5} Hz</span>
                      </div>
                      <div className="kpi-card">
                        <span className="kpi-title">Vibration Severity</span>
                        <span className="kpi-value" style={{ color: bearingResult?.risk_level === 'CRITICAL' ? 'var(--color-critical)' : '#10b981' }}>
                          {bearingRms.toFixed(2)} g RMS
                        </span>
                        <span className="kpi-sub">Kurtosis: {bearingKurt.toFixed(1)}</span>
                      </div>
                      <div className="kpi-card">
                        <span className="kpi-title">Degradation Stage</span>
                        <span className="kpi-value" style={{ color: '#fff', fontSize: '1.05rem' }}>
                          {bearingResult?.degradation_stage || 'Healthy (Normal)'}
                        </span>
                        <span className="kpi-sub">Continuous run cycle</span>
                      </div>
                    </div>

                    <div style={{ background: 'rgba(15, 23, 42, 0.75)', border: '1px solid var(--card-border)', borderRadius: '10px', padding: '14px' }}>
                      <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '4px' }}>RECOMMENDED ACTION</div>
                      <div style={{ color: '#fff', fontSize: '0.9rem' }}>{bearingResult?.recommendation || 'Normal operation. Maintain regular vibration trending.'}</div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ------------------------------------------------------------- */}
            {/* TWIN 1: CNC MILLING MACHINE (ORIGINAL) */}
            {/* ------------------------------------------------------------- */}
            {activeTwin === 'milling' && (
              <div>
            <div className="section-header">⚙️ Telemetry Controls & AI Assessment</div>
            <div className="grid-2">
              {/* Telemetry overrides */}
              <div className="glass-card">
                <h4>🎛️ Live Parameter Override</h4>
                
                <div className="form-group">
                  <label>Component Grade Variant</label>
                  <select className="form-select" value={simType} onChange={(e) => setSimType(e.target.value)}>
                    <option>L (Low)</option>
                    <option>M (Medium)</option>
                    <option>H (High)</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Outside Intake Temperature (K)</label>
                  <div className="slider-container">
                    <input type="range" className="slider-input" min="295" max="305" step="0.1" value={simAir} onChange={(e) => setSimAir(parseFloat(e.target.value))} />
                    <span className="slider-value">{simAir.toFixed(1)} K</span>
                  </div>
                </div>

                <div className="form-group">
                  <label>Internal Chamber Temperature (K)</label>
                  <div className="slider-container">
                    <input type="range" className="slider-input" min="305" max="315" step="0.1" value={simProc} onChange={(e) => setSimProc(parseFloat(e.target.value))} />
                    <span className="slider-value">{simProc.toFixed(1)} K</span>
                  </div>
                </div>

                <div className="form-group">
                  <label>Agitator / Rotational Speed (RPM)</label>
                  <div className="slider-container">
                    <input type="range" className="slider-input" min="1168" max="2886" step="10" value={simRpm} onChange={(e) => setSimRpm(parseInt(e.target.value))} />
                    <span className="slider-value">{simRpm} RPM</span>
                  </div>
                </div>

                <div className="form-group">
                  <label>Torque Output Load (Nm)</label>
                  <div className="slider-container">
                    <input type="range" className="slider-input" min="3.8" max="76.6" step="0.1" value={simTorque} onChange={(e) => setSimTorque(parseFloat(e.target.value))} />
                    <span className="slider-value">{simTorque.toFixed(1)} Nm</span>
                  </div>
                </div>

                <div className="form-group">
                  <label>Component Run-Time Wear (min)</label>
                  <div className="slider-container">
                    <input type="range" className="slider-input" min="0" max="253" step="1" value={simWear} onChange={(e) => setSimWear(parseInt(e.target.value))} />
                    <span className="slider-value">{simWear} min</span>
                  </div>
                </div>

                <hr style={{ border: '0', borderTop: '1px solid rgba(255,255,255,0.06)', margin: '20px 0' }} />
                <h5 style={{ marginBottom: '12px', fontSize: '0.9rem', color: '#fff' }}>🧪 Stress Load Test Profiles</h5>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <button className="btn btn-danger" style={{ flexGrow: 1 }} onClick={() => { setSimAir(298.5); setSimProc(307.0); setSimRpm(1340); setSimTorque(48.0); setSimWear(50); setSimType("L (Low)"); }}>
                    🔴 Heat Lock (HDF)
                  </button>
                  <button className="btn" style={{ flexGrow: 1, borderColor: '#f59e0b', color: '#fbbf24' }} onClick={() => { setSimAir(300.0); setSimProc(310.0); setSimRpm(2700); setSimTorque(10.0); setSimWear(80); setSimType("M (Medium)"); }}>
                    🟡 High Power (PWF)
                  </button>
                  <button className="btn" style={{ flexGrow: 1, borderColor: '#10b981', color: '#34d399' }} onClick={() => { setSimAir(300.0); setSimProc(310.0); setSimRpm(1550); setSimTorque(38.0); setSimWear(80); setSimType("H (High)"); }}>
                    🟢 Stable Median
                  </button>
                </div>
              </div>

              {/* Assessment ring and status */}
              <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                <div>
                  <h4>AI Assessment Output</h4>
                  <Gauge prob={simResult.probability} activeThresh={activeThresh} />
                </div>

                <div style={{ marginTop: '16px' }}>
                  {simResult.probability >= activeThresh ? (
                    <div className="alert-critical">
                      <h2>🚨 HIGH FAILURE SHUTDOWN RISK 🚨</h2>
                      <p className="alert-desc">Anomalous load patterns exceed safe operating limit of {(activeThresh * 100).toFixed(0)}%</p>
                    </div>
                  ) : simResult.probability >= 0.35 ? (
                    <div className="alert-warning">
                      <h2>⚠️ MAINTENANCE ADVISORY</h2>
                      <p className="alert-desc">Operating threshold is warning ({(simResult.probability * 100).toFixed(1)}%). Inspections required.</p>
                    </div>
                  ) : (
                    <div className="alert-safe">
                      <h2>🟢 COMPONENT HEALTH STABLE</h2>
                      <p className="alert-desc">Nominal risk factor ({(simResult.probability * 100).toFixed(1)}%)</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Computed signals */}
            <div className="section-header">📐 Computed Operational Signals</div>
            <div className="grid-4">
              <div className="metric-card border-top" style={{ borderTopColor: '#0284c7' }}>
                <div className="metric-title">Thermodynamic Delta</div>
                <div className="metric-value-container">
                  <span className="metric-value">{simResult.temp_delta.toFixed(2)}</span>
                  <span className="metric-unit">Kelvin (K)</span>
                </div>
              </div>
              <div className="metric-card border-top" style={{ borderTopColor: simResult.power_W >= 3500 && simResult.power_W <= 9000 ? '#22c55e' : '#ef4444' }}>
                <div className="metric-title">Rotational Power</div>
                <div className="metric-value-container">
                  <span className="metric-value">{simResult.power_W.toFixed(0)}</span>
                  <span className="metric-unit">Watts (W)</span>
                </div>
              </div>
              <div className="metric-card border-top" style={{ borderTopColor: '#8b5cf6' }}>
                <div className="metric-title">Overstrain Index</div>
                <div className="metric-value-container">
                  <span className="metric-value">{simResult.torque_x_wear.toFixed(0)}</span>
                  <span className="metric-unit">N-m-Min</span>
                </div>
              </div>
              <div className="metric-card border-top" style={{ borderTopColor: '#14b8a6' }}>
                <div className="metric-title">Life Wear Factor</div>
                <div className="metric-value-container">
                  <span className="metric-value">{(simResult.wear_pct * 100).toFixed(1)}</span>
                  <span className="metric-unit">Percent (%)</span>
                </div>
              </div>
            </div>

            {/* Deviations comparison */}
            <div className="section-header">📈 Parameter Deviation Analysis (Compared to Healthy Fleet Medians)</div>
            <div className="grid-2">
              <div className="glass-card">
                {simResult.deviations.map((dev) => (
                  <div key={dev.parameter} style={{ margin: '14px 0' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                      <span>{dev.parameter}</span>
                      <span style={{ fontFamily: 'monospace' }}>
                        {dev.current} vs {dev.median} ({dev.deviation >= 0 ? '+' : ''}{dev.deviation}%)
                      </span>
                    </div>
                    <div style={{ height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden', position: 'relative' }}>
                      <div style={{
                        position: 'absolute',
                        left: '50%',
                        width: `${Math.min(50, Math.abs(dev.deviation) / 2)}%`,
                        transform: dev.deviation < 0 ? 'translateX(-100%)' : 'none',
                        height: '100%',
                        background: Math.abs(dev.deviation) > 20 ? '#ef4444' : (Math.abs(dev.deviation) > 10 ? '#f59e0b' : '#10b981'),
                        boxShadow: `0 0 8px ${Math.abs(dev.deviation) > 20 ? 'rgba(239,68,68,0.5)' : (Math.abs(dev.deviation) > 10 ? 'rgba(245,158,11,0.5)' : 'rgba(16,185,129,0.5)')}`,
                        borderRadius: '2px'
                      }}></div>
                      <div style={{ position: 'absolute', left: '50%', width: '1px', height: '100%', background: '#fff', opacity: 0.3 }}></div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="glass-card">
                <h5 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#fff', marginBottom: '14px' }}>📊 Out-of-Spec Diagnostic Flags</h5>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '0.9rem' }}>
                  {simResult.deviations.filter(d => Math.abs(d.deviation) > 10).length > 0 ? (
                    simResult.deviations.map((dev) => {
                      if (Math.abs(dev.deviation) > 20) {
                        return (
                          <div key={dev.parameter} style={{ borderLeft: '3px solid #ef4444', paddingLeft: '12px', margin: '4px 0' }}>
                            <span style={{ color: '#f87171', fontWeight: 600 }}>🛑 Critical Deviation:</span> <strong>{dev.parameter}</strong> deviates by {dev.deviation > 0 ? '+' : ''}{dev.deviation}% from baseline median.
                          </div>
                        );
                      } else if (Math.abs(dev.deviation) > 10) {
                        return (
                          <div key={dev.parameter} style={{ borderLeft: '3px solid #f59e0b', paddingLeft: '12px', margin: '4px 0' }}>
                            <span style={{ color: '#fbbf24', fontWeight: 600 }}>⚠️ Warning Deviation:</span> <strong>{dev.parameter}</strong> deviates by {dev.deviation > 0 ? '+' : ''}{dev.deviation}% from baseline.
                          </div>
                        );
                      }
                      return null;
                    })
                  ) : (
                    <div style={{ color: 'var(--text-safe)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <CheckCircle2 size={16} color="#10b981" /> All telemetry signals match historical healthy fleet baselines.
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Failure modes sub models */}
            <div className="section-header">🤖 Failure Mode Probabilities (Sub-Model Outputs)</div>
            <div className="grid-2">
              <div className="glass-card">
                {Object.entries(simResult.sub_probs).map(([mode, prob]) => {
                  const pct = prob * 100;
                  return (
                    <div key={mode} style={{ margin: '14px 0' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                        <span>{mode} Classifier</span>
                        <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{pct.toFixed(1)}%</span>
                      </div>
                      <div style={{ height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{
                          width: `${pct}%`,
                          height: '100%',
                          background: pct >= 35 ? 'linear-gradient(90deg, #b91c1c, #ef4444)' : 'linear-gradient(90deg, #1e293b, #334155)',
                          boxShadow: pct >= 35 ? '0 0 8px rgba(239, 68, 68, 0.4)' : 'none',
                          borderRadius: '4px',
                          transition: 'width 0.6s ease'
                        }}></div>
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                <div>
                  <h5 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#fff', marginBottom: '12px' }}>⚡ Sub-Model Diagnosis</h5>
                  {Object.entries(simResult.sub_probs).some(([_, p]) => p >= 0.35) ? (
                    (() => {
                      const highestMode = Object.entries(simResult.sub_probs).sort((a,b) => b[1] - a[1])[0];
                      return (
                        <div style={{ borderLeft: '3px solid #ef4444', paddingLeft: '14px' }}>
                          <p style={{ fontWeight: 600, color: '#f87171', fontSize: '0.95rem' }}>AI predicts elevated risk of {highestMode[0]} ({(highestMode[1]*100).toFixed(1)}%)</p>
                          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '8px' }}>
                            Ensure corresponding coolant feeds, lubrication systems, and torque loads are inspected.
                          </p>
                        </div>
                      );
                    })()
                  ) : (
                    <div style={{ color: 'var(--text-safe)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <CheckCircle2 size={16} color="#10b981" /> Stable state. All target failure mode sub-model probabilities are below warning limits.
                    </div>
                  )}
                </div>

                <div style={{ marginTop: '20px' }}>
                  <h5 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#fff', marginBottom: '8px' }}>💡 Facility Operator Action Guidelines</h5>
                  {simResult.guidelines.map((g, i) => (
                    <div key={i} className="insight-box" style={{ margin: '6px 0', fontSize: '0.8rem' }}>
                      {g.replace(/\*\*/g, '')}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    )}



        {/* ─────────────────────────────────────────────
            TAB 4: FLEET BATCH ANALYZER
           ───────────────────────────────────────────── */}
        {activeTab === 'batch' && !apiError && (
          <div>
            <div className="section-header">📦 Fleet Telemetry Batch Processing</div>
            
            <div className="grid-2">
              <div className="glass-card">
                <h4>CSV Stream Telemetry Uploader</h4>
                <div className="file-upload-area" onClick={() => fileInputRef.current.click()}>
                  <Upload size={32} color="var(--color-primary)" />
                  <div>
                    <span style={{ color: '#fff', fontWeight: 600 }}>Click to browse or drop CSV telemetry export</span>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '4px' }}>Columns: Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]</p>
                  </div>
                  <input type="file" ref={fileInputRef} style={{ display: 'none' }} accept=".csv" onChange={handleFileUpload} />
                </div>
                
                <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <input 
                      type="checkbox" 
                      id="builtin_chk" 
                      checked={useBuiltin} 
                      onChange={(e) => setUseBuiltin(e.target.checked)} 
                    />
                    <label htmlFor="builtin_chk" style={{ cursor: 'pointer', fontSize: '0.9rem' }}>📂 Simulate Live Factory Stream (Analyze all 10,000 active records)</label>
                  </div>

                  <div style={{ display: 'flex', gap: '8px', marginTop: '4px', flexWrap: 'wrap' }}>
                    <a 
                      href="/ai4i2020_train_imbalanced.csv" 
                      download="ai4i2020_train_imbalanced.csv"
                      style={{ fontSize: '0.78rem', color: '#f59e0b', textDecoration: 'none', background: 'rgba(245,158,11,0.1)', padding: '4px 8px', borderRadius: '4px', border: '1px solid rgba(245,158,11,0.3)' }}
                    >
                      📥 Download Training Dataset (8,000 Rows)
                    </a>
                    <a 
                      href="/ai4i2020_test_real.csv" 
                      download="ai4i2020_test_real.csv"
                      style={{ fontSize: '0.78rem', color: '#60a5fa', textDecoration: 'none', background: 'rgba(59,130,246,0.1)', padding: '4px 8px', borderRadius: '4px', border: '1px solid rgba(59,130,246,0.3)' }}
                    >
                      📥 Download Test Dataset (2,000 Rows)
                    </a>
                    <a 
                      href="/sanity_test_100.csv" 
                      download="sanity_test_100.csv"
                      style={{ fontSize: '0.78rem', color: '#34d399', textDecoration: 'none', background: 'rgba(16,185,129,0.1)', padding: '4px 8px', borderRadius: '4px', border: '1px solid rgba(16,185,129,0.3)' }}
                    >
                      📥 Download Sanity 100 CSV
                    </a>
                  </div>
                </div>
              </div>

              {batchData && batchData.stats && (
                <div className="glass-card">
                  <h4>Batch Process Results</h4>
                  <div className="grid-5" style={{ gap: '10px' }}>
                    <div className="metric-card">
                      <div className="metric-title">Fleet Units</div>
                      <div className="metric-value-container">
                        <span className="metric-value">{batchData.stats.total}</span>
                      </div>
                    </div>
                    <div className="metric-card" style={{ borderTop: '3px solid #ef4444' }}>
                      <div className="metric-title">Critical Risks</div>
                      <div className="metric-value-container">
                        <span className="metric-value" style={{ color: '#ef4444' }}>{batchData.stats.critical}</span>
                      </div>
                    </div>
                    <div className="metric-card" style={{ borderTop: '3px solid #f59e0b' }}>
                      <div className="metric-title">Warning Risks</div>
                      <div className="metric-value-container">
                        <span className="metric-value" style={{ color: '#f59e0b' }}>{batchData.stats.warning}</span>
                      </div>
                    </div>
                    <div className="metric-card" style={{ borderTop: '3px solid #10b981' }}>
                      <div className="metric-title">Stable Units</div>
                      <div className="metric-value-container">
                        <span className="metric-value" style={{ color: '#10b981' }}>{batchData.stats.safe}</span>
                      </div>
                    </div>
                    <div className="metric-card" style={{ borderTop: '3px solid #3b82f6' }}>
                      <div className="metric-title">Triggered Alarms</div>
                      <div className="metric-value-container">
                        <span className="metric-value" style={{ color: '#60a5fa' }}>{batchData.stats.alarms}</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Live accuracy metrics section */}
            {batchData && batchData.evaluation && (
              <div>
                <div className="section-header">📐 Model Accuracy on Your Uploaded Data</div>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '14px', marginTop: '-6px' }}>
                  Metrics computed against the <strong>Machine failure</strong> ground-truth labels in your file — not the training dataset.
                </p>
                <div className="grid-7">
                  <div className="metric-card border-top" style={{ borderTopColor: '#3b82f6' }}>
                    <div className="metric-title">Accuracy</div>
                    <div className="metric-value-container">
                      <span className="metric-value">{(batchData.evaluation.accuracy * 100).toFixed(2)}</span>
                      <span className="metric-unit">%</span>
                    </div>
                  </div>
                  <div className="metric-card border-top" style={{ borderTopColor: batchData.evaluation.precision >= 0.5 ? '#10b981' : '#f59e0b' }}>
                    <div className="metric-title">Precision</div>
                    <div className="metric-value-container">
                      <span className="metric-value">{batchData.evaluation.precision.toFixed(4)}</span>
                    </div>
                  </div>
                  <div className="metric-card border-top" style={{ borderTopColor: batchData.evaluation.recall >= 0.7 ? '#10b981' : '#f59e0b' }}>
                    <div className="metric-title">Recall</div>
                    <div className="metric-value-container">
                      <span className="metric-value">{batchData.evaluation.recall.toFixed(4)}</span>
                    </div>
                  </div>
                  <div className="metric-card border-top" style={{ borderTopColor: batchData.evaluation.f1_score >= 0.5 ? '#10b981' : '#f59e0b' }}>
                    <div className="metric-title">F1-Score</div>
                    <div className="metric-value-container">
                      <span className="metric-value">{batchData.evaluation.f1_score.toFixed(4)}</span>
                    </div>
                  </div>
                  <div className="metric-card border-top" style={{ borderTopColor: batchData.evaluation.pr_auc >= 0.6 ? '#10b981' : '#f59e0b' }}>
                    <div className="metric-title">PR-AUC</div>
                    <div className="metric-value-container">
                      <span className="metric-value">{batchData.evaluation.pr_auc.toFixed(4)}</span>
                    </div>
                  </div>
                  <div className="metric-card border-top" style={{ borderTopColor: '#f59e0b' }}>
                    <div className="metric-title">False Pos</div>
                    <div className="metric-value-container">
                      <span className="metric-value">{batchData.evaluation.false_positives}</span>
                      <span className="metric-unit">rows</span>
                    </div>
                  </div>
                  <div className="metric-card border-top" style={{ borderTopColor: '#ef4444' }}>
                    <div className="metric-title">False Neg</div>
                    <div className="metric-value-container">
                      <span className="metric-value">{batchData.evaluation.false_negatives}</span>
                      <span className="metric-unit">rows</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Process list table */}
            {batchData && batchData.records && (
              <div>
                <div className="section-header">🚨 Fleet Risk Status & Warning List</div>
                <div className="table-container" style={{ maxHeight: '400px' }}>
                                    {/* Asset Category Banner */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '14px', background: 'rgba(15, 23, 42, 0.6)', padding: '10px 16px', borderRadius: '10px', border: '1px solid var(--card-border)' }}>
                    <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Detected Stream Asset:</span>
                    <span style={{ background: 'var(--color-primary)', color: '#fff', padding: '3px 10px', borderRadius: '6px', fontSize: '0.85rem', fontWeight: 600 }}>
                      {batchData.stats.asset_category || 'Industrial Fleet'}
                    </span>
                    <span style={{ fontSize: '0.85rem', color: '#fff', marginLeft: 'auto' }}>
                      Evaluated: <strong>{batchData.stats.total.toLocaleString()} rows</strong>
                    </span>
                  </div>

                  <table className="styled-table">
                    <thead>
                      <tr>
                        {batchData.stats.dataset_type === 'heat_exchanger' ? (
                          <>
                            <th>Row #</th>
                            <th>Crude API</th>
                            <th>Crude TAN</th>
                            <th>E01 Tube In (°C)</th>
                            <th>E01 Tube Out (°C)</th>
                            <th>E02 Tube In (°C)</th>
                            <th>E02 Tube Out (°C)</th>
                            <th>Fouling Rf (m²K/W)</th>
                            <th>Failure Probability (%)</th>
                            <th>Risk Level</th>
                          </>
                        ) : batchData.stats.dataset_type === 'bearing' ? (
                          <>
                            <th>Sample #</th>
                            <th>Test ID</th>
                            <th>Timestamp</th>
                            <th>Bearing RMS (g)</th>
                            <th>Kurtosis</th>
                            <th>ISO Severity</th>
                            <th>Failure Probability (%)</th>
                            <th>Risk Level</th>
                          </>
                        ) : (
                          <>
                            <th>UDI</th>
                            <th>Type</th>
                            <th>Air Temp [K]</th>
                            <th>Process Temp [K]</th>
                            <th>Rotational Speed [rpm]</th>
                            <th>Torque [Nm]</th>
                            <th>Tool Wear [min]</th>
                            <th>Failure Probability (%)</th>
                            <th>Risk Level</th>
                          </>
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {batchData.records.slice(0, 500).map((r, idx) => (
                        <tr key={idx}>
                          {batchData.stats.dataset_type === 'heat_exchanger' ? (
                            <>
                              <td style={{ color: '#fff', fontWeight: 600 }}>{idx + 1}</td>
                              <td>{r.Crude_API || '32.0'}</td>
                              <td>{r.Crude_TAN || '0.8'}</td>
                              <td>{r.E01_Crude_Tube_T_In_degC ?? r['E01_Crude_Tube_T_In_degC'] ?? '-'}</td>
                              <td>{r.E01_Crude_Tube_T_Out_degC ?? r['E01_Crude_Tube_T_Out_degC'] ?? '-'}</td>
                              <td>{r.E02_Crude_Tube_T_In_degC ?? r['E02_Crude_Tube_T_In_degC'] ?? '-'}</td>
                              <td>{r.E02_Crude_Tube_T_Out_degC ?? r['E02_Crude_Tube_T_Out_degC'] ?? '-'}</td>
                              <td style={{ color: '#38bdf8' }}>{r['Fouling Resistance (m²K/W)'] ?? '0.00032'}</td>
                              <td style={{ fontWeight: 600, color: '#fff' }}>{r["Failure Probability (%)"]}%</td>
                              <td>
                                <span className={`status-pill ${
                                  r["Risk Level"]?.includes('CRITICAL') ? 'critical' : (r["Risk Level"]?.includes('WARNING') ? 'warning' : 'safe')
                                }`}>
                                  {r["Risk Level"]}
                                </span>
                              </td>
                            </>
                          ) : batchData.stats.dataset_type === 'bearing' ? (
                            <>
                              <td style={{ color: '#fff', fontWeight: 600 }}>{idx + 1}</td>
                              <td>{r.test_id || 2}</td>
                              <td style={{ fontSize: '0.78rem' }}>{r.timestamp || `T+${idx * 10}m`}</td>
                              <td>{Number(r.Bearing1_x_rms || r.Bearing1_rms || r.rms || 0.35).toFixed(3)}</td>
                              <td>{Number(r.Bearing1_x_kurtosis || r.Bearing1_kurtosis || r.kurtosis || 3.2).toFixed(2)}</td>
                              <td style={{ color: '#38bdf8' }}>{r['ISO Severity'] || 'Zone B (Normal)'}</td>
                              <td style={{ fontWeight: 600, color: '#fff' }}>{r["Failure Probability (%)"]}%</td>
                              <td>
                                <span className={`status-pill ${
                                  r["Risk Level"]?.includes('CRITICAL') ? 'critical' : (r["Risk Level"]?.includes('WARNING') ? 'warning' : 'safe')
                                }`}>
                                  {r["Risk Level"]}
                                </span>
                              </td>
                            </>
                          ) : (
                            <>
                              <td style={{ color: '#fff', fontWeight: 600 }}>{r.UDI || idx + 1}</td>
                              <td>{r.Type}</td>
                              <td>{r["Air temperature [K]"]}</td>
                              <td>{r["Process temperature [K]"]}</td>
                              <td>{r["Rotational speed [rpm]"]}</td>
                              <td>{r["Torque [Nm]"]}</td>
                              <td>{r["Tool wear [min]"]}</td>
                              <td style={{ fontWeight: 600, color: '#fff' }}>{r["Failure Probability (%)"]}%</td>
                              <td>
                                <span className={`status-pill ${
                                  r["Risk Level"]?.includes('CRITICAL') ? 'critical' : (r["Risk Level"]?.includes('WARNING') ? 'warning' : 'safe')
                                }`}>
                                  {r["Risk Level"]?.split(' ').slice(1).join(' ') || r["Risk Level"]}
                                </span>
                              </td>
                            </>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {batchData.records.length > 500 && (
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '8px', textAlign: 'center' }}>
                    Displaying top 500 of {batchData.records.length.toLocaleString()} fleet records for smooth performance.
                  </p>
                )}
              </div>
            )}

            {batchLoading && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '40px', gap: '10px' }}>
                <RefreshCw size={24} className="pulse-green" style={{ animation: 'spin 1.5s infinite linear' }} />
                <span>Running batch analysis over fleet logs...</span>
                <style>{`@keyframes spin { 100% { transform: rotate(360deg); } }`}</style>
              </div>
            )}
          </div>
        )}

        {/* ── TAB 3: FINANCIAL ROI & LOSS/PROFIT IMPACT CALCULATOR ── */}
        {activeTab === 'financial' && !apiError && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <div className="glass-card" style={{ padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div>
                  <h2 style={{ fontSize: '1.4rem', color: '#fff', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <DollarSign color="var(--color-primary)" /> Financial Loss & Profit ROI Estimator
                  </h2>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', marginTop: '4px' }}>
                    Calculate financial consequences of machine failure vs. proactive maintenance based on current ML probability.
                  </p>
                </div>
                {finResult && (
                  <span className={`status-pill ${finResult.net_savings > 0 && finResult.failure_probability >= 0.35 ? 'critical' : 'safe'}`}>
                    {finResult.recommendation}
                  </span>
                )}
              </div>

              {/* Financial Metrics Row */}
              {finResult && (
                <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px', marginBottom: '24px' }}>
                  <div className="metric-card" style={{ borderLeft: '4px solid var(--color-critical)' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Expected Failure Loss</div>
                    <div style={{ fontSize: '1.8rem', fontWeight: '700', color: 'var(--color-critical)', margin: '4px 0' }}>
                      ${finResult.expected_failure_loss.toLocaleString()}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-dark)' }}>
                      Risk-Weighted ({(finResult.failure_probability * 100).toFixed(1)}% prob)
                    </div>
                  </div>

                  <div className="metric-card" style={{ borderLeft: '4px solid var(--color-primary)' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Preventive Repair Cost</div>
                    <div style={{ fontSize: '1.8rem', fontWeight: '700', color: 'var(--color-primary)', margin: '4px 0' }}>
                      ${finResult.planned_maintenance_cost.toLocaleString()}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-dark)' }}>
                      Scheduled downtime ({plannedHours} hrs) + parts
                    </div>
                  </div>

                  <div className="metric-card" style={{ borderLeft: `4px solid ${finResult.net_savings > 0 ? 'var(--color-safe)' : 'var(--text-muted)'}` }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Net Profit Saved</div>
                    <div style={{ fontSize: '1.8rem', fontWeight: '700', color: finResult.net_savings > 0 ? 'var(--color-safe)' : '#fff', margin: '4px 0' }}>
                      ${finResult.net_savings.toLocaleString()}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-dark)' }}>
                      ROI: <strong style={{ color: 'var(--color-safe)' }}>+{finResult.roi_pct}%</strong> profit preserved
                    </div>
                  </div>

                  <div className="metric-card">
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Max Unplanned Breakdown Impact</div>
                    <div style={{ fontSize: '1.8rem', fontWeight: '700', color: '#f87171', margin: '4px 0' }}>
                      ${finResult.unplanned_breakdown_cost.toLocaleString()}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-dark)' }}>
                      If machine suffers total catastrophic failure
                    </div>
                  </div>
                </div>
              )}

              {/* Controls Sliders Grid */}
              <h3 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '16px', borderTop: '1px solid var(--card-border)', paddingTop: '16px' }}>
                ⚙️ Operational Financial Parameters
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
                
                <div>
                  <label style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-main)', fontSize: '0.85rem', marginBottom: '6px' }}>
                    <span>Downtime Cost / Hour:</span>
                    <strong style={{ color: 'var(--color-primary)' }}>${downtimeCostHr.toLocaleString()}/hr</strong>
                  </label>
                  <input 
                    type="range" min="500" max="10000" step="250" 
                    value={downtimeCostHr} 
                    onChange={e => setDowntimeCostHr(e.target.value)} 
                    style={{ width: '100%', accentColor: 'var(--color-primary)' }} 
                  />
                </div>

                <div>
                  <label style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-main)', fontSize: '0.85rem', marginBottom: '6px' }}>
                    <span>Unplanned Breakdown Duration:</span>
                    <strong style={{ color: 'var(--color-critical)' }}>{unplannedHours} hours</strong>
                  </label>
                  <input 
                    type="range" min="1" max="48" step="1" 
                    value={unplannedHours} 
                    onChange={e => setUnplannedHours(e.target.value)} 
                    style={{ width: '100%', accentColor: 'var(--color-critical)' }} 
                  />
                </div>

                <div>
                  <label style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-main)', fontSize: '0.85rem', marginBottom: '6px' }}>
                    <span>Replacement Part Cost:</span>
                    <strong style={{ color: '#fff' }}>${replacementPartCost.toLocaleString()}</strong>
                  </label>
                  <input 
                    type="range" min="500" max="25000" step="500" 
                    value={replacementPartCost} 
                    onChange={e => setReplacementPartCost(e.target.value)} 
                    style={{ width: '100%', accentColor: 'var(--color-warning)' }} 
                  />
                </div>

                <div>
                  <label style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-main)', fontSize: '0.85rem', marginBottom: '6px' }}>
                    <span>Lost Units Produced / Hour:</span>
                    <strong style={{ color: '#fff' }}>{lostUnitsHr} units/hr</strong>
                  </label>
                  <input 
                    type="range" min="0" max="200" step="10" 
                    value={lostUnitsHr} 
                    onChange={e => setLostUnitsHr(e.target.value)} 
                    style={{ width: '100%', accentColor: 'var(--color-primary)' }} 
                  />
                </div>

                <div>
                  <label style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-main)', fontSize: '0.85rem', marginBottom: '6px' }}>
                    <span>Profit Margin per Unit:</span>
                    <strong style={{ color: 'var(--color-safe)' }}>${profitMarginUnit}/unit</strong>
                  </label>
                  <input 
                    type="range" min="5" max="150" step="5" 
                    value={profitMarginUnit} 
                    onChange={e => setProfitMarginUnit(e.target.value)} 
                    style={{ width: '100%', accentColor: 'var(--color-safe)' }} 
                  />
                </div>

                <div>
                  <label style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-main)', fontSize: '0.85rem', marginBottom: '6px' }}>
                    <span>Scheduled Maintenance Downtime:</span>
                    <strong style={{ color: 'var(--color-safe)' }}>{plannedHours} hours</strong>
                  </label>
                  <input 
                    type="range" min="0.5" max="8" step="0.5" 
                    value={plannedHours} 
                    onChange={e => setPlannedHours(e.target.value)} 
                    style={{ width: '100%', accentColor: 'var(--color-safe)' }} 
                  />
                </div>

              </div>
            </div>
          </div>
        )}

        {/* ── TAB 4: LOCAL AI MACHINE ASSISTANT ── */}
        {activeTab === 'assistant' && !apiError && (
          <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px', minHeight: '650px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--card-border)', paddingBottom: '14px' }}>
              <div>
                <h2 style={{ fontSize: '1.4rem', color: '#fff', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Bot color="var(--color-safe)" /> Local AI Machine Assistant & Advisory
                </h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', marginTop: '4px' }}>
                  Powered by <strong>Qwen2.5 3B LLM</strong> running locally via Ollama. Query machine health, diagnose failure modes, and calculate financial ROI.
                </p>
              </div>
              <div style={{ background: 'rgba(16, 185, 129, 0.15)', border: '1px solid var(--color-safe)', padding: '6px 14px', borderRadius: '20px', fontSize: '0.8rem', color: 'var(--color-safe)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Sparkles size={14} /> Telemetry Context Live
              </div>
            </div>

            {/* Live Telemetry Context Strip */}
            <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', background: 'rgba(0,0,0,0.25)', padding: '10px 16px', borderRadius: '8px', fontSize: '0.82rem', fontFamily: 'var(--font-mono)', border: '1px solid var(--card-border)' }}>
              <span>RPM: <strong style={{ color: '#fff' }}>{simRpm}</strong></span>
              <span>•</span>
              <span>Torque: <strong style={{ color: '#fff' }}>{simTorque} Nm</strong></span>
              <span>•</span>
              <span>Air Temp: <strong style={{ color: '#fff' }}>{simAir} K</strong></span>
              <span>•</span>
              <span>Process Temp: <strong style={{ color: '#fff' }}>{simProc} K</strong></span>
              <span>•</span>
              <span>Tool Wear: <strong style={{ color: '#fff' }}>{simWear} min</strong></span>
              <span>•</span>
              <span>XGB Risk: <strong style={{ color: (simResult?.prediction?.probabilities?.XGBoost || 0.15) >= 0.35 ? 'var(--color-critical)' : 'var(--color-safe)' }}>{((simResult?.prediction?.probabilities?.XGBoost || 0.15)*100).toFixed(1)}%</strong></span>
            </div>

            {/* Quick Prompt Chips */}
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              <button 
                onClick={() => handleSendChat("Calculate financial loss if this machine suffers an unplanned failure")} 
                style={{ background: 'rgba(59, 130, 246, 0.12)', border: '1px solid rgba(59, 130, 246, 0.3)', color: '#60a5fa', padding: '6px 12px', borderRadius: '16px', fontSize: '0.8rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}>
                💰 Calculate Financial Loss
              </button>

              <button 
                onClick={() => handleSendChat("Why is tool wear high and what action should I take?")} 
                style={{ background: 'rgba(245, 158, 11, 0.12)', border: '1px solid rgba(245, 158, 11, 0.3)', color: '#fbbf24', padding: '6px 12px', borderRadius: '16px', fontSize: '0.8rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}>
                🛠️ Tool Wear Diagnostics
              </button>

              <button 
                onClick={() => handleSendChat("Is current torque and RPM within safe mechanical limits?")} 
                style={{ background: 'rgba(16, 185, 129, 0.12)', border: '1px solid rgba(16, 185, 129, 0.3)', color: '#34d399', padding: '6px 12px', borderRadius: '16px', fontSize: '0.8rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}>
                ⚡ Torque & Mechanical Strain
              </button>

              <button 
                onClick={() => handleSendChat("Give me a full predictive maintenance recommendation report")} 
                style={{ background: 'rgba(239, 68, 68, 0.12)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#f87171', padding: '6px 12px', borderRadius: '16px', fontSize: '0.8rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}>
                📋 Maintenance Summary Report
              </button>
            </div>

            {/* Chat Box Conversation History */}
            <div style={{ flex: '1', minHeight: '350px', maxHeight: '500px', overflowY: 'auto', background: 'rgba(6, 9, 16, 0.6)', border: '1px solid var(--card-border)', borderRadius: '10px', padding: '16px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
              {chatMessages.map((msg, idx) => (
                <div key={idx} style={{ alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start', maxWidth: '85%' }}>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-dark)', marginBottom: '4px', textAlign: msg.sender === 'user' ? 'right' : 'left' }}>
                    {msg.sender === 'user' ? 'You (Operator)' : (msg.source || 'Local AI Assistant')}
                  </div>
                  <div style={{
                    background: msg.sender === 'user' ? 'var(--color-primary)' : 'rgba(30, 41, 59, 0.85)',
                    color: '#fff',
                    padding: '12px 16px',
                    borderRadius: '12px',
                    borderTopLeftRadius: msg.sender === 'user' ? '12px' : '2px',
                    borderTopRightRadius: msg.sender === 'user' ? '2px' : '12px',
                    fontSize: '0.9rem',
                    lineHeight: '1.5',
                    whiteSpace: 'pre-wrap',
                    border: msg.sender === 'user' ? 'none' : '1px solid var(--card-border)'
                  }}>
                    {msg.sender === 'user' ? (
                      <div style={{ whiteSpace: 'pre-wrap' }}>{msg.text}</div>
                    ) : (
                      <MarkdownMessage text={msg.text} />
                    )}
                  </div>
                </div>
              ))}
              {chatLoading && (
                <div style={{ alignSelf: 'flex-start', background: 'rgba(30, 41, 59, 0.5)', padding: '10px 14px', borderRadius: '12px', fontSize: '0.85rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <RefreshCw size={14} style={{ animation: 'spin 1.5s infinite linear' }} />
                  Thinking and querying local LLM...
                </div>
              )}
            </div>

            {/* Chat Input Bar */}
            <form onSubmit={e => { e.preventDefault(); handleSendChat(); }} style={{ display: 'flex', gap: '10px' }}>
              <input 
                type="text" 
                placeholder="Ask about machine status, failure risks, tool wear, financial impact..." 
                value={chatInput} 
                onChange={e => setChatInput(e.target.value)} 
                style={{ flex: '1', background: 'rgba(15, 23, 42, 0.8)', border: '1px solid var(--card-border)', borderRadius: '8px', padding: '12px 16px', color: '#fff', fontSize: '0.92rem', outline: 'none' }}
              />
              <button 
                type="submit" 
                disabled={chatLoading || !chatInput.trim()} 
                className="tab-button active" 
                style={{ padding: '0 24px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', opacity: (chatLoading || !chatInput.trim()) ? 0.5 : 1 }}>
                <Send size={16} /> Ask LLM
              </button>
            </form>
          </div>
        )}
      </div>

    </div>
  );
}
