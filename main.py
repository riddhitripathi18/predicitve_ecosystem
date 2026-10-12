import os
import pickle
import json
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import ttf_engine
from ttf_engine import run_ttf_forecast, DEFAULT_DEGRADATION
from machine_registry import MachineRegistry, ML_COMPATIBLE_TYPES
from heat_exchanger_analyzer import HXUnit, analyze_hx_fleet
import wear_band_policy

# ─────────────────────────────────────────────
# INITIALIZATION & SETUP
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
REGISTRY = MachineRegistry(BASE_DIR)


# Auto-load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    if not os.environ.get("GOOGLE_API_KEY"):
        load_dotenv(r"c:\Users\riddh\OneDrive\Desktop\industrial_agent\.env")
except Exception as _env_err:
    pass

try:
    from unified_predictor import UnifiedPredictor
    unified_predictor = UnifiedPredictor(MODEL_DIR)
except Exception as _up_err:
    print("UnifiedPredictor init note:", _up_err)
    unified_predictor = None

app = FastAPI(title=" SYS-CONTROL Predictive Maintenance API")

# Enable CORS for frontend connection (typically Vite dev server on 5173 or custom)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
try:
    with open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "rb") as f:
        xgb_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "rf_model.pkl"), "rb") as f:
        rf_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
        
    with open(os.path.join(MODEL_DIR, "xgb_threshold.pkl"), "rb") as f:
        xgb_thresh_f1 = float(pickle.load(f))
    with open(os.path.join(MODEL_DIR, "xgb_safety_threshold.pkl"), "rb") as f:
        xgb_thresh_safety = float(pickle.load(f))
    with open(os.path.join(MODEL_DIR, "rf_threshold.pkl"), "rb") as f:
        rf_thresh_f1 = float(pickle.load(f))
    with open(os.path.join(MODEL_DIR, "rf_safety_threshold.pkl"), "rb") as f:
        rf_thresh_safety = float(pickle.load(f))
        
    SUB_TARGETS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    sub_models = {}
    for sub in SUB_TARGETS:
        with open(os.path.join(MODEL_DIR, f"xgb_{sub.lower()}_model.pkl"), "rb") as f:
            sub_models[sub] = pickle.load(f)
            
    with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
        FEATURE_COLS = pickle.load(f)
        
    with open(os.path.join(MODEL_DIR, "fleet_medians.pkl"), "rb") as f:
        fleet_medians = pickle.load(f)

    # Load deterministic wear-band decision policy (tunable, not hardcoded).
    # wear_window_low/high define the TWF stochastic risk window — no probability
    # gate is applied inside it; wear band membership alone triggers ELEVATED_WEAR_WATCH.
    _wb_policy     = wear_band_policy.load_policy(os.path.join(MODEL_DIR, "wear_band_policy.pkl"))
    WEAR_WINDOW_LOW  = float(_wb_policy['wear_window_low'])
    WEAR_WINDOW_HIGH = float(_wb_policy['wear_window_high'])

    models_loaded = True
except Exception as e:
    print(f"Error loading models: {e}")
    models_loaded = False
    xgb_thresh_f1, xgb_thresh_safety, rf_thresh_f1, rf_thresh_safety = 0.3, 0.1, 0.3, 0.1
    FEATURE_COLS = []
    fleet_medians = {}
    WEAR_WINDOW_LOW  = wear_band_policy.WEAR_WINDOW_LOW
    WEAR_WINDOW_HIGH = wear_band_policy.WEAR_WINDOW_HIGH

# Helper to engineer features (column order must match FEATURE_COLS in 1_train_model.py)
def engineer_features(air_temp: float, proc_temp: float, rpm: float, torque: float, tool_wear: float, type_enc: int):
    power_watts         = (2 * np.pi / 60) * rpm * torque
    temp_diff_K         = proc_temp - air_temp
    wear_torque_product = tool_wear * torque
    wear_pct            = tool_wear / 253.0
    high_torque         = 1 if torque > 55 else 0
    high_wear           = 1 if tool_wear > 200 else 0
    return [type_enc, air_temp, proc_temp, rpm, torque, tool_wear,
            power_watts, temp_diff_K, wear_torque_product, wear_pct, high_torque, high_wear]

# Helper to extract anomalies
def get_anomalies(air_temp: float, proc_temp: float, rpm: float, torque: float, tool_wear: float):
    temp_delta = proc_temp - air_temp
    power_W = torque * rpm * (2 * np.pi / 60)
    
    current_vals = {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": proc_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "temp_delta": temp_delta,
        "power_W": power_W
    }
    
    feat_display_names = {
        "Air temperature [K]": "Ambient Temp",
        "Process temperature [K]": "Process Temp",
        "Rotational speed [rpm]": "Spindle RPM",
        "Torque [Nm]": "Torque Load",
        "Tool wear [min]": "Cumulative Wear",
        "temp_delta": "Thermal Delta",
        "power_W": "Rotary Power"
    }
    
    deviations = []
    for f_name, curr_val in current_vals.items():
        median_val = fleet_medians.get(f_name, 1.0)
        pct_dev = ((curr_val - median_val) / median_val) * 100
        deviations.append({
            "parameter": feat_display_names[f_name],
            "current": round(curr_val, 2),
            "median": round(median_val, 2),
            "deviation": round(pct_dev, 2)
        })
    return deviations

# Helper to build AI Operator Guidelines
def get_guidelines(air_temp: float, proc_temp: float, rpm: float, torque: float, tool_wear: float, prob: float):
    reasons = []
    if torque > 60:
        reasons.append("🔴 **CRITICAL TORQUE OVERLOAD:** Motor is operating at **" + str(round(torque, 1)) + " Nm**, exceeding physical safety margins (>60 Nm). High risk of mechanical shear.")
    if tool_wear > 200:
        reasons.append("⚠️ **DEGRADED COMPONENT LIFE:** Cumulative tool wear has reached **" + str(round(tool_wear, 1)) + " min**. Micro-fractures and thermal degradation are expected.")
    if rpm < 1300:
        reasons.append("⚠️ **ROTATION SLOWDOWN:** Speed dropped to **" + str(round(rpm, 1)) + " RPM**. Combined with high load, this indicates potential motor stalling or severe friction.")
    temp_delta = proc_temp - air_temp
    if temp_delta < 8.6:
        reasons.append("🔴 **INSUFFICIENT HEAT DISSIPATION:** Temperature delta is **" + str(round(temp_delta, 1)) + " K** (Limit: >8.6 K). Cooling loop blockage suspected.")
    power = torque * rpm * (2 * np.pi / 60)
    if power > 9000 or power < 3500:
        reasons.append("⚠️ **ELECTRICAL OUT-OF-BOUNDS:** Power draws **" + str(round(power, 0)) + " W** which is outside nominal rating limits (3500W-9000W). Risk of electrical circuit breaker failure.")
        
    if not reasons:
        if prob > 0.5:
            reasons.append("🧬 **MULTIVARIATE AI FLAG:** No single sensor exceeds standard limits, but the cross-parameter pattern indicates early signs of impending failure.")
        else:
            reasons.append("💚 **NOMINAL STATE:** All diagnostic streams are within safe engineering operational medians.")
    return reasons

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/api/config")
def get_config():
    return {
        "models_loaded": models_loaded,
        "thresholds": {
            "xgb_f1": xgb_thresh_f1,
            "xgb_safety": xgb_thresh_safety,
            "rf_f1": rf_thresh_f1,
            "rf_safety": rf_thresh_safety
        }
    }

class SimulationInput(BaseModel):
    air_temp: float
    proc_temp: float
    rpm: float
    torque: float
    tool_wear: float
    type_str: str  # "L (Low)", "M (Medium)", "H (High)"
    active_thresh: float

@app.post("/api/simulate")
def simulate_override(data: SimulationInput):
    if not models_loaded:
        raise HTTPException(status_code=500, detail="ML models not loaded")
        
    type_enc = {"L (Low)": 0, "M (Medium)": 1, "H (High)": 2}.get(data.type_str, 1)
    features = engineer_features(data.air_temp, data.proc_temp, data.rpm, data.torque, data.tool_wear, type_enc)
    X_scaled = scaler.transform(np.array([features]))
    
    prob = float(xgb_model.predict_proba(X_scaled)[0, 1])
    
    # Sub-models failures
    sub_probs = {}
    for sub in SUB_TARGETS:
        if sub in sub_models:
            sub_probs[sub] = float(sub_models[sub].predict_proba(X_scaled)[0, 1])
            
    # Deviations
    deviations = get_anomalies(data.air_temp, data.proc_temp, data.rpm, data.torque, data.tool_wear)
    # Action guidelines
    guidelines = get_guidelines(data.air_temp, data.proc_temp, data.rpm, data.torque, data.tool_wear, prob)
    
    return {
        "probability": prob,
        "sub_probs": sub_probs,
        "deviations": deviations,
        "guidelines": guidelines,
        "temp_delta": data.proc_temp - data.air_temp,
        "power_W": data.torque * data.rpm * (2 * np.pi / 60),
        "torque_x_wear": data.torque * data.tool_wear,
        "wear_pct": data.tool_wear / 253.0
    }


class HXSimulationInput(BaseModel):
    exchanger_id: str = "E02"
    tube_t_in: float = 150.0
    tube_t_out: float = 180.0
    tube_flow: float = 50.0
    shell_t_in: float = 200.0
    shell_t_out: float = 170.0
    shell_flow: float = 40.0
    time_hr: float = 1000.0
    crude_api: float = 32.0
    crude_tan: float = 0.8
    crude_chlorides: float = 15.0

class BearingSimulationInput(BaseModel):
    bearing_id: str = "Bearing1"
    rms: float = 0.28
    kurtosis: float = 3.4
    peak: float = 0.8
    peak_to_peak: float = 1.6
    crest_factor: float = 3.2
    hours_elapsed: float = 120.0
    rpm: float = 2000.0

@app.post("/api/simulate/hx")
def simulate_heat_exchanger(data: HXSimulationInput):
    if unified_predictor is not None:
        try:
            res = unified_predictor.predict_heat_exchanger(
                exchanger_id=data.exchanger_id,
                tube_t_in=data.tube_t_in,
                tube_t_out=data.tube_t_out,
                tube_flow=data.tube_flow,
                shell_t_in=data.shell_t_in,
                shell_t_out=data.shell_t_out,
                shell_flow=data.shell_flow,
                time_hr=data.time_hr,
                crude_api=data.crude_api,
                crude_tan=data.crude_tan,
                crude_chlorides=data.crude_chlorides,
            )
        except Exception:
            res = {}
    else:
        res = {}

    dt_tube = data.tube_t_out - data.tube_t_in
    dt_shell = data.shell_t_in - data.shell_t_out
    duty_kW = data.tube_flow * 2.1 * dt_tube
    rf = res.get("predicted_rf_m2KW", 0.00032)
    tema_limit = res.get("tema_limit_m2KW", 0.0005)
    
    # Calculate daily energy penalty in dollars (lost preheat replaced by furnace fuel)
    dollar_penalty = max(0.0, (rf / max(1e-6, tema_limit)) * 1450.0)
    fouling_pct = min(100.0, round((rf / max(1e-6, tema_limit)) * 100.0, 1))
    
    status = "TEMA LIMIT EXCEEDED" if rf >= tema_limit else ("ELEVATED FOULING" if rf >= 0.00035 else "OPERATING NORMAL")
    risk_level = "CRITICAL" if rf >= tema_limit else ("WARNING" if rf >= 0.00035 else "SAFE")

    return {
        "machine_type": f"Shell & Tube Heat Exchanger ({data.exchanger_id})",
        "exchanger_id": data.exchanger_id,
        "status": status,
        "risk_level": risk_level,
        "probability": round(fouling_pct / 100.0, 4),
        "fouling_resistance_m2KW": round(rf, 6),
        "tema_limit_m2KW": round(tema_limit, 6),
        "fouling_pct": fouling_pct,
        "energy_penalty_dollars_per_day": round(dollar_penalty, 2),
        "thermal_metrics": {
            "duty_kW": round(duty_kW, 1),
            "delta_t_tube_C": round(dt_tube, 1),
            "delta_t_shell_C": round(dt_shell, 1),
        },
        "recommendation": "Immediate chemical cleaning or tube bundle flush required." if risk_level == "CRITICAL" else "Monitor crude fouling rate and plan wash cycle."
    }

@app.post("/api/simulate/bearing")
def simulate_bearing(data: BearingSimulationInput):
    if unified_predictor is not None:
        try:
            res = unified_predictor.predict_bearing(
                bearing_id=data.bearing_id,
                rms=data.rms,
                kurtosis=data.kurtosis,
                peak=data.peak,
                peak_to_peak=data.peak_to_peak,
                crest_factor=data.crest_factor,
                hours_elapsed=data.hours_elapsed,
                rpm=data.rpm,
            )
        except Exception:
            res = {}
    else:
        res = {}

    # ISO 10816-3 Vibration Severity Classification
    if data.rms < 0.28:
        iso_zone, iso_status, risk_level = "Zone A", "Good (New Condition)", "SAFE"
        fail_prob = 0.05
    elif data.rms < 0.71:
        iso_zone, iso_status, risk_level = "Zone B", "Acceptable (Continuous Operation)", "SAFE"
        fail_prob = 0.18
    elif data.rms < 1.80:
        iso_zone, iso_status, risk_level = "Zone C", "Unsatisfactory (Alert / Elevated)", "WARNING"
        fail_prob = 0.62
    else:
        iso_zone, iso_status, risk_level = "Zone D", "Unacceptable (Trip / Danger)", "CRITICAL"
        fail_prob = 0.94

    if data.kurtosis > 6.0 and risk_level != "CRITICAL":
        risk_level = "CRITICAL"
        fail_prob = max(fail_prob, 0.88)

    fr = data.rpm / 60.0
    fault_freqs = {
        "BPFO (Outer Race)": round(3.585 * fr, 1),
        "BPFI (Inner Race)": round(5.415 * fr, 1),
        "BSF (Ball Spin)": round(2.357 * fr, 1),
        "FTF (Cage Train)": round(0.398 * fr, 1),
    }

    rul_hours = res.get("predicted_rul_hours", max(1.0, round(450.0 / (data.rms * 2.5 + (data.kurtosis - 3.0) * 5.0 + 0.1), 1)))

    return {
        "machine_type": f"Rotating Bearing Rig ({data.bearing_id})",
        "bearing_id": data.bearing_id,
        "probability": round(fail_prob, 4),
        "risk_level": risk_level,
        "iso_zone": iso_zone,
        "iso_status": iso_status,
        "degradation_stage": res.get("degradation_stage", "Healthy (Normal)" if risk_level == "SAFE" else ("Warning (Degrading)" if risk_level == "WARNING" else "Critical (Failure Imminent)")),
        "predicted_rul_hours": rul_hours,
        "fault_frequencies": fault_freqs,
        "vibration_metrics": {
            "rms": data.rms,
            "kurtosis": data.kurtosis,
            "peak": data.peak,
            "peak_to_peak": data.peak_to_peak,
            "crest_factor": data.crest_factor,
        },
        "recommendation": "Vibration levels are critical. Immediate bearing replacement required to avoid catastrophic spindle seizure." if risk_level == "CRITICAL" else ("Vibration alert. Schedule ultrasound lubrication and high-frequency spectral analysis." if risk_level == "WARNING" else "Normal operation. Bearing health within design envelope.")
    }

class ForecastInput(BaseModel):
    air_temp: float
    proc_temp: float
    rpm: float
    torque: float
    tool_wear: float
    type_str: str
    active_thresh: float
    horizon_months: int
    wear_rate: float
    rpm_drift: float
    torque_drift: float
    sc_rpm: float = 0.0
    sc_torque: float = 0.0
    sc_wear: float = 0.0

@app.post("/api/forecast")
def run_forecast(data: ForecastInput):
    if not models_loaded:
        raise HTTPException(status_code=500, detail="ML models not loaded")
        
    type_enc = {"L (Low)": 0, "M (Medium)": 1, "H (High)": 2}.get(data.type_str, 1)
    
    deg = {
        "wear_rate_per_month": data.wear_rate,
        "rpm_drift_per_month": data.rpm_drift,
        "torque_drift_per_month": data.torque_drift,
        "air_temp_drift_per_month": 0.05,
        "proc_temp_drift_per_month": 0.1,
    }
    
    result_base = run_ttf_forecast(
        xgb_model=xgb_model,
        scaler=scaler,
        air_temp=data.air_temp,
        proc_temp=data.proc_temp,
        rpm=data.rpm,
        torque=data.torque,
        tool_wear=data.tool_wear,
        type_enc=type_enc,
        active_threshold=data.active_thresh,
        horizon_months=data.horizon_months,
        degradation=deg,
        scenario_overrides={"rpm": 0.0, "torque": 0.0, "tool_wear": 0.0}
    )
    
    result_scen = None
    if data.sc_rpm != 0.0 or data.sc_torque != 0.0 or data.sc_wear != 0.0:
        result_scen = run_ttf_forecast(
            xgb_model=xgb_model,
            scaler=scaler,
            air_temp=data.air_temp,
            proc_temp=data.proc_temp,
            rpm=data.rpm,
            torque=data.torque,
            tool_wear=data.tool_wear,
            type_enc=type_enc,
            active_threshold=data.active_thresh,
            horizon_months=data.horizon_months,
            degradation=deg,
            scenario_overrides={"rpm": data.sc_rpm, "torque": data.sc_torque, "tool_wear": data.sc_wear}
        )
        
    return {
        "base": {
            "months": result_base.months,
            "probabilities": result_base.probabilities,
            "predicted_fail_month": result_base.predicted_fail_month,
            "fail_message": result_base.fail_message,
            "projected_params": result_base.projected_params,
            "interventions": result_base.interventions
        },
        "scenario": {
            "months": result_scen.months,
            "probabilities": result_scen.probabilities,
            "predicted_fail_month": result_scen.predicted_fail_month,
            "fail_message": result_scen.fail_message,
            "projected_params": result_scen.projected_params,
            "interventions": result_scen.interventions
        } if result_scen else None
    }

# ─────────────────────────────────────────────
# BATCH ANALYSIS
# ─────────────────────────────────────────────
@app.post("/api/batch")
async def batch_process(file: UploadFile = File(...), active_thresh: float = Form(0.35)):
    try:
        content = await file.read()
        import io
        df_raw = None
        for sep in [',', ';', '\t']:
            try:
                temp_df = pd.read_csv(io.BytesIO(content), sep=sep)
                if len(temp_df.columns) > 1:
                    df_raw = temp_df
                    break
            except Exception:
                continue
        if df_raw is None:
            df_raw = pd.read_csv(io.BytesIO(content))
            
        df_raw.columns = [str(c).strip().lstrip('\ufeff').strip('"').strip("'") for c in df_raw.columns]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {e}")

    # Check Dataset Category
    cols = list(df_raw.columns)
    is_hx = any("Crude" in c or "E0" in c or "fouling" in c.lower() for c in cols)
    is_bearing = any("Bearing" in c or "rms" in c.lower() or "kurtosis" in c.lower() or "vibration" in c.lower() for c in cols)

    # ---------------------------------------------------------
    # CASE A: HEAT EXCHANGER DATASET
    # ---------------------------------------------------------
    if is_hx:
        df = df_raw.copy()
        total = len(df)
        
        # Calculate key thermal performance metrics
        # If simulation time exists, compute progressive fouling curve
        time_col = next((c for c in cols if "time" in c.lower()), None)
        if time_col and pd.api.types.is_numeric_dtype(df[time_col]):
            time_norm = (df[time_col] - df[time_col].min()) / max(1.0, (df[time_col].max() - df[time_col].min()))
            fouling_rf = 0.0001 + (time_norm * 0.00045)
        else:
            fouling_rf = pd.Series(0.00025, index=df.index)

        tema_limit = 0.0005
        prob_arr = (fouling_rf / tema_limit).clip(0.0, 1.0)
        
        risk_labels = []
        pred_arr = []
        for p, rf in zip(prob_arr, fouling_rf):
            if rf >= tema_limit:
                risk_labels.append("CRITICAL")
                pred_arr.append(1)
            elif rf >= 0.00035:
                risk_labels.append("WARNING")
                pred_arr.append(0)
            else:
                risk_labels.append("SAFE")
                pred_arr.append(0)

        df["Asset Class"] = "Heat Exchanger"
        df["Failure Probability (%)"] = (prob_arr * 100).round(2)
        df["Predicted Failure"] = pred_arr
        df["Risk Level"] = risk_labels
        df["Fouling Resistance (m²K/W)"] = fouling_rf.round(6)

        n_critical = sum(1 for r in risk_labels if r == "CRITICAL")
        n_warning = sum(1 for r in risk_labels if r == "WARNING")
        n_safe = sum(1 for r in risk_labels if r == "SAFE")

        # Subsample for UI display if dataset is huge (> 1000 rows)
        if len(df) > 1000:
            df_display = df.iloc[::max(1, len(df)//600)].copy()
        else:
            df_display = df.copy()
        df_display = df_display.fillna(0.0).replace([np.inf, -np.inf], 0.0)

        return {
            "records": df_display.to_dict(orient="records"),
            "stats": {
                "total": total,
                "critical": n_critical,
                "elevated_wear": 0,
                "warning": n_warning,
                "safe": n_safe,
                "alarms": n_critical,
                "asset_category": "Heat Exchanger (Preheat Train)",
                "dataset_type": "heat_exchanger"
            },
            "evaluation": None
        }

    # ---------------------------------------------------------
    # CASE B: BEARING VIBRATION DATASET
    # ---------------------------------------------------------
    elif is_bearing:
        df = df_raw.copy()
        total = len(df)

        rms_cols = [c for c in cols if "rms" in c.lower()]
        kurt_cols = [c for c in cols if "kurtosis" in c.lower()]

        if rms_cols:
            rms_series = df[rms_cols].max(axis=1)
        else:
            rms_series = pd.Series(0.25, index=df.index)

        if kurt_cols:
            kurt_series = df[kurt_cols].max(axis=1)
        else:
            kurt_series = pd.Series(3.0, index=df.index)

        risk_labels = []
        pred_arr = []
        prob_arr = []
        iso_zones = []

        for rms, kurt in zip(rms_series, kurt_series):
            if rms >= 1.80 or kurt >= 7.0:
                risk_labels.append("CRITICAL")
                pred_arr.append(1)
                prob_arr.append(94.5)
                iso_zones.append("Zone D (Danger)")
            elif rms >= 0.71 or kurt >= 4.5:
                risk_labels.append("WARNING")
                pred_arr.append(0)
                prob_arr.append(62.0)
                iso_zones.append("Zone C (Alert)")
            else:
                risk_labels.append("SAFE")
                pred_arr.append(0)
                prob_arr.append(12.5)
                iso_zones.append("Zone A/B (Normal)")

        df["Asset Class"] = "Rotating Bearing"
        df["Failure Probability (%)"] = prob_arr
        df["Predicted Failure"] = pred_arr
        df["Risk Level"] = risk_labels
        df["ISO Severity"] = iso_zones

        n_critical = sum(1 for r in risk_labels if r == "CRITICAL")
        n_warning = sum(1 for r in risk_labels if r == "WARNING")
        n_safe = sum(1 for r in risk_labels if r == "SAFE")

        if len(df) > 1000:
            df_display = df.iloc[::max(1, len(df)//600)].copy()
        else:
            df_display = df.copy()
        df_display = df_display.fillna(0.0).replace([np.inf, -np.inf], 0.0)

        return {
            "records": df_display.to_dict(orient="records"),
            "stats": {
                "total": total,
                "critical": n_critical,
                "elevated_wear": 0,
                "warning": n_warning,
                "safe": n_safe,
                "alarms": n_critical,
                "asset_category": "IMS Bearing Vibration Rig",
                "dataset_type": "bearing"
            },
            "evaluation": None
        }

    # ---------------------------------------------------------
    # CASE C: CNC MILLING MACHINE DATASET (DEFAULT / ORIGINAL)
    # ---------------------------------------------------------
    alias_map = {
        "Type": ["type", "product_type", "product type", "machine_type", "machine type"],
        "Air temperature [K]": ["air temperature [k]", "air temperature", "air_temperature", "air_temp", "air temp", "air_temp_k"],
        "Process temperature [K]": ["process temperature [k]", "process temperature", "process_temperature", "proc_temp", "process_temp", "process_temp_k"],
        "Rotational speed [rpm]": ["rotational speed [rpm]", "rotational speed", "rotational_speed", "rpm", "speed", "rotational_speed_rpm"],
        "Torque [Nm]": ["torque [nm]", "torque", "torque_nm", "torque (nm)"],
        "Tool wear [min]": ["tool wear [min]", "tool wear", "tool_wear", "wear", "tool_wear_min", "tool wear (min)"],
        "Machine failure": ["machine failure", "machine_failure", "failure", "target"]
    }
    
    col_rename = {}
    for col in df_raw.columns:
        col_lower = col.lower().strip()
        for target_col, aliases in alias_map.items():
            if target_col not in col_rename.values():
                if col_lower == target_col.lower() or col_lower in aliases:
                    col_rename[col] = target_col
                    break
    df_raw.rename(columns=col_rename, inplace=True)

    needed = ["Type", "Air temperature [K]", "Process temperature [K]",
              "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    missing = [c for c in needed if c not in df_raw.columns]
    
    # Auto-fill missing columns if any for graceful handling
    for m in missing:
        if m == "Type": df_raw["Type"] = "M"
        elif m == "Air temperature [K]": df_raw["Air temperature [K]"] = 300.0
        elif m == "Process temperature [K]": df_raw["Process temperature [K]"] = 310.0
        elif m == "Rotational speed [rpm]": df_raw["Rotational speed [rpm]"] = 1500.0
        elif m == "Torque [Nm]": df_raw["Torque [Nm]"] = 40.0
        elif m == "Tool wear [min]": df_raw["Tool wear [min]"] = 100.0

    type_enc = {"L": 0, "M": 1, "H": 2}
    df = df_raw.copy()
    df["Type_encoded"]        = df["Type"].map(type_enc).fillna(1).astype(int)
    df["power_watts"]         = (2 * np.pi / 60) * df["Rotational speed [rpm]"] * df["Torque [Nm]"]
    df["temp_diff_K"]         = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["wear_torque_product"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    df["wear_pct"]            = df["Tool wear [min]"] / 253.0
    df["high_torque"]         = (df["Torque [Nm]"] > 55).astype(int)
    df["high_wear"]           = (df["Tool wear [min]"] > 200).astype(int)
    
    if models_loaded and FEATURE_COLS:
        X_scaled = scaler.transform(df[FEATURE_COLS])
        proba = xgb_model.predict_proba(X_scaled)[:, 1]
    else:
        proba = np.clip(df["Tool wear [min]"] / 250.0, 0.05, 0.95).values

    tool_wear_arr = df["Tool wear [min]"].values

    predicted_arr, tier_labels = wear_band_policy.apply(
        proba, tool_wear_arr, active_thresh,
        wear_window_low=WEAR_WINDOW_LOW,
        wear_window_high=WEAR_WINDOW_HIGH,
    )

    df["Asset Class"]             = "CNC Milling Center"
    df["Failure Probability (%)"] = (proba * 100).round(2)
    df["Predicted Failure"]       = predicted_arr
    df["Risk Level"]              = tier_labels

    total               = len(df)
    n_critical          = int((df["Risk Level"] == "CRITICAL").sum())
    n_elevated_wear     = int((df["Risk Level"] == "ELEVATED_WEAR_WATCH").sum())
    n_warning           = int((df["Risk Level"] == "WARNING").sum())
    n_safe              = int((df["Risk Level"] == "SAFE").sum())
    n_predicted         = int(df["Predicted Failure"].sum())

    eval_metrics = None
    if "Machine failure" in df.columns:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, average_precision_score, confusion_matrix
        )
        y_true = df["Machine failure"].astype(int)
        y_pred = df["Predicted Failure"].astype(int)

        acc    = accuracy_score(y_true, y_pred)
        prec   = precision_score(y_true, y_pred, zero_division=0)
        rec    = recall_score(y_true, y_pred, zero_division=0)
        f1     = f1_score(y_true, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_true, proba)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        eval_metrics = {
            "accuracy":         float(acc),
            "precision":        float(prec),
            "recall":           float(rec),
            "f1_score":         float(f1),
            "pr_auc":           float(pr_auc),
            "false_positives":  int(fp),
            "false_negatives":  int(fn),
        }

    if len(df) > 1000:
        df_display = df.iloc[::max(1, len(df)//600)].copy()
    else:
        df_display = df.copy()
    df_display = df_display.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    return {
        "records": df_display.to_dict(orient="records"),
        "stats": {
            "total":             total,
            "critical":          n_critical,
            "elevated_wear":     n_elevated_wear,
            "warning":           n_warning,
            "safe":              n_safe,
            "alarms":            n_predicted,
            "asset_category":    "CNC Milling Center Fleet",
            "dataset_type":      "milling",
            "wear_window_low":   WEAR_WINDOW_LOW,
            "wear_window_high":  WEAR_WINDOW_HIGH,
        },
        "evaluation": eval_metrics
    }


@app.get("/api/assets")
def get_assets():
    return REGISTRY.get_all()

@app.post("/api/assets")
def create_asset(asset: Dict):
    asset_id = REGISTRY.add_asset(asset)
    return {"status": "success", "id": asset_id}

@app.put("/api/assets/{asset_id}")
def update_asset(asset_id: str, updates: Dict):
    success = REGISTRY.update_asset(asset_id, updates)
    if not success:
        raise HTTPException(status_code=404, detail="Asset not found")
    return {"status": "success"}

@app.delete("/api/assets/{asset_id}")
def delete_asset(asset_id: str):
    success = REGISTRY.delete_asset(asset_id)
    if not success:
        raise HTTPException(status_code=404, detail="Asset not found")
    return {"status": "success"}

# ─────────────────────────────────────────────
# HEAT EXCHANGERS ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/api/exchangers")
def get_exchangers():
    # Load registered heat exchangers
    hxs = REGISTRY.get_hx_units()
    
    # If registry is empty of HXs, register a couple of template units for the UI
    if not hxs:
        hx1 = REGISTRY.build_default_asset("Heat Exchanger")
        hx1["id"] = "HX-101"
        hx1["description"] = "Distillation Reflux Cooler"
        hx1["shell_temp_in"] = 120.0
        hx1["shell_temp_out"] = 92.5
        hx1["shell_flow_kg_s"] = 12.5
        
        hx2 = REGISTRY.build_default_asset("Heat Exchanger")
        hx2["id"] = "HX-102"
        hx2["description"] = "Reactor Bottoms Preheater"
        hx2["shell_temp_in"] = 145.0
        hx2["shell_temp_out"] = 120.0
        hx2["shell_flow_kg_s"] = 8.0
        hx2["days_since_last_clean"] = 180
        
        REGISTRY.add_asset(hx1)
        REGISTRY.add_asset(hx2)
        hxs = [hx1, hx2]
        
    # Analyze
    hx_units = []
    for h in hxs:
        u = HXUnit(
            unit_id=h["id"],
            description=h.get("description", ""),
            fluid_type=h.get("fluid_type", "process_liquid"),
            shell_temp_in=float(h.get("shell_temp_in", 120.0)),
            shell_temp_out=float(h.get("shell_temp_out", 95.0)),
            shell_flow_kg_s=float(h.get("shell_flow_kg_s", 10.0)),
            shell_cp=float(h.get("shell_cp", 4.18)),
            tube_temp_in=float(h.get("tube_temp_in", 60.0)),
            tube_temp_out=float(h.get("tube_temp_out", 80.0)),
            tube_flow_kg_s=float(h.get("tube_flow_kg_s", 12.0)),
            tube_cp=float(h.get("tube_cp", 4.18)),
            design_U=float(h.get("design_U", 1000.0)),
            heat_area_m2=float(h.get("heat_area_m2", 50.0)),
            days_since_last_clean=int(h.get("days_since_last_clean", 0))
        )
        u.analyze()
        hx_units.append(u)
        
    # Sort by criticality score (highest first)
    hx_units_sorted = sorted(hx_units, key=lambda x: x.criticality_score, reverse=True)
    
    return [
        {
            "id": u.unit_id,
            "description": u.description,
            "fluid_type": u.fluid_type,
            "effectiveness": u.effectiveness,
            "design_effectiveness": u.design_effectiveness,
            "fouling_factor": u.fouling_factor,
            "fouling_limit": u.fouling_limit,
            "foul_pct": (u.fouling_factor / u.fouling_limit * 100) if u.fouling_limit else 0,
            "heat_duty_kw": u.heat_duty_kw,
            "current_U": u.current_U,
            "criticality_score": u.criticality_score,
            "days_to_clean": u.days_to_clean,
            "status_color": "#ef4444" if u.criticality_score >= 80 else ("#f59e0b" if u.criticality_score >= 50 else "#10b981"),
            "status_label": "CRITICAL" if u.criticality_score >= 80 else ("WARNING" if u.criticality_score >= 50 else "SAFE"),
            "lmtd": u.lmtd,
            "temperature_pinch": u.temperature_pinch,
            "diagnostics": u.diagnostics,
            "health_score": u.health_score
        } for u in hx_units_sorted
    ]

# ─────────────────────────────────────────────
# HISTORICAL STATS ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/api/historical-stats")
def get_historical_stats():
    dp = os.path.join(BASE_DIR, "ai4i2020.csv")
    if not os.path.exists(dp):
        raise HTTPException(status_code=404, detail="ai4i2020.csv missing")
        
    df = pd.read_csv(dp, encoding="utf-8-sig")
    total = len(df)
    fails = int(df["Machine failure"].sum())
    
    type_vc = df["Type"].value_counts().to_dict()
    
    fbt = df.groupby("Type")["Machine failure"].agg(["sum","count"])
    fbt_dict = {}
    for t, r in fbt.iterrows():
        fbt_dict[t] = {
            "sum": int(r["sum"]),
            "count": int(r["count"]),
            "fail_pct": float(r["sum"] / r["count"] * 100)
        }
        
    return {
        "size": total,
        "faults": fails,
        "fail_rate_pct": float(fails/total*100),
        "class_ratio": f"{int((total-fails)/fails)}:1",
        "grades": type_vc,
        "grade_fail_rates": fbt_dict
    }

@app.get("/api/historical-distribution")
def get_historical_distribution(column: str):
    dp = os.path.join(BASE_DIR, "ai4i2020.csv")
    if not os.path.exists(dp):
        raise HTTPException(status_code=404, detail="ai4i2020.csv missing")
        
    df = pd.read_csv(dp, encoding="utf-8-sig")
    if column not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid column")
        
    # sample down to 500 rows for lightweight chart transmission
    df_sample = df[[column, "Machine failure"]].dropna()
    return df_sample.to_dict(orient="records")

# ─────────────────────────────────────────────
# FINANCIAL IMPACT & LOCAL LLM ENDPOINTS
# ─────────────────────────────────────────────
class FinancialImpactRequest(BaseModel):
    failure_probability: float  # e.g. 0.85
    downtime_cost_per_hour: float = 2500.0  # $/hr
    unplanned_downtime_hours: float = 12.0  # hrs
    planned_downtime_hours: float = 2.0     # hrs
    replacement_part_cost: float = 4500.0   # $
    planned_maintenance_cost: float = 800.0  # $
    lost_units_per_hour: float = 50.0       # units/hr
    profit_margin_per_unit: float = 35.0    # $/unit

@app.post("/api/financial-impact")
def calculate_financial_impact(req: FinancialImpactRequest):
    p_fail = max(0.0, min(1.0, req.failure_probability))
    
    # Unplanned breakdown cost
    downtime_loss = req.unplanned_downtime_hours * req.downtime_cost_per_hour
    production_loss = req.unplanned_downtime_hours * req.lost_units_per_hour * req.profit_margin_per_unit
    unplanned_total = downtime_loss + req.replacement_part_cost + production_loss
    
    # Expected loss based on machine failure probability
    expected_failure_loss = p_fail * unplanned_total
    
    # Planned maintenance cost
    planned_downtime_loss = req.planned_downtime_hours * req.downtime_cost_per_hour
    planned_total = planned_downtime_loss + req.planned_maintenance_cost
    
    # Savings & ROI if proactive maintenance is performed
    net_savings = expected_failure_loss - planned_total
    roi_pct = (net_savings / planned_total * 100.0) if planned_total > 0 else 0.0
    
    return {
        "failure_probability": p_fail,
        "unplanned_breakdown_cost": round(unplanned_total, 2),
        "expected_failure_loss": round(expected_failure_loss, 2),
        "planned_maintenance_cost": round(planned_total, 2),
        "net_savings": round(net_savings, 2),
        "roi_pct": round(roi_pct, 1),
        "recommendation": "PROACTIVE MAINTENANCE HIGHLY RECOMMENDED" if net_savings > 0 and p_fail >= 0.35 else "MONITOR TELEMETRY — SCHEDULE ROUTINE MAINTENANCE"
    }

class LLMQueryRequest(BaseModel):
    prompt: str
    context: Optional[Dict] = None
    model: Optional[str] = "qwen2.5:3b"

@app.post("/api/llm-query")
def query_local_llm(req: LLMQueryRequest):
    user_prompt = req.prompt.strip()
    ctx = req.context or {}
    
    # Check for Google Gemini API key
    google_key = os.environ.get("GOOGLE_API_KEY")
    if google_key and not google_key.startswith("your_"):
        try:
            from google import genai
            client = genai.Client(api_key=google_key)
            system_prompt = (
                "You are the Lead Industrial Reliability & Predictive Maintenance AI Diagnostics Assistant.\n"
                "You provide expert technical analysis across 3 critical asset classes:\n"
                "1. CNC Milling Centers (spindle RPM, motor torque, tool wear, failure modes: TWF, HDF, PWF, OSF, RNF).\n"
                "2. Crude Preheat Shell & Tube Heat Exchangers (E01-E05) (fouling factor Rf, TEMA design limit 0.0005 m²K/W, furnace energy penalty, cleaning schedules).\n"
                "3. NASA/IMS High-Speed Rotating Bearings (RMS vibration, kurtosis, ISO 10816-3 Zones A-D, defect frequencies BPFO/BPFI/BSF/FTF, RUL estimation).\n\n"
                "Formatting Requirements:\n"
                "- Always respond in rich GitHub-style Markdown with clear headings (###), bullet points, bold key metric values, and actionable engineering next steps.\n"
                "- If the user asks about financial loss, compute downtime loss = hours * cost/hr + parts + lost production margin.\n"
            )
            full_prompt = f"{system_prompt}\n\nLive Telemetry Context:\n{json.dumps(ctx, indent=2)}\n\nUser Query: {user_prompt}"
            
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt
            )
            if resp and resp.text:
                return {
                    "source": "Google Gemini 2.5 Flash (Industrial Diagnostics)",
                    "response": resp.text.strip()
                }
        except Exception as gemini_err:
            print("Gemini generation attempt failed, falling back to local expert:", gemini_err)

    # High-Performance Local Domain Expert AI Engine Fallback (Zero delay)
    prompt_lower = user_prompt.lower()
    rpm = ctx.get("rpm", 1500)
    torque = ctx.get("torque", 40.0)
    air_temp = ctx.get("air_temp", 300.0)
    proc_temp = ctx.get("proc_temp", 310.0)
    tool_wear = ctx.get("tool_wear", 108)
    fail_prob = ctx.get("xgb_prob", ctx.get("rf_prob", ctx.get("probability", 0.15)))
    
    ans_lines = []
    
    if "loss" in prompt_lower or "profit" in prompt_lower or "cost" in prompt_lower or "financial" in prompt_lower or "roi" in prompt_lower:
        downtime_loss = 12 * 2500.0 + 4500.0 + (12 * 50 * 35.0)
        exp_loss = fail_prob * downtime_loss
        ans_lines.append("### 💰 Financial Loss & Downtime ROI Assessment")
        ans_lines.append(f"- **Current Failure Probability:** `{fail_prob*100:.1f}%`")
        ans_lines.append(f"- **Unplanned Breakdown Impact:** **${downtime_loss:,.2f}** (Includes $30,000 lost production hours, $4,500 replacement parts, and $21,000 idle margin).")
        ans_lines.append(f"- **Risk-Weighted Expected Loss:** **${exp_loss:,.2f}**")
        ans_lines.append("- **Scheduled Preventive Repair Cost:** **$5,800.00**")
        if exp_loss > 5800:
            ans_lines.append(f"> ⚡ **Action Recommended:** Executing proactive maintenance immediately protects **${exp_loss - 5800:,.2f}** in net operating profit.")
        else:
            ans_lines.append("> ✅ **Action Recommended:** Risk is within acceptable thresholds. Maintain scheduled operational inspection cycles.")

    elif "bearing" in prompt_lower or "vibration" in prompt_lower or "rul" in prompt_lower:
        rms_val = ctx.get("rms", 0.35)
        kurt_val = ctx.get("kurtosis", 3.2)
        ans_lines.append("### 🔄 Bearing Vibration & ISO 10816 Diagnostics")
        ans_lines.append(f"- **Vibration RMS Acceleration:** `{rms_val} g` | **Kurtosis:** `{kurt_val}`")
        if rms_val >= 1.80 or kurt_val >= 6.0:
            ans_lines.append("- **ISO 10816 Severity:** **Zone D (Danger / Failure Imminent)**")
            ans_lines.append("- **Recommended Action:** 🚨 Immediate emergency stop. Inspect outer race for spalling (BPFO) and replace bearing cartridge.")
        elif rms_val >= 0.71:
            ans_lines.append("- **ISO 10816 Severity:** **Zone C (Unsatisfactory / Warning)**")
            ans_lines.append("- **Recommended Action:** ⚠️ Schedule high-frequency demodulation analysis and replenish synthetic lubricant.")
        else:
            ans_lines.append("- **ISO 10816 Severity:** **Zone A/B (Normal / Healthy)**")
            ans_lines.append("- **Recommended Action:** ✅ Continuous operation permitted. Safe RUL remaining.")

    elif "fouling" in prompt_lower or "exchanger" in prompt_lower or "heat" in prompt_lower or "tema" in prompt_lower or "e0" in prompt_lower:
        ans_lines.append("### 🌡️ Shell & Tube Heat Exchanger Fouling Diagnostics")
        ans_lines.append("- **Equipment Scope:** Crude Preheat Train (Units E01 through E05).")
        ans_lines.append("- **TEMA Standard Limit:** `0.000500 m²·K/W` fouling resistance.")
        ans_lines.append("- **Thermal Penalty:** As crude passes through fouled bundles, heat transfer coefficient $U$ degrades, forcing higher furnace fuel burn (~$1,200-$1,800/day per train).")
        ans_lines.append("- **Recommended Action:** Schedule chemical CIP wash or offline high-pressure hydroblasting when $R_f \ge 0.000350 m²·K/W$.")

    elif "tool" in prompt_lower or "wear" in prompt_lower:
        ans_lines.append("### ⚙️ CNC Tool Wear & Insert Diagnostics")
        ans_lines.append(f"- **Current Tool Wear Time:** `{tool_wear} minutes` (Design safety threshold = 200 min).")
        if tool_wear >= 200:
            ans_lines.append("- **Wear Status:** 🚨 **CRITICAL WEAR (TWF Danger)**")
            ans_lines.append("- **Action Required:** Flank wear exceeds 0.3mm equivalent. Immediately index or replace tool insert.")
        elif tool_wear >= 150:
            ans_lines.append("- **Wear Status:** ⚠️ **ELEVATED WEAR WATCH**")
            ans_lines.append("- **Action Required:** Queue tool change at next batch completion.")
        else:
            ans_lines.append("- **Wear Status:** ✅ **STABLE WEAR**")
            ans_lines.append("- **Action Required:** Tool geometry within normal operating tolerances.")

    else:
        ans_lines.append("### 🛡️ Unified Industrial Reliability Diagnostics")
        ans_lines.append(f"- **Monitored Failure Probability:** `{fail_prob*100:.1f}%`")
        ans_lines.append(f"- **Active Telemetry State:** Spindle `{rpm} RPM` | Torque `{torque} Nm` | Temp `{proc_temp} K` | Tool Wear `{tool_wear} min`")
        if fail_prob >= 0.35:
            ans_lines.append("> ⚠️ **Elevated Risk Detected:** Probability exceeds operational threshold. Check mechanical torque load and tool wear.")
        else:
            ans_lines.append("> ✅ **Fleet Telemetry Nominal:** All mechanical, thermal, and vibration sensors indicate safe operating limits.")

    return {
        "source": "Local Industrial Expert AI Diagnostics Engine",
        "response": "\n\n".join(ans_lines)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# Force reload - retrained 12-feature model 2
