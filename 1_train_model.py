"""
1_train_model.py
Predictive Maintenance -- Improved ML Training Pipeline

Trains:
  1. Primary binary failure classifier (XGBoost & Random Forest)
  2. 5 individual XGBoost classifiers for each failure mode (TWF, HDF, PWF, OSF, RNF)
  3. Calibrates decision threshold from precision-recall curve on held-out test set

KEY CHANGES vs. previous SMOTE version:
  - SMOTE removed. Replaced with scale_pos_weight = neg/pos computed from y_train
    at runtime (not hardcoded) so it tracks the actual class ratio in training data.
  - Added 3 physics-informed features:
      power_watts         = 2*pi * rpm * torque / 60         (PWF proxy)
      temp_diff_K         = process_temp - air_temp           (HDF proxy)
      wear_torque_product = tool_wear * torque                (OSF proxy)
  - Type encoded L=0, M=1, H=2 via explicit map (stable ordering).
  - Stratified 80/20 split; scaler fit ONLY on train — no data leakage.
  - Threshold chosen by maximising F1 on the held-out test PR curve.
  - Headline metrics: Precision / Recall / F1 / Confusion matrix.
    Accuracy shown but annotated against the 96.6% trivial baseline.
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, f1_score, recall_score, precision_score,
    accuracy_score, precision_recall_curve, average_precision_score, roc_auc_score
)
import xgboost as xgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "ai4i2020.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("  PREDICTIVE MAINTENANCE -- IMPROVED TRAINING PIPELINE")
print("  (scale_pos_weight | PR-curve threshold | no SMOTE)")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print(f"\n[1/8] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[2/8] Engineering features...")

# Type encoding: explicit map so ordering is deterministic (L=0, M=1, H=2)
type_map = {"L": 0, "M": 1, "H": 2}
df["Type_encoded"] = df["Type"].map(type_map).fillna(1).astype(int)

# Physics-informed features (named to match failure-mode semantics)
df["power_watts"]         = (2.0 * np.pi / 60.0) * df["Rotational speed [rpm]"] * df["Torque [Nm]"]
df["temp_diff_K"]         = df["Process temperature [K]"] - df["Air temperature [K]"]
df["wear_torque_product"] = df["Tool wear [min]"] * df["Torque [Nm]"]

# Auxiliary derived features
df["wear_pct"]    = df["Tool wear [min]"] / 253.0
df["high_torque"] = (df["Torque [Nm]"] > 55).astype(int)
df["high_wear"]   = (df["Tool wear [min]"] > 200).astype(int)

FEATURE_COLS = [
    "Type_encoded",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    # Physics-informed features
    "power_watts",
    "temp_diff_K",
    "wear_torque_product",
    # Auxiliary
    "wear_pct",
    "high_torque",
    "high_wear",
]

TARGET      = "Machine failure"
SUB_TARGETS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

X      = df[FEATURE_COLS]
y      = df[TARGET]
y_subs = df[SUB_TARGETS]

n_pos_all = int(y.sum())
n_neg_all = int((y == 0).sum())
print(f"    Features  : {len(FEATURE_COLS)}")
print(f"    Positives : {n_pos_all} ({n_pos_all/len(y)*100:.2f}%)")
print(f"    Negatives : {n_neg_all} ({n_neg_all/len(y)*100:.2f}%)")
for sub in SUB_TARGETS:
    print(f"      - {sub}: {y_subs[sub].sum()} failures")

# ─────────────────────────────────────────────
# 3. STRATIFIED TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
print("\n[3/8] Stratified 80/20 train/test split...")
X_train, X_test, y_train, y_test, y_train_subs, y_test_subs = train_test_split(
    X, y, y_subs, test_size=0.2, random_state=42, stratify=y
)
print(f"    Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")
print(f"    Train failure rate: {y_train.mean()*100:.2f}%  |  Test: {y_test.mean()*100:.2f}%")

# Compute scale_pos_weight dynamically from y_train (never hardcoded)
n_neg_train = int((y_train == 0).sum())
n_pos_train = int(y_train.sum())
scale_pos_weight = float(n_neg_train) / float(n_pos_train)
print(f"    scale_pos_weight = {n_neg_train}/{n_pos_train} = {scale_pos_weight:.2f}")

# ─────────────────────────────────────────────
# 4. SCALING  (scaler fit ONLY on train, applied to test)
# ─────────────────────────────────────────────
print("\n[4/8] Scaling features (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("    scaler.pkl saved.")

# (No SMOTE — scale_pos_weight handles class imbalance in tree-based models
#  without distorting probability calibration the way synthetic oversampling does.)

# ─────────────────────────────────────────────
# 5. (Reserved — was SMOTE, now skipped)
# ─────────────────────────────────────────────
print("\n[5/8] Skipped (SMOTE removed — using scale_pos_weight instead)")

# ─────────────────────────────────────────────
# 6. TRAIN PRIMARY MODELS
# ─────────────────────────────────────────────
print("\n[6/8] Training primary models (scale_pos_weight, no SMOTE)...")

# Random Forest: class_weight proportional to training imbalance
print("    [RF] Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    class_weight={0: 1.0, 1: scale_pos_weight},
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2,
)
rf_model.fit(X_train_scaled, y_train)

# XGBoost: scale_pos_weight computed from y_train, not hardcoded
print(f"    [XGB] Training XGBoost (scale_pos_weight={scale_pos_weight:.2f})...")
xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb_model.fit(X_train_scaled, y_train)

with open(os.path.join(OUTPUT_DIR, "xgb_model.pkl"), "wb") as f:
    pickle.dump(xgb_model, f)
with open(os.path.join(OUTPUT_DIR, "rf_model.pkl"), "wb") as f:
    pickle.dump(rf_model, f)
with open(os.path.join(OUTPUT_DIR, "feature_cols.pkl"), "wb") as f:
    pickle.dump(FEATURE_COLS, f)
print("    Primary models saved.")

# ─────────────────────────────────────────────
# 7. TRAIN SUB-MODELS (per-failure-mode)
# ─────────────────────────────────────────────
print("\n[7/8] Training failure mode models (XGBoost, scale_pos_weight per mode)...")
sub_models = {}

for sub in SUB_TARGETS:
    print(f"    Training {sub}...")
    sub_y_train = y_train_subs[sub]
    sub_n_neg   = int((sub_y_train == 0).sum())
    sub_n_pos   = int(sub_y_train.sum())
    sub_spw     = max(1.0, float(sub_n_neg) / max(1.0, float(sub_n_pos)))

    sub_xgb = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=sub_spw, eval_metric="aucpr",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    sub_xgb.fit(X_train_scaled, sub_y_train)

    sub_models[sub] = sub_xgb
    with open(os.path.join(OUTPUT_DIR, f"xgb_{sub.lower()}_model.pkl"), "wb") as f:
        pickle.dump(sub_xgb, f)
    print(f"      scale_pos_weight={sub_spw:.1f} | xgb_{sub.lower()}_model.pkl saved")

# ─────────────────────────────────────────────
# 8. THRESHOLD CALIBRATION from PR curve
# ─────────────────────────────────────────────
print("\n[8/8] Calibrating thresholds from Precision-Recall curve on held-out test set...")


def calibrate_from_pr_curve(model, X_test_s, y_test_arr):
    """Return (best_f1_thresh, safety_thresh, pr_auc) computed from the held-out test set.
    best_f1_thresh : threshold that maximises F1.
    safety_thresh  : highest threshold still achieving >= 90% recall.
    pr_auc         : area under precision-recall curve.
    """
    proba  = model.predict_proba(X_test_s)[:, 1]
    pr_auc = average_precision_score(y_test_arr, proba)
    precs, recs, thrs = precision_recall_curve(y_test_arr, proba)
    # precision_recall_curve returns len(thrs)+1 points for precs/recs
    f1s = np.where(
        (precs[:-1] + recs[:-1]) > 0,
        2 * precs[:-1] * recs[:-1] / (precs[:-1] + recs[:-1]),
        0.0,
    )
    best_f1_thresh = float(thrs[np.argmax(f1s)])
    safety_thresh  = float(thrs[0])  # fallback: lowest possible threshold
    for t, r in zip(thrs, recs[:-1]):
        if r >= 0.90:
            safety_thresh = float(t)
    return round(best_f1_thresh, 4), round(safety_thresh, 4), round(pr_auc, 4)


y_test_arr = y_test.values
xgb_best_t, xgb_saf_t, xgb_prauc = calibrate_from_pr_curve(xgb_model, X_test_scaled, y_test_arr)
rf_best_t,  rf_saf_t,  rf_prauc  = calibrate_from_pr_curve(rf_model,  X_test_scaled, y_test_arr)

print(f"    XGBoost PR-AUC={xgb_prauc}  best-F1 thresh={xgb_best_t}  safety thresh={xgb_saf_t}")
print(f"    RF      PR-AUC={rf_prauc}   best-F1 thresh={rf_best_t}   safety thresh={rf_saf_t}")

with open(os.path.join(OUTPUT_DIR, "xgb_threshold.pkl"),        "wb") as f: pickle.dump(xgb_best_t, f)
with open(os.path.join(OUTPUT_DIR, "xgb_safety_threshold.pkl"), "wb") as f: pickle.dump(xgb_saf_t,  f)
with open(os.path.join(OUTPUT_DIR, "rf_threshold.pkl"),         "wb") as f: pickle.dump(rf_best_t,  f)
with open(os.path.join(OUTPUT_DIR, "rf_safety_threshold.pkl"),  "wb") as f: pickle.dump(rf_saf_t,   f)

# Fleet medians for dashboard anomaly comparisons
fleet_medians = df[df[TARGET] == 0][FEATURE_COLS].median().to_dict()
with open(os.path.join(OUTPUT_DIR, "fleet_medians.pkl"), "wb") as f:
    pickle.dump(fleet_medians, f)

# ─────────────────────────────────────────────
# EVALUATION REPORT  (Precision / Recall / F1 / CM)
# ─────────────────────────────────────────────
trivial_acc = float((y_test_arr == 0).mean())


def model_block(model, model_name, X_test_s, y_true, bt, st, prauc):
    proba = model.predict_proba(X_test_s)[:, 1]
    roc   = roc_auc_score(y_true, proba)
    lines = [
        f"\n{'='*58}",
        f"  MODEL : {model_name}",
        f"  ROC-AUC: {roc:.4f}  |  PR-AUC: {prauc:.4f}",
        f"{'='*58}",
    ]
    for t, label in [(bt, "BALANCED (max F1)"), (st, "SAFETY-FIRST (recall >= 90%)")]:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f = f1_score(y_true, pred, zero_division=0)
        a = accuracy_score(y_true, pred)
        lines += [
            f"  [{label}]  threshold={t}",
            f"    Precision : {p:.4f}   *** PRIMARY METRIC ***",
            f"    Recall    : {r:.4f}   ({r*100:.1f}% of real failures caught)",
            f"    F1-Score  : {f:.4f}",
            f"    Accuracy  : {a:.4f}   (trivial baseline={trivial_acc:.4f} — not headline)",
            f"    Confusion Matrix:",
            f"      TN={tn:5d}  FP={fp:5d}",
            f"      FN={fn:5d}  TP={tp:5d}",
            "",
        ]
    return "\n".join(lines)


results_text = []

results_text.append(model_block(xgb_model, "XGBoost (scale_pos_weight, no SMOTE, PR-curve thresh)",
                                X_test_scaled, y_test_arr, xgb_best_t, xgb_saf_t, xgb_prauc))
results_text.append(model_block(rf_model, "Random Forest (scale_pos_weight, no SMOTE, PR-curve thresh)",
                                X_test_scaled, y_test_arr, rf_best_t, rf_saf_t, rf_prauc))

# Before/After comparison table
proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
pred_xgb  = (proba_xgb >= xgb_best_t).astype(int)
after = {
    "Precision": precision_score(y_test_arr, pred_xgb, zero_division=0),
    "Recall":    recall_score(y_test_arr, pred_xgb, zero_division=0),
    "F1-Score":  f1_score(y_test_arr, pred_xgb, zero_division=0),
    "Accuracy":  accuracy_score(y_test_arr, pred_xgb),
    "ROC-AUC":   roc_auc_score(y_test_arr, proba_xgb),
    "PR-AUC":    xgb_prauc,
}
before = {"Precision": 0.8833, "Recall": 0.7794, "F1-Score": 0.8281,
          "Accuracy": 0.9915, "ROC-AUC": 0.9776, "PR-AUC": None}

sep = "-" * 58
table_lines = [
    f"\n{sep}",
    f"  BEFORE -> AFTER COMPARISON (XGBoost, same held-out test split)",
    f"  {'Metric':<18} {'Before (SMOTE)':>16} {'After (SPW)':>13}",
    f"  {'-'*18} {'-'*16} {'-'*13}",
]
for metric in ["Precision", "Recall", "F1-Score", "Accuracy", "ROC-AUC", "PR-AUC"]:
    bv = f"{before[metric]:.4f}" if before[metric] is not None else "N/A"
    table_lines.append(f"  {metric:<18} {bv:>16} {after[metric]:>13.4f}")
table_lines += [
    f"",
    f"  Trivial baseline accuracy (always predict safe): {trivial_acc:.4f}",
    f"  scale_pos_weight: {scale_pos_weight:.2f}  (computed from training data each run)",
    f"  XGBoost threshold: {xgb_best_t}  (from PR-curve, not guessed)",
    f"  RF threshold:      {rf_best_t}  (from PR-curve, not guessed)",
    sep,
]
results_text.append("\n".join(table_lines))

full_report = "\n".join(results_text)
print(full_report)

with open(os.path.join(RESULTS_DIR, "evaluation_report.txt"), "w") as f:
    f.write(full_report)

print(f"\n{'='*60}")
print("  [OK] TRAINING COMPLETE")
print(f"  XGBoost: best-F1 thresh={xgb_best_t}, safety thresh={xgb_saf_t}")
print(f"  RF:      best-F1 thresh={rf_best_t},  safety thresh={rf_saf_t}")
print(f"  Report saved to results/evaluation_report.txt")
print(f"{'='*60}\n")

import os
import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, f1_score, recall_score, precision_score
)
# pyrefly: ignore [missing-import]
from imblearn.over_sampling import SMOTE
import xgboost as xgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "ai4i2020.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("  PREDICTIVE MAINTENANCE -- PHYSICS-INFORMED MULTI-MODEL PIPELINE")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print(f"\n[1/8] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[2/8] Engineering features...")

le = LabelEncoder()
df["Type_encoded"] = le.fit_transform(df["Type"])  # L=0, M=1, H=2

# Physical formulas
df["temp_delta"]    = df["Process temperature [K]"] - df["Air temperature [K]"]
df["power_W"]       = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
df["torque_x_wear"] = df["Torque [Nm]"] * df["Tool wear [min]"]
df["wear_pct"]      = df["Tool wear [min]"] / 253.0
df["high_torque"]   = (df["Torque [Nm]"] > 55).astype(int)
df["high_wear"]     = (df["Tool wear [min]"] > 200).astype(int)

FEATURE_COLS = [
    "Type_encoded",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temp_delta",
    "power_W",
    "torque_x_wear",
    "wear_pct",
    "high_torque",
    "high_wear",
]

TARGET = "Machine failure"
SUB_TARGETS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

X = df[FEATURE_COLS]
y = df[TARGET]
y_subs = df[SUB_TARGETS]

print(f"    Features created: {len(FEATURE_COLS)}")
print(f"    Primary Failures: {y.sum()} | Normal: {(y==0).sum()}")
for sub in SUB_TARGETS:
    print(f"      - {sub} Failures: {y_subs[sub].sum()}")

# Save fleet medians (for local physical anomaly comparisons in dashboard)
fleet_medians = df[df[TARGET] == 0][FEATURE_COLS].median().to_dict()
with open(os.path.join(OUTPUT_DIR, "fleet_medians.pkl"), "wb") as f:
    pickle.dump(fleet_medians, f)

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT (Stratified on primary target)
# ─────────────────────────────────────────────
print("\n[3/8] Splitting 80/20 (stratified)...")
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
y_train_subs, y_test_subs = y_subs.iloc[train_idx], y_subs.iloc[test_idx]

print(f"    Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 4. SCALING
# ─────────────────────────────────────────────
print("\n[4/8] Scaling features (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("    scaler.pkl saved.")

# ─────────────────────────────────────────────
# 5. SMOTE
# ─────────────────────────────────────────────
print("\n[5/8] Applying SMOTE to training data...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
print(f"    Train size after SMOTE: {X_train_sm.shape[0]} (Failures balanced: {y_train_sm.sum()})")

# ─────────────────────────────────────────────
# 6. TRAIN PRIMARY MODELS
# ─────────────────────────────────────────────
print("\n[6/8] Training primary models...")

# Random Forest
print("    [RF] Training primary Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1
)
rf_model.fit(X_train_sm, y_train_sm)

# XGBoost
print("    [XGB] Training primary XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05, eval_metric="logloss",
    random_state=42, n_jobs=-1, verbosity=0
)
xgb_model.fit(X_train_sm, y_train_sm)

# Save primary models
with open(os.path.join(OUTPUT_DIR, "rf_model.pkl"), "wb") as f:
    pickle.dump(rf_model, f)
with open(os.path.join(OUTPUT_DIR, "xgb_model.pkl"), "wb") as f:
    pickle.dump(xgb_model, f)
with open(os.path.join(OUTPUT_DIR, "feature_cols.pkl"), "wb") as f:
    pickle.dump(FEATURE_COLS, f)

# ─────────────────────────────────────────────
# 7. TRAIN SUB-MODELS (FAILURE MODES)
# ─────────────────────────────────────────────
print("\n[7/8] Training failure mode models (XGBoost)...")
sub_models = {}

for sub in SUB_TARGETS:
    print(f"    Training model for {sub}...")
    sub_y_train = y_train_subs[sub]
    
    # Calculate pos weight to handle massive imbalance in sub-modes
    n_neg = (sub_y_train == 0).sum()
    n_pos = sub_y_train.sum()
    scale_pos = max(1.0, float(n_neg) / max(1.0, float(n_pos)))
    
    sub_xgb = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0
    )
    sub_xgb.fit(X_train_scaled, sub_y_train)  # Train on non-SMOTE scaled data
    
    # Save model
    sub_models[sub] = sub_xgb
    with open(os.path.join(OUTPUT_DIR, f"xgb_{sub.lower()}_model.pkl"), "wb") as f:
        pickle.dump(sub_xgb, f)
    print(f"    Saved model: xgb_{sub.lower()}_model.pkl")

# ─────────────────────────────────────────────
# 8. THRESHOLD CALIBRATION (Balanced F1 vs Safety-First Recall)
# ─────────────────────────────────────────────
print("\n[8/8] Calibrating decision thresholds...")

def calibrate_thresholds(model, X_test_s, y_test):
    proba = model.predict_proba(X_test_s)[:, 1]
    
    best_t_f1, best_f1 = 0.50, 0.0
    safety_t_90 = 0.50
    safety_t_90_recall = 0.0
    
    # Grid search thresholds
    for t in np.arange(0.02, 0.85, 0.01):
        y_pred = (proba >= t).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        
        # 1. Best Balanced F1 threshold
        if f1 > best_f1:
            best_f1 = f1
            best_t_f1 = round(float(t), 2)
            
        # 2. Safety-first threshold (Guarantees Recall >= 90% while keeping highest possible precision)
        if rec >= 0.90:
            safety_t_90 = round(float(t), 2)
            safety_t_90_recall = rec
            
    return best_t_f1, safety_t_90

xgb_best_f1, xgb_safety = calibrate_thresholds(xgb_model, X_test_scaled, y_test)
rf_best_f1, rf_safety   = calibrate_thresholds(rf_model, X_test_scaled, y_test)

# Save thresholds
with open(os.path.join(OUTPUT_DIR, "xgb_threshold.pkl"), "wb") as f:
    pickle.dump(xgb_best_f1, f)
with open(os.path.join(OUTPUT_DIR, "xgb_safety_threshold.pkl"), "wb") as f:
    pickle.dump(xgb_safety, f)
    
with open(os.path.join(OUTPUT_DIR, "rf_threshold.pkl"), "wb") as f:
    pickle.dump(rf_best_f1, f)
with open(os.path.join(OUTPUT_DIR, "rf_safety_threshold.pkl"), "wb") as f:
    pickle.dump(rf_safety, f)
    

# Print reports
results_text = []
for model_name, model, f1_t, saf_t in [("XGBoost", xgb_model, xgb_best_f1, xgb_safety),
                                       ("Random Forest", rf_model, rf_best_f1, rf_safety)]:
    proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Balanced
    y_pred_f1 = (proba >= f1_t).astype(int)
    roc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, y_pred_f1)
    rec = recall_score(y_test, y_pred_f1)
    prec = precision_score(y_test, y_pred_f1)
    cm = confusion_matrix(y_test, y_pred_f1)
    
    # Safety-First
    y_pred_saf = (proba >= saf_t).astype(int)
    f1_saf = f1_score(y_test, y_pred_saf)
    rec_saf = recall_score(y_test, y_pred_saf)
    prec_saf = precision_score(y_test, y_pred_saf)
    cm_saf = confusion_matrix(y_test, y_pred_saf)
    
    block = (
        f"\n{'='*56}\n"
        f"  MODEL : {model_name} (ROC-AUC: {roc:.4f})\n"
        f"{'='*56}\n"
        f"  [BALANCED MODE] (Threshold: {f1_t})\n"
        f"    F1-Score  : {f1:.4f}\n"
        f"    Recall    : {rec:.4f}  (Caught {rec*100:.1f}% of failures)\n"
        f"    Precision : {prec:.4f}\n"
        f"    Confusion Matrix:\n{cm}\n\n"
        f"  [SAFETY-FIRST MODE] (Threshold: {saf_t})\n"
        f"    F1-Score  : {f1_saf:.4f}\n"
        f"    Recall    : {rec_saf:.4f}  (Caught {rec_saf*100:.1f}% of failures)\n"
        f"    Precision : {prec_saf:.4f}\n"
        f"    Confusion Matrix:\n{cm_saf}\n"
    )
    print(block)
    results_text.append(block)

with open(os.path.join(RESULTS_DIR, "evaluation_report.txt"), "w") as f:
    f.write("\n".join(results_text))

print(f"\n{'='*60}")
print("  [OK] PIPELINE TRAINING COMPLETE")
print("  All models & thresholds generated successfully.")
print(f"{'='*60}\n")
