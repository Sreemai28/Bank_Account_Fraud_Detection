#!/usr/bin/env python3
"""
Bank account fraud detection (BAF dataset) - clean single-file version of the notebook.

Put this file in the same folder as Base.csv and Variant I..V.csv and run:

    python bank_fraud_clean.py

Steps
  1. Load and combine Base + Variants I-V into one dataframe (df_combined)
  2. Quick audit of the combined data
  3. Feature preparation (-1 kept and flagged, categories one-hot encoded)
  4. Baseline model (age visible) with time-based split + metrics, importance, age check
  5. Fairness fixes: hide age, and per-age-band thresholds chosen on a validation month
  6. Robustness test on Base and each Variant
  7. Save the final model to fraud_model.joblib (same format the Streamlit app reads)

Why training uses only the Base rows of the combined dataframe: the variants are resampled
from the same bank data, so pooling them creates duplicate rows (which leak between train and
test) and makes x1/x2 empty for four of the six files. The variants are used for testing.
"""
from __future__ import annotations

import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# ----------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent
MODEL_PATH = DATA_DIR / "fraud_model.joblib"
DATASET_NAMES = ["Base", "Variant I", "Variant II", "Variant III", "Variant IV", "Variant V"]

CHECK_DUPLICATES = True      # slow on 6M rows; set False to skip
TARGET_FPR = 0.05            # false alarm rate each age band's threshold aims for
SAVE_MODEL = True

TARGET = "fraud_bool"
TAG = "dataset"              # column added to remember which file each row came from
MISSING_CODED = [
    "prev_address_months_count",
    "bank_months_count",
    "current_address_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]
CATEGORICAL = ["payment_type", "employment_status", "housing_status", "device_os"]
EXTRA_COLUMNS = ["x1", "x2"]                       # exist only in Variants III and V
NOT_FEATURES = [TARGET, "month", "device_fraud_count"]   # month = time index; fraud count is always 0
AGE_BINS = [0, 20, 30, 40, 50, 100]
AGE_LABELS = ["<=20", "30s", "40s", "50s", "60+"]


def banner(text: str) -> None:
    print("\n" + "=" * 90 + f"\n{text}\n" + "=" * 90)


# ----------------------------------------------------------------------------------------
# 1. Load and combine
# ----------------------------------------------------------------------------------------
def find_csv(name: str) -> Path:
    for candidate in (f"{name}.csv", f"{name.replace(' ', '_')}.csv"):
        path = DATA_DIR / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find '{name}.csv' (or '{name.replace(' ', '_')}.csv') in {DATA_DIR}")


def load_combined() -> pd.DataFrame:
    """Read all six files and stack them. Columns are aligned (x1/x2 are NaN where absent)."""
    frames = []
    for name in DATASET_NAMES:
        path = find_csv(name)
        part = pd.read_csv(path, dtype={c: "category" for c in CATEGORICAL + ["source"]})
        part[TAG] = name
        print(f"  loaded {path.name:<18} {part.shape[0]:>9,} rows x {part.shape[1]} columns")
        frames.append(part)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    for c in CATEGORICAL + ["source", TAG]:
        combined[c] = combined[c].astype("category")
    return combined


# ----------------------------------------------------------------------------------------
# 2. Audit of the combined data
# ----------------------------------------------------------------------------------------
def audit_combined(df_combined: pd.DataFrame) -> None:
    print(f"Combined shape: {df_combined.shape}")
    summary = df_combined.groupby(TAG, observed=True)[TARGET].agg(rows="size", fraud_cases="sum", fraud_rate="mean")
    summary["fraud_rate"] = (100 * summary["fraud_rate"]).round(2)
    print(summary.reindex(DATASET_NAMES).to_string())

    missing = df_combined.isnull().sum()
    missing = missing[missing > 0]
    print("\nColumns with empty cells:", dict(missing) if len(missing) else "none")
    print("(x1 and x2 exist only in Variants III and V; they are not used as features.)")

    if CHECK_DUPLICATES:
        cols = [c for c in df_combined.columns if c != TAG]
        dup = int(df_combined.duplicated(subset=cols).sum())
        print(f"\nDuplicate rows across the combined data: {dup:,}")
        print("These come from the variants being resampled from the same bank data, which is why the")
        print("model is trained on Base only and the variants are used for testing.")

    base = df_combined[df_combined[TAG] == "Base"]
    print("\nValues of -1 mean 'unknown', not a real number. Share of rows with -1 (Base):")
    for c in MISSING_CODED:
        print(f"  {c:<32} {100 * (base[c] == -1).mean():5.1f}%")
    print(f"  {'intended_balcon_amount (<0)':<32} {100 * (base['intended_balcon_amount'] < 0).mean():5.1f}%")


# ----------------------------------------------------------------------------------------
# 3. Features
# ----------------------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw columns -> model-ready columns. Unknown (-1) values are kept and flagged."""
    d = df.copy()
    for c in MISSING_CODED:
        d[c + "_missing"] = (d[c] == -1).astype(int)
    d["balcon_neg"] = (d["intended_balcon_amount"] < 0).astype(int)
    d["source"] = (d["source"].astype(str) == "TELEAPP").astype(int)
    for c in CATEGORICAL:
        d[c] = d[c].astype(str)
    return pd.get_dummies(d, columns=CATEGORICAL, drop_first=False, dtype=int)


def unique_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """One dataset's rows without the tag/extra columns, duplicates removed."""
    cols = [c for c in frame.columns if c not in [TAG] + EXTRA_COLUMNS]
    return frame[cols].drop_duplicates().reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# Helpers for thresholds and reports
# ----------------------------------------------------------------------------------------
def make_age_band(age: pd.Series) -> pd.Series:
    band = pd.cut(age, AGE_BINS, labels=AGE_LABELS).astype(object)
    return band.where(band.notna(), "Unknown")


def threshold_at_fpr(y, scores, target_fpr: float = TARGET_FPR) -> float:
    fpr, _, thr = roc_curve(y, scores)
    return float(thr[min(np.searchsorted(fpr, target_fpr), len(thr) - 1)])


def tpr_at_fpr(y, scores, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def audit_table(y, flag, band) -> pd.DataFrame:
    rows = []
    for b in AGE_LABELS:
        k = np.asarray(band == b)
        if not k.any():
            continue
        yy, ff = y[k], flag[k]
        rows.append(
            {
                "Age band": b,
                "People": int(k.sum()),
                "Fraud cases": int(yy.sum()),
                "Caught %": round(100 * ff[yy == 1].mean(), 1) if (yy == 1).any() else np.nan,
                "False alarms %": round(100 * ff[yy == 0].mean(), 1) if (yy == 0).any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def fit_model(X: pd.DataFrame, y: np.ndarray) -> HistGradientBoostingClassifier:
    """Gradient boosting; fraud cases are weighted up because they are only about 1% of rows."""
    weights = np.where(y == 1, (y == 0).sum() / (y == 1).sum(), 1.0)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    return model.fit(X, y, sample_weight=weights)


def print_core_metrics(y, p) -> dict:
    m = {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "tpr_at_1pct_fpr": tpr_at_fpr(y, p, 0.01),
        "tpr_at_5pct_fpr": tpr_at_fpr(y, p, 0.05),
    }
    print(f"ROC-AUC: {m['roc_auc']:.4f}   PR-AUC: {m['pr_auc']:.4f}")
    print(f"Fraud caught at 1% false alarms: {m['tpr_at_1pct_fpr']:.3f}   at 5%: {m['tpr_at_5pct_fpr']:.3f}")
    return m


# ----------------------------------------------------------------------------------------
# 4. Baseline model (age visible)
# ----------------------------------------------------------------------------------------
def baseline_with_age(feats: pd.DataFrame) -> None:
    train, test = feats[feats.month <= 2], feats[feats.month >= 3]
    X_train, y_train = train.drop(columns=NOT_FEATURES), train[TARGET].values
    X_test, y_test = test.drop(columns=NOT_FEATURES), test[TARGET].values
    print(f"train {len(train):,} rows (months 0-2) | test {len(test):,} rows (months 3-7) | {X_train.shape[1]} features")
    print(f"fraud rate: train {y_train.mean():.4f}, test {y_test.mean():.4f}")

    model = fit_model(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    print_core_metrics(y_test, p)

    t = threshold_at_fpr(y_test, p, 0.05)
    pred = (p >= t).astype(int)
    print("\nConfusion matrix at the 5% false alarm threshold (rows: truth, columns: prediction)")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, digits=3))

    print("Feature importance (drop in PR-AUC when a feature is shuffled; dummy columns grouped)")
    idx = X_test.sample(50_000, random_state=0).index
    r = permutation_importance(
        model, X_test.loc[idx], pd.Series(y_test, index=X_test.index).loc[idx],
        scoring="average_precision", n_repeats=3, random_state=0,
    )
    imp = pd.Series(r.importances_mean, index=X_test.columns)
    grouped = {}
    for col, val in imp.items():
        key = next((c for c in CATEGORICAL if col.startswith(c + "_")), col)
        grouped[key] = grouped.get(key, 0.0) + val
    print(pd.Series(grouped).sort_values(ascending=False).head(15).round(4).to_string())

    print("\nAge check: ONE threshold for everyone (age visible to the model)")
    res = pd.DataFrame({"age": X_test["customer_age"].values, "y": y_test, "flag": p >= t})
    rows = []
    for age, d in res.groupby("age"):
        rows.append(
            {
                "age": age,
                "people": len(d),
                "fraud_cases": int(d.y.sum()),
                "caught_%": round(100 * d.flag[d.y == 1].mean(), 1) if (d.y == 1).any() else np.nan,
                "false_alarms_%": round(100 * d.flag[d.y == 0].mean(), 1),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    print("(ages 80+ have very few people; ignore those rows)")


# ----------------------------------------------------------------------------------------
# 5. Fairness fixes: hide age, then per-age-band thresholds
# ----------------------------------------------------------------------------------------
def fairness_pipeline(feats: pd.DataFrame) -> dict:
    no_age = NOT_FEATURES + ["customer_age"]
    tr = feats[feats.month.between(0, 1)]
    va = feats[feats.month == 2]
    te = feats[feats.month >= 3]
    Xtr, ytr = tr.drop(columns=no_age), tr[TARGET].values
    Xva, yva = va.drop(columns=no_age), va[TARGET].values
    Xte, yte = te.drop(columns=no_age), te[TARGET].values
    print(f"train {len(tr):,} (months 0-1) | validation {len(va):,} (month 2) | test {len(te):,} (months 3-7)")

    model = fit_model(Xtr, ytr)
    pva = model.predict_proba(Xva)[:, 1]
    pte = model.predict_proba(Xte)[:, 1]
    metrics = print_core_metrics(yte, pte)

    band_va = make_age_band(va["customer_age"]).values
    band_te = make_age_band(te["customer_age"]).values
    thresholds = {b: threshold_at_fpr(yva[band_va == b], pva[band_va == b]) for b in AGE_LABELS}
    global_threshold = threshold_at_fpr(yva, pva)

    flag_global = pte >= global_threshold
    flag_band = np.zeros(len(yte), bool)
    for b in AGE_LABELS:
        k = band_te == b
        flag_band[k] = pte[k] >= thresholds[b]

    metrics.update(
        global_caught=float(flag_global[yte == 1].mean()),
        global_false_alarm=float(flag_global[yte == 0].mean()),
        per_band_caught=float(flag_band[yte == 1].mean()),
        per_band_false_alarm=float(flag_band[yte == 0].mean()),
    )
    eval_global = audit_table(yte, flag_global, band_te)
    eval_band = audit_table(yte, flag_band, band_te)

    print(f"\nFix 1: age hidden from the model, ONE threshold ({global_threshold:.3f})")
    print(f"overall caught {100 * metrics['global_caught']:.1f}% | false alarms {100 * metrics['global_false_alarm']:.1f}%")
    print(eval_global.to_string(index=False))

    print("\nFix 2: age hidden from the model, ONE THRESHOLD PER AGE BAND (chosen on month 2)")
    print("thresholds:", {k: round(v, 3) for k, v in thresholds.items()})
    print(f"overall caught {100 * metrics['per_band_caught']:.1f}% | false alarms {100 * metrics['per_band_false_alarm']:.1f}%")
    print(eval_band.to_string(index=False))
    print("Note: false alarms land above the 5% target because later months drift from the validation month.")

    return {
        "model": model,
        "feature_cols": list(Xtr.columns),
        "thresholds": thresholds,
        "global_threshold": global_threshold,
        "metrics": metrics,
        "eval": {"global": eval_global.to_dict("records"), "per_band": eval_band.to_dict("records")},
        "rows": {"train": len(tr), "validation": len(va), "test": len(te)},
    }


# ----------------------------------------------------------------------------------------
# 6. Robustness on Base and the variants
# ----------------------------------------------------------------------------------------
def score_frame(bundle: dict, raw: pd.DataFrame):
    X = add_features(raw).reindex(columns=bundle["feature_cols"], fill_value=0)
    scores = bundle["model"].predict_proba(X)[:, 1]
    band = make_age_band(raw["customer_age"])
    thr = band.map(bundle["thresholds"]).astype(float).fillna(bundle["global_threshold"]).values
    return scores, band.values, thr


def test_on_variants(df_combined: pd.DataFrame, bundle: dict) -> None:
    rows = []
    for name in DATASET_NAMES:
        part = unique_rows(df_combined[df_combined[TAG] == name])
        part = part[part.month >= 3]
        y = part[TARGET].values
        s, band, thr = score_frame(bundle, part)
        flag_band = s >= thr
        fa_global = [(s[(band == b) & (y == 0)] >= bundle["global_threshold"]).mean() * 100 for b in AGE_LABELS]
        fa_band = [flag_band[(band == b) & (y == 0)].mean() * 100 for b in AGE_LABELS]
        rows.append(
            {
                "data": name,
                "fraud_%": round(100 * y.mean(), 2),
                "PR-AUC": round(average_precision_score(y, s), 3),
                "TPR@5%FPR": round(tpr_at_fpr(y, s, 0.05), 3),
                "gap_one_threshold": round(max(fa_global) - min(fa_global), 1),
                "gap_per_band": round(max(fa_band) - min(fa_band), 1),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    print("gap = difference in false alarm rate (points) between the most and least flagged age band.")


# ----------------------------------------------------------------------------------------
# 7. Save
# ----------------------------------------------------------------------------------------
def save_model(bundle: dict) -> None:
    bundle.update(
        age_bins=AGE_BINS,
        age_labels=AGE_LABELS,
        target_fpr=TARGET_FPR,
        sklearn_version=sklearn.__version__,
        trained_at=datetime.now().isoformat(timespec="seconds"),
        data_file="Base.csv (months 0-1 train, month 2 thresholds)",
    )
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"Saved model to {MODEL_PATH}")


# ----------------------------------------------------------------------------------------
def main() -> None:
    start = time.time()

    banner("1. Load and combine Base + Variants I-V")
    df_combined = load_combined()

    banner("2. Audit of the combined data")
    audit_combined(df_combined)

    banner("3. Feature preparation (Base rows, duplicates removed)")
    base = unique_rows(df_combined[df_combined[TAG] == "Base"])
    feats = add_features(base)
    print(f"{len(base):,} unique Base rows -> {feats.shape[1]} columns after encoding")
    print("\nFraud rate by housing_status (codes are anonymised in the dataset):")
    print((base.groupby("housing_status", observed=True)[TARGET].agg(["size", "mean"])
           .rename(columns={"size": "people", "mean": "fraud_rate"}).round(4)).to_string())
    print("\nFraud rate by age group:")
    print((base.groupby("customer_age")[TARGET].agg(["size", "sum", "mean"])
           .rename(columns={"size": "people", "sum": "fraud_cases", "mean": "fraud_rate"}).round(4)).to_string())

    banner("4. Baseline model (age visible)")
    baseline_with_age(feats)

    banner("5. Fairness fixes (age hidden, thresholds per age band)")
    bundle = fairness_pipeline(feats)

    banner("6. Robustness: final model on Base and each Variant (months 3-7)")
    test_on_variants(df_combined, bundle)

    if SAVE_MODEL:
        banner("7. Save model")
        save_model(bundle)

    print(f"\nFinished in {(time.time() - start) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
