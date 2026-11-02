#!/usr/bin/env python3
"""
Bank account fraud detection (BAF dataset).

What this script does
---------------------
* Cleans the data: -1 codes are treated as "missing" (kept, plus a flag column),
  duplicates are dropped, month / device_fraud_count are never used as features.
* Splits by time: months 0-1 train, month 2 validation, months 3-7 test.
* Trains a gradient boosting model WITHOUT customer_age as an input.
* Sets one decision threshold per age band on the validation month (target false
  positive rate, default 5%), so good customers are flagged at similar rates
  across age groups. Customer_age is still needed at scoring time to pick the band.
* Saves everything needed for scoring into one joblib file.

Simplest use
------------
Put this file in the same folder as Base.csv and run:

    python fraud_pipeline.py

That trains the model and creates fraud_model.joblib next to the script.

Other commands
--------------
    python fraud_pipeline.py train    --data Base.csv --out fraud_model.joblib
    python fraud_pipeline.py predict  --model fraud_model.joblib --input new.csv --output scored.csv
    python fraud_pipeline.py variants --model fraud_model.joblib --data-dir .

In Python:
    from fraud_pipeline import load_model, score
    bundle = load_model("fraud_model.joblib")
    result = score(bundle, df)      # df has the raw BAF columns
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

TARGET = "fraud_bool"
NEVER_FEATURES = [TARGET, "month", "device_fraud_count", "customer_age", "x1", "x2"]
MISSING_CODED = [
    "prev_address_months_count",
    "bank_months_count",
    "current_address_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]
CATEGORICAL = ["payment_type", "employment_status", "housing_status", "device_os"]
AGE_BINS = [0, 20, 30, 40, 50, 100]
AGE_LABELS = ["<=20", "30s", "40s", "50s", "60+"]

TRAIN_MONTHS = (0, 1)
VAL_MONTH = 2
TEST_FIRST_MONTH = 3


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw BAF columns -> numeric feature frame (all columns, before selection)."""
    d = df.copy()
    for c in MISSING_CODED:
        d[c + "_missing"] = (d[c] == -1).astype(int)
    d["balcon_neg"] = (d["intended_balcon_amount"] < 0).astype(int)
    d["source"] = (d["source"] == "TELEAPP").astype(int)
    return pd.get_dummies(d, columns=CATEGORICAL, drop_first=False, dtype=int)


def make_age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, AGE_BINS, labels=AGE_LABELS)


def threshold_at_fpr(y, scores, target_fpr: float) -> float:
    fpr, _, thr = roc_curve(y, scores)
    return float(thr[min(np.searchsorted(fpr, target_fpr), len(thr) - 1)])


def tpr_at_fpr(y, scores, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def band_report(y, flag, band) -> pd.DataFrame:
    rows = []
    for b in AGE_LABELS:
        k = np.asarray(band == b)
        yy, ff = y[k], flag[k]
        rows.append(
            {
                "age_band": b,
                "people": int(k.sum()),
                "fraud_cases": int(yy.sum()),
                "caught_%": round(100 * ff[yy == 1].mean(), 1) if (yy == 1).any() else np.nan,
                "false_alarm_%": round(100 * ff[yy == 0].mean(), 1),
            }
        )
    return pd.DataFrame(rows)


def train(data_path: str, out_path: str, target_fpr: float) -> None:
    raw = pd.read_csv(data_path).drop_duplicates()
    full = add_features(raw)

    tr = full[full.month.between(*TRAIN_MONTHS)]
    va = full[full.month == VAL_MONTH]
    te = full[full.month >= TEST_FIRST_MONTH]

    feature_cols = [c for c in full.columns if c not in NEVER_FEATURES]
    Xtr, ytr = tr[feature_cols], tr[TARGET].values
    Xva, yva = va[feature_cols], va[TARGET].values
    Xte, yte = te[feature_cols], te[TARGET].values

    print(f"train {len(tr):,} | validation {len(va):,} | test {len(te):,} | features {len(feature_cols)}")
    print(f"fraud rate: train {ytr.mean():.4f}, validation {yva.mean():.4f}, test {yte.mean():.4f}")

    weights = np.where(ytr == 1, (ytr == 0).sum() / (ytr == 1).sum(), 1.0)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    model.fit(Xtr, ytr, sample_weight=weights)

    pva = model.predict_proba(Xva)[:, 1]
    pte = model.predict_proba(Xte)[:, 1]

    band_va = make_age_band(va["customer_age"]).astype(str).values
    band_te = make_age_band(te["customer_age"]).astype(str).values
    thresholds = {
        b: threshold_at_fpr(yva[band_va == b], pva[band_va == b], target_fpr) for b in AGE_LABELS
    }
    global_threshold = threshold_at_fpr(yva, pva, target_fpr)

    flag_band = np.zeros(len(yte), bool)
    for b in AGE_LABELS:
        k = band_te == b
        flag_band[k] = pte[k] >= thresholds[b]
    flag_global = pte >= global_threshold

    metrics = {
        "roc_auc": float(roc_auc_score(yte, pte)),
        "pr_auc": float(average_precision_score(yte, pte)),
        "tpr_at_1pct_fpr": tpr_at_fpr(yte, pte, 0.01),
        "tpr_at_5pct_fpr": tpr_at_fpr(yte, pte, 0.05),
        "per_band_caught": float(flag_band[yte == 1].mean()),
        "per_band_false_alarm": float(flag_band[yte == 0].mean()),
        "global_caught": float(flag_global[yte == 1].mean()),
        "global_false_alarm": float(flag_global[yte == 0].mean()),
    }

    print("\nTest months 3-7")
    print(f"  ROC-AUC {metrics['roc_auc']:.4f} | PR-AUC {metrics['pr_auc']:.4f}")
    print(f"  fraud caught at 1% FPR {metrics['tpr_at_1pct_fpr']:.3f} | at 5% FPR {metrics['tpr_at_5pct_fpr']:.3f}")
    print(f"\nOne threshold for everyone (caught {100 * metrics['global_caught']:.1f}%, "
          f"false alarms {100 * metrics['global_false_alarm']:.1f}%)")
    print(band_report(yte, flag_global, band_te).to_string(index=False))
    print(f"\nPer age band thresholds (caught {100 * metrics['per_band_caught']:.1f}%, "
          f"false alarms {100 * metrics['per_band_false_alarm']:.1f}%)")
    print(band_report(yte, flag_band, band_te).to_string(index=False))

    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "thresholds": thresholds,
        "global_threshold": global_threshold,
        "age_bins": AGE_BINS,
        "age_labels": AGE_LABELS,
        "target_fpr": target_fpr,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, out_path, compress=3)
    print(f"\nSaved model bundle to {out_path}")


def load_model(path: str) -> dict:
    bundle = joblib.load(path)
    if bundle.get("sklearn_version") != sklearn.__version__:
        print(
            f"Warning: model was saved with scikit-learn {bundle.get('sklearn_version')}, "
            f"you have {sklearn.__version__}.",
            file=sys.stderr,
        )
    return bundle


def score(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Score raw BAF rows. Needs the raw columns including customer_age."""
    if "customer_age" not in df.columns:
        raise ValueError("customer_age is required to choose the age-band threshold.")
    feats = add_features(df)
    X = feats.reindex(columns=bundle["feature_cols"], fill_value=0)
    scores = bundle["model"].predict_proba(X)[:, 1]

    band = pd.cut(df["customer_age"], bundle["age_bins"], labels=bundle["age_labels"]).astype(str)
    thr = band.map(bundle["thresholds"]).astype(float)
    thr = thr.fillna(bundle["global_threshold"]).values

    return pd.DataFrame(
        {
            "fraud_score": scores,
            "age_band": band.values,
            "threshold": thr,
            "flag_fraud": (scores >= thr).astype(int),
        },
        index=df.index,
    )


def predict(model_path: str, input_path: str, output_path: str) -> None:
    bundle = load_model(model_path)
    df = pd.read_csv(input_path)
    result = score(bundle, df)
    out = pd.concat([df, result], axis=1)
    out.to_csv(output_path, index=False)
    print(f"Scored {len(df):,} rows. Flagged {int(result.flag_fraud.sum()):,} as fraud.")
    print(f"Saved to {output_path}")


def variants(model_path: str, data_dir: str) -> None:
    bundle = load_model(model_path)
    rows = []
    for name in ["Base", "Variant I", "Variant II", "Variant III", "Variant IV", "Variant V"]:
        path = Path(data_dir) / f"{name}.csv"
        if not path.exists():
            print(f"Skipping {name}: {path} not found")
            continue
        df = pd.read_csv(path).drop_duplicates()
        df = df[df.month >= TEST_FIRST_MONTH]
        res = score(bundle, df)
        y = df[TARGET].values
        s = res.fraud_score.values
        flag = res.flag_fraud.values.astype(bool)
        fa = [flag[(res.age_band.values == b) & (y == 0)].mean() * 100 for b in AGE_LABELS]
        rows.append(
            {
                "data": name,
                "fraud_%": round(100 * y.mean(), 2),
                "PR-AUC": round(average_precision_score(y, s), 3),
                "TPR@5%FPR": round(tpr_at_fpr(y, s, 0.05), 3),
                "caught_%": round(100 * flag[y == 1].mean(), 1),
                "false_alarm_%": round(100 * flag[y == 0].mean(), 1),
                "age_gap_pts": round(max(fa) - min(fa), 1),
            }
        )
        print(rows[-1], flush=True)
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bank fraud detection pipeline")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", help="train and save the model")
    p_train.add_argument("--data", default="Base.csv")
    p_train.add_argument("--out", default="fraud_model.joblib")
    p_train.add_argument("--fpr", type=float, default=0.05, help="target false positive rate per age band")

    p_pred = sub.add_parser("predict", help="score a csv of raw BAF rows")
    p_pred.add_argument("--model", default="fraud_model.joblib")
    p_pred.add_argument("--input", required=True)
    p_pred.add_argument("--output", default="scored.csv")

    p_var = sub.add_parser("variants", help="evaluate a saved model on Base and Variants I-V")
    p_var.add_argument("--model", default="fraud_model.joblib")
    p_var.add_argument("--data-dir", default=".")

    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    if args.cmd is None:
        data = here / "Base.csv"
        if not data.exists():
            sys.exit(f"Could not find {data}. Put Base.csv in the same folder as this script.")
        train(str(data), str(here / "fraud_model.joblib"), 0.05)
    elif args.cmd == "train":
        train(args.data, args.out, args.fpr)
    elif args.cmd == "predict":
        predict(args.model, args.input, args.output)
    else:
        variants(args.model, args.data_dir)


if __name__ == "__main__":
    main()
