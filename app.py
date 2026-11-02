#!/usr/bin/env python3
"""
Bank Account Fraud Console - offline Streamlit app (single file).

Run:
    pip install streamlit pandas numpy scikit-learn joblib
    python app.py                (or: streamlit run app.py)

Files (all optional to configure, all local, nothing is sent over the network):
    Base.csv              training data, next to this file
    fraud_model.joblib    trained model, created by the app, next to this file
Override locations with environment variables:
    FRAUD_MODEL_PATH, FRAUD_DATA_PATH

Scores are model scores, not probabilities (the model is trained with class weights).
Decision support only: a human should review every flagged application.
"""
from __future__ import annotations

import os
import sys

# Launcher: "python app.py" starts Streamlit with offline-friendly flags.
if __name__ == "__main__" and "streamlit.runtime" not in sys.modules:
    import importlib.util
    import subprocess

    if importlib.util.find_spec("streamlit") is None:
        sys.exit("Streamlit is not installed. Run: pip install streamlit pandas numpy scikit-learn joblib")
    sys.exit(
        subprocess.call(
            [
                sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__),
                "--browser.gatherUsageStats=false",
                "--server.maxUploadSize=2048",
            ]
        )
    )

import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

# --------------------------------------------------------------------------------------
# Configuration: where files live
# --------------------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("FRAUD_MODEL_PATH", APP_DIR / "fraud_model.joblib"))
DATA_PATH = Path(os.environ.get("FRAUD_DATA_PATH", APP_DIR / "Base.csv"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fraud_app")

# --------------------------------------------------------------------------------------
# Pipeline (same logic as the training script)
# --------------------------------------------------------------------------------------
TARGET = "fraud_bool"
NEVER_FEATURES = [TARGET, "month", "device_fraud_count", "customer_age", "x1", "x2"]
MISSING_CODED = [
    "prev_address_months_count",
    "bank_months_count",
    "current_address_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]
CATEGORIES = {
    "payment_type": ["AA", "AB", "AC", "AD", "AE"],
    "employment_status": ["CA", "CB", "CC", "CD", "CE", "CF", "CG"],
    "housing_status": ["BA", "BB", "BC", "BD", "BE", "BF", "BG"],
    "device_os": ["linux", "macintosh", "other", "windows", "x11"],
}
REQUIRED_COLUMNS = [
    "income", "name_email_similarity", "prev_address_months_count", "current_address_months_count",
    "customer_age", "days_since_request", "intended_balcon_amount", "payment_type", "zip_count_4w",
    "velocity_6h", "velocity_24h", "velocity_4w", "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w", "employment_status", "credit_risk_score", "email_is_free",
    "housing_status", "phone_home_valid", "phone_mobile_valid", "bank_months_count", "has_other_cards",
    "proposed_credit_limit", "foreign_request", "source", "session_length_in_minutes", "device_os",
    "keep_alive_session", "device_distinct_emails_8w",
]
AGE_BINS = [0, 20, 30, 40, 50, 100]
AGE_LABELS = ["<=20", "30s", "40s", "50s", "60+"]
TRAIN_MONTHS = (0, 1)
VAL_MONTH = 2
TEST_FIRST_MONTH = 3
CHUNK = 250_000


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in MISSING_CODED:
        d[c + "_missing"] = (d[c] == -1).astype(int)
    d["balcon_neg"] = (d["intended_balcon_amount"] < 0).astype(int)
    d["source"] = (d["source"] == "TELEAPP").astype(int)
    return pd.get_dummies(d, columns=list(CATEGORIES), drop_first=False, dtype=int)


def make_age_band(age: pd.Series) -> pd.Series:
    band = pd.cut(age, AGE_BINS, labels=AGE_LABELS).astype(object)
    return band.where(band.notna(), "Unknown")


def threshold_at_fpr(y, scores, target_fpr: float) -> float:
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


def train_model(data_path: Path, target_fpr: float = 0.05, say=print) -> dict:
    say(f"Reading {data_path} ...")
    raw = pd.read_csv(data_path).drop_duplicates()
    for c in (TARGET, "month", "customer_age"):
        if c not in raw.columns:
            raise ValueError(f"Training file is missing the '{c}' column.")
    say(f"{len(raw):,} unique rows. Building features ...")
    full = add_features(raw)
    del raw

    tr = full[full.month.between(*TRAIN_MONTHS)]
    va = full[full.month == VAL_MONTH]
    te = full[full.month >= TEST_FIRST_MONTH]
    feature_cols = [c for c in full.columns if c not in NEVER_FEATURES]
    ytr, yva, yte = tr[TARGET].values, va[TARGET].values, te[TARGET].values
    say(f"Train {len(tr):,} | validation {len(va):,} | test {len(te):,} | {len(feature_cols)} features")

    say("Training gradient boosting model ...")
    weights = np.where(ytr == 1, (ytr == 0).sum() / (ytr == 1).sum(), 1.0)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    model.fit(tr[feature_cols], ytr, sample_weight=weights)
    pva = model.predict_proba(va[feature_cols])[:, 1]
    pte = model.predict_proba(te[feature_cols])[:, 1]

    say("Setting one threshold per age band on the validation month ...")
    band_va = make_age_band(va["customer_age"]).values
    band_te = make_age_band(te["customer_age"]).values
    thresholds = {b: threshold_at_fpr(yva[band_va == b], pva[band_va == b], target_fpr) for b in AGE_LABELS}
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
    say("Done.")
    return {
        "model": model,
        "feature_cols": feature_cols,
        "thresholds": thresholds,
        "global_threshold": global_threshold,
        "age_bins": AGE_BINS,
        "age_labels": AGE_LABELS,
        "target_fpr": target_fpr,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data_file": str(data_path),
        "rows": {"train": len(tr), "validation": len(va), "test": len(te)},
        "eval": {
            "global": audit_table(yte, flag_global, band_te).to_dict("records"),
            "per_band": audit_table(yte, flag_band, band_te).to_dict("records"),
        },
    }


def check_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of problems with a frame that is about to be scored (empty = OK)."""
    problems = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append("Missing columns: " + ", ".join(missing))
        return problems
    for col, allowed in CATEGORIES.items():
        unseen = sorted(set(df[col].dropna().astype(str)) - set(allowed))
        if unseen:
            problems.append(f"Unexpected values in '{col}': {', '.join(unseen[:6])}")
    if df["customer_age"].isna().any():
        problems.append("Some rows have no customer_age; they use the single global threshold.")
    return problems


def _score_chunk(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    X = add_features(df).reindex(columns=bundle["feature_cols"], fill_value=0)
    scores = bundle["model"].predict_proba(X)[:, 1]
    band = make_age_band(df["customer_age"])
    thr = band.map(bundle["thresholds"]).astype(float).fillna(bundle["global_threshold"]).values
    return pd.DataFrame(
        {"fraud_score": scores, "age_band": band.values, "threshold": thr, "flag_fraud": (scores >= thr).astype(int)},
        index=df.index,
    )


def score_frame(bundle: dict, df: pd.DataFrame, progress=None) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    parts = []
    for i in range(0, len(df), CHUNK):
        parts.append(_score_chunk(bundle, df.iloc[i:i + CHUNK]))
        if progress:
            progress(min((i + CHUNK) / len(df), 1.0))
    return pd.concat(parts) if parts else pd.DataFrame(columns=["fraud_score", "age_band", "threshold", "flag_fraud"])


# --------------------------------------------------------------------------------------
# Streamlit helpers
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_bundle(path: str, mtime: float) -> dict:
    return joblib.load(path)


def get_bundle() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    try:
        return _load_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    except Exception as exc:  # noqa: BLE001
        log.exception("Could not load model")
        st.error(f"The model file could not be loaded: {exc}")
        return None


def inject_css() -> None:
    st.markdown(
        """
        <style>
        #MainMenu, footer {visibility: hidden;}
        .block-container {padding-top: 2rem; max-width: 1200px;}
        .card {border: 1px solid rgba(128,128,128,.28); border-radius: 12px; padding: 18px 22px;
               background: rgba(128,128,128,.06); margin-bottom: 12px;}
        .card h3 {margin: 0 0 4px 0; font-size: 1.35rem;}
        .card p {margin: 0; opacity: .8;}
        .badge {display: inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 600;
                font-size: .8rem; letter-spacing: .03em; margin-bottom: 8px;}
        .badge.flag {background: rgba(220,53,69,.16); color: #dc3545; border: 1px solid rgba(220,53,69,.45);}
        .badge.ok {background: rgba(25,135,84,.16); color: #198754; border: 1px solid rgba(25,135,84,.45);}
        .muted {opacity: .7; font-size: .85rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def no_model_notice() -> None:
    st.warning(
        f"No trained model found at `{MODEL_PATH}`. Open **Model management** and train one "
        f"(needs `{DATA_PATH.name}`), or place an existing `fraud_model.joblib` at that path."
    )


def show_table(records_or_df) -> None:
    df = pd.DataFrame(records_or_df)
    st.dataframe(df, hide_index=True)


# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------
def page_overview(bundle: dict | None) -> None:
    st.title("Overview")
    st.caption("How the model performs on held-out months and how the decision thresholds are set.")
    if bundle is None:
        no_model_notice()
        return

    m = bundle["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{m['roc_auc']:.3f}", help="Chance a fraud case scores above a good one. 0.5 is a coin flip.")
    c2.metric("PR-AUC", f"{m['pr_auc']:.3f}", help="Precision/recall quality; fraud is rare, so a random model scores about 0.01.")
    c3.metric("Fraud caught at 1% false alarms", f"{100 * m['tpr_at_1pct_fpr']:.1f}%")
    c4.metric("Fraud caught at 5% false alarms", f"{100 * m['tpr_at_5pct_fpr']:.1f}%")

    st.subheader("Decision thresholds by age band")
    st.caption(
        f"Each band's threshold targets about {100 * bundle.get('target_fpr', 0.05):.0f}% false alarms on a validation month. "
        "A score at or above the threshold is flagged for review."
    )
    thr = pd.DataFrame({"Age band": list(bundle["thresholds"]), "Threshold": list(bundle["thresholds"].values())})
    left, right = st.columns([1, 1])
    with left:
        show_table(thr.round(3))
        st.caption(f"Global single threshold: {bundle['global_threshold']:.3f}")
    with right:
        st.bar_chart(thr.set_index("Age band"), height=220)

    ev = bundle.get("eval")
    if ev:
        st.subheader("Fairness check on test months")
        a, b = st.columns(2)
        with a:
            st.markdown(
                f"**One threshold for everyone**  \n"
                f"caught {100 * m['global_caught']:.1f}% | false alarms {100 * m['global_false_alarm']:.1f}%"
            )
            show_table(ev["global"])
        with b:
            st.markdown(
                f"**One threshold per age band**  \n"
                f"caught {100 * m['per_band_caught']:.1f}% | false alarms {100 * m['per_band_false_alarm']:.1f}%"
            )
            show_table(ev["per_band"])
        st.caption(
            "With a single threshold, older customers are flagged wrongly far more often. Band thresholds even this out "
            "at the cost of catching somewhat less fraud overall."
        )

    with st.expander("Limitations and responsible use"):
        st.markdown(
            "- Scores are not probabilities. Use them to rank and to compare with the threshold.\n"
            "- `customer_age` is not a model input, but it selects the threshold. Age-based decision rules may be "
            "restricted in real banking; this tool is for analysis and decision support.\n"
            "- Thresholds drift over time. False alarms on later months run above the validation target, so re-train "
            "or re-calibrate regularly.\n"
            "- Other features (housing, address history) partly encode age, so hiding age alone does not remove the effect."
        )


def applicant_form() -> pd.DataFrame | None:
    with st.form("applicant_form"):
        with st.expander("Applicant profile", expanded=True):
            c1, c2, c3 = st.columns(3)
            age = c1.select_slider(
                "Age group (decade)", options=[10, 20, 30, 40, 50, 60, 70, 80, 90], value=30,
                help="Coded as in the dataset: 30 means 30 to 39.",
            )
            income = c2.slider("Income (normalised)", 0.1, 0.9, 0.6, 0.1)
            employment = c3.selectbox("Employment status", CATEGORIES["employment_status"])
            housing = c1.selectbox("Housing status", CATEGORIES["housing_status"])
            credit_risk = c2.number_input("Credit risk score", -200, 400, 100, 1)
            credit_limit = c3.number_input("Proposed credit limit", 190.0, 2100.0, 200.0, 10.0)

        with st.expander("Contact and identity", expanded=False):
            c1, c2, c3 = st.columns(3)
            name_sim = c1.slider("Name / email similarity", 0.0, 1.0, 0.5, 0.01)
            email_free = c2.checkbox("Free email provider", value=True)
            phone_home = c3.checkbox("Home phone valid", value=False)
            phone_mobile = c1.checkbox("Mobile phone valid", value=True)
            dob_emails = c2.number_input("Emails sharing this birth date (4 weeks)", 0, 50, 10, 1)

        with st.expander("Address and banking", expanded=False):
            c1, c2, c3 = st.columns(3)
            prev_addr = c1.number_input("Months at previous address", 0, 400, 0, 1)
            prev_unknown = c1.checkbox("Previous address unknown", value=True)
            cur_addr = c2.number_input("Months at current address", 0, 450, 55, 1)
            cur_unknown = c2.checkbox("Current address unknown", value=False)
            bank_months = c3.number_input("Months with bank", 0, 40, 6, 1)
            bank_unknown = c3.checkbox("Bank history unknown", value=False)
            branch = c1.number_input("Branch applications (8 weeks)", 0, 2500, 10, 1)
            other_cards = c2.checkbox("Has other cards", value=False)
            foreign = c3.checkbox("Foreign request", value=False)
            balcon = c1.number_input("Intended initial transfer amount", 0.0, 120.0, 0.0, 0.1)
            balcon_unknown = c1.checkbox("Transfer amount not provided", value=False)
            payment = c2.selectbox("Payment type", CATEGORIES["payment_type"])

        with st.expander("Session and device", expanded=False):
            c1, c2, c3 = st.columns(3)
            source = c1.selectbox("Channel", ["INTERNET", "TELEAPP"])
            device_os = c2.selectbox("Device OS", CATEGORIES["device_os"], index=3)
            keep_alive = c3.checkbox("Keep-alive session", value=True)
            session_len = c1.number_input("Session length (minutes)", 0.0, 90.0, 5.4, 0.1)
            session_unknown = c1.checkbox("Session length unknown", value=False)
            dev_emails = c2.number_input("Emails seen on device (8 weeks)", 0, 5, 1, 1)
            dev_unknown = c2.checkbox("Device email count unknown", value=False)
            days_req = c3.number_input("Days since request", 0.0, 80.0, 0.02, 0.01)

        with st.expander("Velocity", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            zip_count = c1.number_input("Applications from ZIP (4 weeks)", 0, 7000, 1500, 1)
            v6 = c2.number_input("Velocity 6h", -50.0, 18000.0, 7000.0, 10.0)
            v24 = c3.number_input("Velocity 24h", 1000.0, 10000.0, 5700.0, 10.0)
            v4w = c4.number_input("Velocity 4 weeks", 2500.0, 7500.0, 5600.0, 10.0)

        submitted = st.form_submit_button("Score applicant", type="primary")

    if not submitted:
        return None
    row = {
        "income": income, "name_email_similarity": name_sim,
        "prev_address_months_count": -1 if prev_unknown else prev_addr,
        "current_address_months_count": -1 if cur_unknown else cur_addr,
        "customer_age": age, "days_since_request": days_req,
        "intended_balcon_amount": -1.0 if balcon_unknown else balcon,
        "payment_type": payment, "zip_count_4w": zip_count,
        "velocity_6h": v6, "velocity_24h": v24, "velocity_4w": v4w,
        "bank_branch_count_8w": branch, "date_of_birth_distinct_emails_4w": dob_emails,
        "employment_status": employment, "credit_risk_score": credit_risk,
        "email_is_free": int(email_free), "housing_status": housing,
        "phone_home_valid": int(phone_home), "phone_mobile_valid": int(phone_mobile),
        "bank_months_count": -1 if bank_unknown else bank_months,
        "has_other_cards": int(other_cards), "proposed_credit_limit": credit_limit,
        "foreign_request": int(foreign), "source": source,
        "session_length_in_minutes": -1.0 if session_unknown else session_len,
        "device_os": device_os, "keep_alive_session": int(keep_alive),
        "device_distinct_emails_8w": -1 if dev_unknown else dev_emails,
    }
    return pd.DataFrame([row])


def page_single(bundle: dict | None) -> None:
    st.title("Score an applicant")
    st.caption("Enter one application and see whether it would be flagged for review.")
    if bundle is None:
        no_model_notice()
        return

    row = applicant_form()
    if row is not None:
        try:
            res = score_frame(bundle, row).iloc[0]
            st.session_state["single_result"] = res.to_dict()
        except Exception as exc:  # noqa: BLE001
            log.exception("Single scoring failed")
            st.error(f"Could not score this applicant: {exc}")
            return

    res = st.session_state.get("single_result")
    if not res:
        return
    flagged = bool(res["flag_fraud"])
    badge = '<span class="badge flag">FLAG FOR REVIEW</span>' if flagged else '<span class="badge ok">NO FLAG</span>'
    msg = ("Score is at or above the threshold for this age band." if flagged
           else "Score is below the threshold for this age band.")
    st.markdown(f'<div class="card">{badge}<h3>Score {res["fraud_score"]:.3f} vs threshold {res["threshold"]:.3f}</h3>'
                f'<p>{msg} Age band: {res["age_band"]}.</p></div>', unsafe_allow_html=True)
    st.progress(float(min(max(res["fraud_score"], 0.0), 1.0)))
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud score", f"{res['fraud_score']:.3f}")
    c2.metric("Threshold", f"{res['threshold']:.3f}")
    c3.metric("Margin", f"{res['fraud_score'] - res['threshold']:+.3f}")
    st.markdown('<p class="muted">Model score, not a probability. A flag means a person should review the application.</p>',
                unsafe_allow_html=True)


def read_batch_source() -> tuple[pd.DataFrame | None, str]:
    mode = st.radio("Data source", ["Upload a CSV", "CSV file on this computer"], horizontal=True)
    if mode == "Upload a CSV":
        up = st.file_uploader("CSV with the application columns", type=["csv"])
        if up is None:
            return None, ""
        return pd.read_csv(up), up.name
    default = str(DATA_PATH) if DATA_PATH.exists() else ""
    path = st.text_input("Full path to the CSV file", value=default)
    if not path:
        return None, ""
    p = Path(path).expanduser()
    if not p.is_file():
        st.error(f"File not found: {p}")
        return None, ""
    return None, str(p)


def page_batch(bundle: dict | None) -> None:
    st.title("Batch scoring")
    st.caption("Score a whole file, review the flagged applications, and audit fairness if the file has labels.")
    if bundle is None:
        no_model_notice()
        return

    df, label = read_batch_source()
    if st.button("Run scoring", type="primary"):
        try:
            with st.spinner("Reading file ..."):
                if df is None:
                    if not label:
                        st.error("Choose a file first.")
                        return
                    df = pd.read_csv(label)
                df = df.drop_duplicates().reset_index(drop=True)
            problems = check_columns(df)
            blocking = [p for p in problems if p.startswith("Missing")]
            for p in problems:
                (st.error if p in blocking else st.warning)(p)
            if blocking:
                return
            bar = st.progress(0.0, text="Scoring ...")
            res = score_frame(bundle, df, progress=lambda f: bar.progress(f, text="Scoring ..."))
            bar.empty()
            st.session_state["batch"] = {"df": df, "res": res, "name": Path(label).name if label else "upload"}
            st.session_state.pop("csv_bytes", None)
        except Exception as exc:  # noqa: BLE001
            log.exception("Batch scoring failed")
            st.error(f"Scoring failed: {exc}")
            return

    state = st.session_state.get("batch")
    if not state:
        return
    df, res = state["df"], state["res"]
    flagged = int(res["flag_fraud"].sum())

    st.subheader(f"Results for {state['name']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Applications scored", f"{len(res):,}")
    c2.metric("Flagged for review", f"{flagged:,}")
    c3.metric("Flag rate", f"{100 * flagged / max(len(res), 1):.2f}%")

    left, right = st.columns(2)
    with left:
        st.markdown("**Score distribution**")
        counts, edges = np.histogram(res["fraud_score"], bins=40, range=(0, 1))
        st.bar_chart(pd.DataFrame({"applications": counts}, index=[f"{e:.2f}" for e in edges[:-1]]), height=240)
    with right:
        st.markdown("**Flag rate by age band (%)**")
        rate = (res.groupby("age_band")["flag_fraud"].mean() * 100).reindex(AGE_LABELS).dropna()
        st.bar_chart(rate.rename("flag rate %"), height=240)

    st.markdown("**Highest-scoring applications**")
    top = pd.concat([res, df], axis=1).sort_values("fraud_score", ascending=False).head(100)
    st.dataframe(top, hide_index=True)

    if TARGET in df.columns:
        st.subheader("Audit against known outcomes")
        y = df[TARGET].values.astype(int)
        band = res["age_band"].values
        flag = res["flag_fraud"].values.astype(bool)
        flag_g = res["fraud_score"].values >= bundle["global_threshold"]
        a, b = st.columns(2)
        with a:
            st.markdown(f"**Age-band thresholds**  \ncaught {100 * flag[y == 1].mean():.1f}% | "
                        f"false alarms {100 * flag[y == 0].mean():.1f}%")
            show_table(audit_table(y, flag, band))
        with b:
            st.markdown(f"**Single threshold**  \ncaught {100 * flag_g[y == 1].mean():.1f}% | "
                        f"false alarms {100 * flag_g[y == 0].mean():.1f}%")
            show_table(audit_table(y, flag_g, band))

    st.subheader("Export")
    if st.button("Prepare CSV with scores"):
        with st.spinner("Preparing ..."):
            st.session_state["csv_bytes"] = pd.concat([df, res], axis=1).to_csv(index=False).encode("utf-8")
    if "csv_bytes" in st.session_state:
        st.download_button("Download scored CSV", st.session_state["csv_bytes"],
                           file_name="scored_applications.csv", mime="text/csv")


def page_model(bundle: dict | None) -> None:
    st.title("Model management")
    st.caption("Where the files live, what is loaded, and how to retrain.")

    st.subheader("Files")
    rows = [
        {"Item": "App folder", "Location": str(APP_DIR), "Status": "ok"},
        {"Item": "Model file", "Location": str(MODEL_PATH), "Status": "found" if MODEL_PATH.exists() else "missing"},
        {"Item": "Training data", "Location": str(DATA_PATH), "Status": "found" if DATA_PATH.exists() else "missing"},
    ]
    show_table(rows)
    st.caption("Change locations with the FRAUD_MODEL_PATH and FRAUD_DATA_PATH environment variables.")

    if bundle is not None:
        st.subheader("Loaded model")
        c1, c2, c3 = st.columns(3)
        c1.metric("Trained", bundle.get("trained_at", "unknown"))
        c2.metric("Features", len(bundle["feature_cols"]))
        c3.metric("File size", f"{MODEL_PATH.stat().st_size / 1e6:.1f} MB")
        if bundle.get("sklearn_version") != sklearn.__version__:
            st.warning(f"Model was saved with scikit-learn {bundle.get('sklearn_version')}; "
                       f"this environment has {sklearn.__version__}. Retrain if you see errors.")
        if bundle.get("rows"):
            st.caption(f"Rows used: {bundle['rows']['train']:,} train, {bundle['rows']['validation']:,} validation, "
                       f"{bundle['rows']['test']:,} test.")

    st.subheader("Train a new model")
    data_text = st.text_input("Training CSV (Base.csv format)", value=str(DATA_PATH))
    fpr = st.slider("Target false-alarm rate per age band", 0.01, 0.10, 0.05, 0.01)
    st.caption("Training reads the whole file and takes a few minutes. The current model is replaced when it finishes.")
    confirm = st.checkbox("Replace the current model file")
    if st.button("Start training", type="primary", disabled=not confirm):
        data = Path(data_text).expanduser()
        if not data.is_file():
            st.error(f"File not found: {data}")
            return
        try:
            with st.status("Training ...", expanded=True) as status:
                new_bundle = train_model(data, fpr, say=st.write)
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                tmp = MODEL_PATH.with_suffix(".tmp")
                joblib.dump(new_bundle, tmp, compress=3)
                os.replace(tmp, MODEL_PATH)
                status.update(label="Training complete", state="complete")
            st.cache_resource.clear()
            st.session_state.pop("single_result", None)
            st.session_state.pop("batch", None)
            st.success("Model saved. Open Overview to see the results.")
        except Exception as exc:  # noqa: BLE001
            log.exception("Training failed")
            st.error(f"Training failed: {exc}")


# --------------------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Fraud Console", page_icon=":shield:", layout="wide")
    inject_css()
    bundle = get_bundle()

    with st.sidebar:
        st.markdown("### Fraud Console")
        st.caption("Bank account fraud detection. Runs fully offline.")
        page = st.radio("Navigate", ["Overview", "Score an applicant", "Batch scoring", "Model management"],
                        label_visibility="collapsed")
        st.divider()
        if bundle is None:
            st.caption("Model: not loaded")
        else:
            st.caption(f"Model: loaded ({bundle.get('trained_at', 'date unknown')})")
        st.caption("Decision support only. Review every flag.")

    if page == "Overview":
        page_overview(bundle)
    elif page == "Score an applicant":
        page_single(bundle)
    elif page == "Batch scoring":
        page_batch(bundle)
    else:
        page_model(bundle)


if __name__ == "__main__":
    main()
