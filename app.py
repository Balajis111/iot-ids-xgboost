"""
app.py  (updated)
-----------------
Lightweight Explainable IDS for IoT Devices
Integrates:
  - Real-time / manual flow input  (realtime_input.py)
  - SHAP-weighted alert priority    (alert_priority.py)

Drop-in additions to your existing app — the marked sections
show exactly where new code plugs in.
"""

import streamlit as st
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import joblib
import time

# NEW imports
from realtime_input import render_manual_input, generate_synthetic_flow, PROFILES
from alert_priority  import compute_priority, render_triage_card, AlertQueue

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IoT IDS – SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL + EXPLAINER  (your existing code, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("model/xgboost_ids.pkl")       # your trained model
    explainer = shap.TreeExplainer(model)
    return model, explainer

model, explainer = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image("assets/logo.png", use_column_width=True)  # optional
st.sidebar.title("🛡️ IoT IDS · SOC View")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "📝 Manual Analysis", "⚡ Live Stream", "📈 Alert Queue", "🔬 Model Insights"],
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MANUAL ANALYSIS  ← NEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "📝 Manual Analysis":
    st.title("Manual Flow Analysis")
    st.caption("Enter network flow features manually for on-demand SOC triage.")

    X_row = render_manual_input()

    if X_row is not None:
        with st.spinner("Running XGBoost + SHAP analysis..."):
            prediction = int(model.predict(X_row)[0])
            score, breakdown = compute_priority(model, explainer, X_row)

        if prediction == 0:
            st.success("✅ Predicted: **NORMAL** traffic")
        else:
            st.error("🚨 Predicted: **ATTACK** detected")

        st.markdown("---")
        render_triage_card(breakdown, X_row)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: LIVE STREAM  ← NEW
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚡ Live Stream":
    st.title("⚡ Simulated Live Flow Stream")
    st.caption(
        "Simulates real-time IoT network flows. "
        "In production, replace `generate_synthetic_flow()` with your "
        "live packet capture / CICFlowMeter feed."
    )

    col_l, col_r = st.columns([1, 2])

    with col_l:
        attack_mix = st.multiselect(
            "Include attack types",
            list(PROFILES.keys()),
            default=["Normal", "DoS", "Scan"],
        )
        interval   = st.slider("Interval between flows (s)", 0.5, 5.0, 1.5, 0.5)
        n_flows    = st.number_input("Number of flows to generate", 1, 200, 20)
        show_only_attacks = st.checkbox("Show triage card only for attacks", value=True)
        run = st.button("▶️  Start Stream", type="primary")

    # Initialise alert queue in session state
    if "alert_queue" not in st.session_state:
        st.session_state.alert_queue = AlertQueue(max_size=200)

    if run and attack_mix:
        queue = st.session_state.alert_queue
        placeholder = col_r.empty()
        progress = st.progress(0)
        status   = st.empty()

        for i in range(int(n_flows)):
            attack_type = np.random.choice(attack_mix)
            X_row = generate_synthetic_flow(attack_type)

            with st.spinner(""):
                prediction = int(model.predict(X_row)[0])
                score, breakdown = compute_priority(model, explainer, X_row)

            flow_id = f"F{i+1:04d}"
            queue.add(flow_id, X_row, prediction, breakdown)

            status.markdown(
                f"**Flow {flow_id}** | Type: `{attack_type}` | "
                f"Prediction: {'🔴 Attack' if prediction else '🟢 Normal'} | "
                f"Priority: `{score:.1f}`"
            )

            if prediction or not show_only_attacks:
                with placeholder.container():
                    render_triage_card(breakdown, X_row, show_gauge=False)

            progress.progress((i + 1) / int(n_flows))
            time.sleep(interval)

        st.success(f"✅ Stream complete. {int(n_flows)} flows analysed.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ALERT QUEUE  ← NEW
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Alert Queue":
    st.title("📈 Alert Priority Queue")
    st.caption("All analysed flows sorted by SHAP-weighted priority score.")

    if "alert_queue" not in st.session_state:
        st.session_state.alert_queue = AlertQueue()

    queue = st.session_state.alert_queue

    col1, col2, col3, col4 = st.columns(4)
    df_q = queue.to_dataframe()
    if not df_q.empty:
        col1.metric("Total Alerts",   len(df_q))
        col2.metric("Critical",       len(df_q[df_q["severity"] == "CRITICAL"]))
        col3.metric("High",           len(df_q[df_q["severity"] == "HIGH"]))
        col4.metric("Avg Score",      f"{df_q['score'].mean():.1f}")

    queue.render()

    if st.button("🗑️ Clear Queue"):
        st.session_state.alert_queue = AlertQueue()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DASHBOARD  (your existing dashboard code goes here, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Dashboard":
    st.title("📊 IDS Overview Dashboard")
    st.info("Your existing dashboard page — plug your current code here.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL INSIGHTS  (your existing SHAP global plots, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Model Insights":
    st.title("🔬 Model Explainability")
    st.info("Your existing SHAP summary plots go here.")
