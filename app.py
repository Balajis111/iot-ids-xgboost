"""
app.py
------
Lightweight Explainable IDS for IoT Devices
"""

import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import time

from realtime_input import render_manual_input, generate_synthetic_flow, PROFILES
from alert_priority  import compute_priority, render_triage_card, AlertQueue

st.set_page_config(
    page_title="IoT IDS – SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
)

@st.cache_resource
def load_model():
    model     = joblib.load("models/xgboost_ids.pkl")
    explainer = shap.TreeExplainer(model)
    threshold = joblib.load("models/threshold.pkl")
    return model, explainer, threshold

model, explainer, threshold = load_model()

st.sidebar.title("🛡️ IoT IDS · SOC View")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "📝 Manual Analysis", "⚡ Live Stream", "📈 Alert Queue", "🔬 Model Insights"],
)

# ─────────────────────────────────────────────────────────────────────────────
# MANUAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
if page == "📝 Manual Analysis":
    st.title("Manual Flow Analysis")
    st.caption("Enter network flow features manually for on-demand SOC triage.")

    X_row = render_manual_input()

    if X_row is not None:
        with st.spinner("Running XGBoost + SHAP analysis..."):
            prob = model.predict_proba(X_row)[0][1]
            prediction = int(prob >= threshold)
            score, breakdown = compute_priority(model, explainer, X_row)

        if prediction == 0:
            st.success("✅ Predicted: **NORMAL** traffic")
        else:
            st.error("🚨 Predicted: **ATTACK** detected")

        st.markdown("---")
        render_triage_card(breakdown, X_row)

# ─────────────────────────────────────────────────────────────────────────────
# LIVE STREAM
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
        interval          = st.slider("Interval between flows (s)", 0.5, 5.0, 1.5, 0.5)
        n_flows           = st.number_input("Number of flows to generate", 1, 200, 20)
        show_only_attacks = st.checkbox("Show triage card only for attacks", value=True)
        run               = st.button("▶️  Start Stream", type="primary")

    if "alert_queue" not in st.session_state:
        st.session_state.alert_queue = AlertQueue(max_size=200)

    if run and attack_mix:
        queue       = st.session_state.alert_queue
        placeholder = col_r.empty()
        progress    = st.progress(0)
        status      = st.empty()

        for i in range(int(n_flows)):
            attack_type = np.random.choice(attack_mix)
            X_row       = generate_synthetic_flow(attack_type)

            prob       = model.predict_proba(X_row)[0][1]
            prediction = int(prob >= threshold)
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
# ALERT QUEUE
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
        col1.metric("Total Alerts", len(df_q))
        col2.metric("Critical",     len(df_q[df_q["severity"] == "CRITICAL"]))
        col3.metric("High",         len(df_q[df_q["severity"] == "HIGH"]))
        col4.metric("Avg Score",    f"{df_q['score'].mean():.1f}")

    queue.render()

    if st.button("🗑️ Clear Queue"):
        st.session_state.alert_queue = AlertQueue()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Dashboard":
    st.title("📊 IDS Overview Dashboard")

    uploaded_file = st.file_uploader("Upload your dataset (CSV or Parquet)", type=["csv", "parquet"])

    if uploaded_file is not None:
        import plotly.express as px
        import plotly.graph_objects as go

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_parquet(uploaded_file)

        total      = len(df)
        attacks    = int(df["label"].sum()) if "label" in df.columns else 0
        normal     = total - attacks
        attack_pct = (attacks / total) * 100

        st.markdown("""
        <style>
        .metric-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #0f3460;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        .metric-value { font-size: 2.2rem; font-weight: 700; margin: 0; }
        .metric-label { font-size: 0.85rem; color: #888; margin: 0; text-transform: uppercase; letter-spacing: 1px; }
        </style>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><p class="metric-value" style="color:#6C7BFF">{total:,}</p><p class="metric-label">Total Flows</p></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><p class="metric-value" style="color:#34C759">{normal:,}</p><p class="metric-label">Normal</p></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><p class="metric-value" style="color:#FF2D55">{attacks:,}</p><p class="metric-label">Attacks</p></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><p class="metric-value" style="color:#FF9500">{attack_pct:.1f}%</p><p class="metric-label">Attack Rate</p></div>', unsafe_allow_html=True)

        st.markdown("---")

        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.subheader("Traffic Split")
            fig_donut = go.Figure(go.Pie(
                labels=["Normal", "Attack"],
                values=[normal, attacks],
                hole=0.6,
                marker_colors=["#34C759", "#FF2D55"],
                textinfo="percent",
                hovertemplate="%{label}: %{value:,}<extra></extra>",
            ))
            fig_donut.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                annotations=[dict(
                    text=f"{attack_pct:.0f}%<br>Attack",
                    x=0.5, y=0.5,
                    font_size=16,
                    showarrow=False,
                    font_color="#FF2D55"
                )],
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_b:
            if "attack_cat" in df.columns:
                st.subheader("Attack Category Breakdown")
                cat_counts = df[df["label"] == 1]["attack_cat"].value_counts().reset_index()
                cat_counts.columns = ["Category", "Count"]
                cat_counts = cat_counts[cat_counts["Category"] != "Normal"]
                fig_cat = px.bar(
                    cat_counts, x="Count", y="Category",
                    orientation="h",
                    color="Count",
                    color_continuous_scale="oranges",
                    text="Count"
                )
                fig_cat.update_traces(textposition="outside")
                fig_cat.update_layout(
                    height=280,
                    margin=dict(l=0, r=40, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    coloraxis_showscale=False,
                    yaxis=dict(showgrid=False),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                )
                st.plotly_chart(fig_cat, use_container_width=True)

        st.markdown("---")

        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader("Top 10 Features by Variance")
            from realtime_input import FEATURES
            num_cols  = [f for f in FEATURES if f in df.columns]
            variances = df[num_cols].select_dtypes(include="number").var().sort_values(ascending=False).head(10)
            fig_var   = px.bar(
                x=variances.values,
                y=variances.index,
                orientation="h",
                color=variances.values,
                color_continuous_scale="blues",
            )
            fig_var.update_layout(
                height=300,
                margin=dict(l=0, r=20, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                coloraxis_showscale=False,
                yaxis=dict(showgrid=False),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig_var, use_container_width=True)

        with col_d:
            st.subheader("Attack vs Normal — Rate Distribution")
            if "rate" in df.columns:
                sample = df.sample(min(2000, len(df)), random_state=42)
                sample["Traffic"] = sample["label"].map({0: "Normal", 1: "Attack"})
                fig_dist = px.histogram(
                    sample, x="rate", color="Traffic",
                    nbins=60, barmode="overlay",
                    color_discrete_map={"Normal": "#34C759", "Attack": "#FF2D55"},
                    opacity=0.7,
                )
                fig_dist.update_layout(
                    height=300,
                    margin=dict(l=0, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                )
                st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("---")

        with st.expander("📋 Raw Data Preview (first 100 rows)"):
            st.dataframe(df.head(100), use_container_width=True)

    else:
        st.info("👆 Upload your UNSW-NB15 dataset above to see the overview.")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Model Insights":
    st.title("🔬 Model Explainability")
    st.caption("Global SHAP analysis of the trained XGBoost model.")

    uploaded_file = st.file_uploader("Upload your dataset to generate SHAP plots (CSV or Parquet)", type=["csv", "parquet"])

    if uploaded_file is not None:
        import matplotlib.pyplot as plt

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_parquet(uploaded_file)

        drop_cols = [c for c in ["label", "attack_cat"] if c in df.columns]
        X = df.drop(columns=drop_cols)

        from realtime_input import FEATURES
        X = X[[f for f in FEATURES if f in X.columns]]

        sample_size = min(500, len(X))
        X_sample    = X.sample(sample_size, random_state=42)

        for col in X_sample.select_dtypes(include=["category", "object"]).columns:
            X_sample[col] = X_sample[col].astype(str).astype("category").cat.codes

        with st.spinner("Generating SHAP values... (takes 30-60 seconds)"):
            shap_values = explainer.shap_values(X_sample)

        st.success(f"✅ SHAP values computed on {sample_size} samples")

        st.subheader("Top Features by Mean |SHAP|")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

        st.subheader("SHAP Value Distribution (Beeswarm)")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.subheader("Feature Importance Ranking")
        mean_shap     = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "Feature":    X_sample.columns,
            "Mean |SHAP|": mean_shap,
        }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
        importance_df.index += 1
        st.dataframe(importance_df, use_container_width=True)

    else:
        st.info("👆 Upload your dataset above to generate SHAP explainability plots.")