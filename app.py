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
    ["📊 Dashboard", "📝 Manual Analysis", "⚡ Live Stream", "📈 Alert Queue", "🔬 Model Insights", "🎯 Attack Type Detector"],
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
    # ── Export PDF ────────────────────────────────────────────────────────────
    if not df_q.empty and st.button("📄 Export PDF Report"):
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        import io

        buffer = io.BytesIO()
        doc    = SimpleDocTemplate(buffer, pagesize=A4,
                                   rightMargin=15*mm, leftMargin=15*mm,
                                   topMargin=20*mm, bottomMargin=20*mm)
        styles = getSampleStyleSheet()
        story  = []

        # Title
        title_style = ParagraphStyle("title", parent=styles["Title"],
                                     fontSize=18, textColor=colors.HexColor("#1a1a2e"),
                                     spaceAfter=6)
        story.append(Paragraph("IoT IDS — SOC Triage Report", title_style))
        story.append(Paragraph(
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"]
        ))
        story.append(Spacer(1, 8*mm))

        # Summary metrics
        total_a   = len(df_q)
        critical  = len(df_q[df_q["severity"] == "CRITICAL"])
        high      = len(df_q[df_q["severity"] == "HIGH"])
        avg_score = df_q["score"].mean()

        summary_data = [
            ["Metric", "Value"],
            ["Total Alerts",    str(total_a)],
            ["Critical",        str(critical)],
            ["High",            str(high)],
            ["Average Score",   f"{avg_score:.1f}"],
        ]
        summary_table = Table(summary_data, colWidths=[80*mm, 80*mm])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 10),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f5f5f5"), colors.white]),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.grey),
            ("PADDING",     (0,0), (-1,-1), 6),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 8*mm))

        # Alert table
        story.append(Paragraph("Alert Queue — Sorted by Priority Score", styles["Heading2"]))
        story.append(Spacer(1, 4*mm))

        table_data = [["ID", "Timestamp", "Prediction", "Severity", "Score", "Confidence", "Top Feature"]]
        for _, row in df_q.iterrows():
            table_data.append([
                str(row["id"]),
                str(row["timestamp"])[:19],
                str(row["prediction"]),
                str(row["severity"]),
                str(row["score"]),
                str(row["confidence"]),
                str(row["top_feature"]),
            ])

        col_widths = [20*mm, 40*mm, 25*mm, 22*mm, 18*mm, 22*mm, 25*mm]
        alert_table = Table(table_data, colWidths=col_widths, repeatRows=1)
        alert_table.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#fff0f0"), colors.white]),
            ("GRID",        (0,0), (-1,-1), 0.3, colors.grey),
            ("PADDING",     (0,0), (-1,-1), 4),
        ]))
        story.append(alert_table)

        doc.build(story)
        buffer.seek(0)

        st.download_button(
            label="⬇️ Download PDF Report",
            data=buffer,
            file_name=f"soc_triage_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )

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
        # ── Confusion Matrix ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Model Performance on Uploaded Data")
        drop_cols = [c for c in ["attack_cat"] if c in df.columns]
        if "label" in df.columns:
            from realtime_input import FEATURES
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

            X_eval = df.drop(columns=[c for c in ["label","attack_cat"] if c in df.columns])
            X_eval = X_eval[[f for f in FEATURES if f in X_eval.columns]]
            le_eval = joblib.load("models/label_encoder.pkl")
            cat_cols_eval = X_eval.select_dtypes(include=["category", "object"]).columns.tolist()
            for col in cat_cols_eval:
                X_eval[col] = X_eval[col].astype(str).map(
                    lambda x: le_eval.transform([x])[0] if x in le_eval.classes_ else -1
                )
            X_eval = X_eval.astype(float)
            scaler_eval = joblib.load("models/scaler.pkl")
            X_eval = pd.DataFrame(scaler_eval.transform(X_eval), columns=X_eval.columns)

            y_true = df["label"].values
            y_prob_eval = model.predict_proba(X_eval)[:,1]
            y_pred_eval = (y_prob_eval >= threshold).astype(int)

            from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
            acc  = accuracy_score(y_true, y_pred_eval)
            prec = precision_score(y_true, y_pred_eval)
            rec  = recall_score(y_true, y_pred_eval)
            f1   = f1_score(y_true, y_pred_eval)
            fpr  = ((y_pred_eval==1)&(y_true==0)).sum() / (y_true==0).sum()

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy",  f"{acc*100:.1f}%")
            m2.metric("Precision", f"{prec*100:.1f}%")
            m3.metric("Recall",    f"{rec*100:.1f}%")
            m4.metric("F1 Score",  f"{f1*100:.1f}%")
            m5.metric("FP Rate",   f"{fpr*100:.1f}%")

            cm = confusion_matrix(y_true, y_pred_eval)
            fig_cm, ax_cm = plt.subplots(figsize=(5,4))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal","Attack"])
            disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
            ax_cm.set_facecolor("#0e1117")
            fig_cm.patch.set_facecolor("#0e1117")
            plt.title("Confusion Matrix (Threshold 0.7)", color="white")
            plt.tick_params(colors="white")
            ax_cm.xaxis.label.set_color("white")
            ax_cm.yaxis.label.set_color("white")
            st.pyplot(fig_cm)
            plt.close()

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
        # ─────────────────────────────────────────────────────────────────────────────
# ATTACK TYPE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎯 Attack Type Detector":
    st.title("🎯 Attack Type Detector")
    st.caption("Multiclass XGBoost model — identifies specific attack category from network flow.")

    @st.cache_resource
    def load_multiclass():
        mc_model  = joblib.load("models/xgboost_multiclass.pkl")
        mc_scaler = joblib.load("models/scaler_multiclass.pkl")
        mc_le     = joblib.load("models/label_encoder_target.pkl")
        return mc_model, mc_scaler, mc_le

    mc_model, mc_scaler, mc_le = load_multiclass()

    # ── Attack type colours ───────────────────────────────────────────────────
    ATTACK_COLOURS = {
        "Normal":          "#34C759",
        "DoS":             "#FF2D55",
        "Reconnaissance":  "#FF9500",
        "Exploits":        "#FF3B30",
        "Fuzzers":         "#AF52DE",
        "Backdoor":        "#FF2D55",
        "Analysis":        "#5AC8FA",
        "Shellcode":       "#FF6B6B",
        "Worms":           "#FF0000",
        "Generic":         "#FFD60A",
    }

    tab1, tab2 = st.tabs(["📝 Single Flow", "📂 Batch Detection"])

    # ── Single flow ───────────────────────────────────────────────────────────
    with tab1:
        X_row = render_manual_input()

        if X_row is not None:
            X_scaled = mc_scaler.transform(X_row)
            pred_encoded  = mc_model.predict(X_scaled)[0]
            pred_proba    = mc_model.predict_proba(X_scaled)[0]
            attack_type   = mc_le.inverse_transform([pred_encoded])[0]
            confidence    = pred_proba[pred_encoded] * 100
            colour        = ATTACK_COLOURS.get(attack_type, "#6C7BFF")

            st.markdown(
                f"""
                <div style="
                    border-left: 6px solid {colour};
                    background: rgba(255,255,255,0.04);
                    border-radius: 8px;
                    padding: 20px;
                    margin: 16px 0;
                ">
                    <h2 style="color:{colour}; margin:0">
                        Detected: {attack_type}
                    </h2>
                    <p style="color:#aaa; margin:4px 0 0 0">
                        Confidence: {confidence:.1f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Probability bar chart
            import plotly.graph_objects as go
            classes = mc_le.classes_
            fig = go.Figure(go.Bar(
                x=pred_proba * 100,
                y=classes,
                orientation="h",
                marker_color=[ATTACK_COLOURS.get(c, "#6C7BFF") for c in classes],
            ))
            fig.update_layout(
                title="Probability per Attack Type",
                height=350,
                margin=dict(l=0, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                xaxis=dict(title="Probability (%)", showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Batch detection ───────────────────────────────────────────────────────
    with tab2:
        st.subheader("Upload dataset for batch attack type detection")
        batch_file = st.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"], key="batch_mc")

        if batch_file is not None:
            import plotly.express as px

            if batch_file.name.endswith(".csv"):
                df_batch = pd.read_csv(batch_file)
            else:
                df_batch = pd.read_parquet(batch_file)

            from realtime_input import FEATURES
            drop_cols = [c for c in ["label", "attack_cat"] if c in df_batch.columns]
            X_batch   = df_batch.drop(columns=drop_cols)
            X_batch   = X_batch[[f for f in FEATURES if f in X_batch.columns]]

            for col in X_batch.select_dtypes(include=["category", "object"]).columns:
                X_batch[col] = X_batch[col].astype(str).astype("category").cat.codes
            X_batch = X_batch.astype(float)

            X_batch_scaled   = mc_scaler.transform(X_batch)
            preds_encoded    = mc_model.predict(X_batch_scaled)
            preds_labels     = mc_le.inverse_transform(preds_encoded)

            df_batch["predicted_attack_type"] = preds_labels

            # Summary chart
            type_counts = pd.Series(preds_labels).value_counts().reset_index()
            type_counts.columns = ["Attack Type", "Count"]

            st.success(f"✅ Analysed {len(df_batch):,} flows")

            col1, col2 = st.columns(2)

            with col1:
                fig2 = px.pie(
                    type_counts, names="Attack Type", values="Count",
                    color="Attack Type",
                    color_discrete_map=ATTACK_COLOURS,
                    hole=0.4,
                )
                fig2.update_layout(
                    height=350,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig2, use_container_width=True)

            with col2:
                fig3 = px.bar(
                    type_counts, x="Attack Type", y="Count",
                    color="Attack Type",
                    color_discrete_map=ATTACK_COLOURS,
                )
                fig3.update_layout(
                    height=350,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    showlegend=False,
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                )
                st.plotly_chart(fig3, use_container_width=True)

            with st.expander("📋 Full Predictions Table"):
                st.dataframe(
                    df_batch[["predicted_attack_type"] + [f for f in FEATURES if f in df_batch.columns]].head(500),
                    use_container_width=True
                )