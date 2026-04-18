"""
alert_priority.py
-----------------
SHAP-Weighted Alert Priority Scoring for SOC Triage.

Core idea (novel angle for the paper):
  Instead of ranking alerts purely by model confidence (predict_proba),
  we compute a composite PRIORITY SCORE that weighs:
    1. Model confidence   – how certain is XGBoost this is an attack?
    2. SHAP magnitude     – how strongly do the top features push toward malicious?
    3. Feature criticality weights – not all features matter equally in SOC context
       (e.g., high packet rate matters more than TCP window size for triage)

  PRIORITY SCORE = α * confidence + β * shap_magnitude + γ * critical_feature_boost

  This produces a [0-100] triage score that tells analysts:
    ≥ 80  → CRITICAL  (auto-escalate)
    60-79 → HIGH
    40-59 → MEDIUM
    < 40  → LOW / INFO

Usage:
    from alert_priority import compute_priority, render_triage_card
    score, breakdown = compute_priority(model, explainer, X_row)
    render_triage_card(score, breakdown, shap_values_row)
"""

import numpy as np
import pandas as pd
import shap
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
from typing import Tuple

# ── Triage severity levels ────────────────────────────────────────────────────
SEVERITY_LEVELS = {
    "CRITICAL": {"min": 80, "color": "#FF2D55", "emoji": "🔴", "action": "Auto-escalate to L2/L3"},
    "HIGH":     {"min": 60, "color": "#FF9500", "emoji": "🟠", "action": "Immediate investigation"},
    "MEDIUM":   {"min": 40, "color": "#FFCC00", "emoji": "🟡", "action": "Queue for analysis"},
    "LOW":      {"min":  0, "color": "#34C759", "emoji": "🟢", "action": "Log and monitor"},
}

# ── SOC-context feature criticality weights ───────────────────────────────────
# These reflect how operationally important each feature signal is for a SOC
# analyst performing triage — e.g., burst rate matters more than jitter.
# Tune these based on your deployment context.
FEATURE_CRITICALITY = {
    # Volume/rate features — high operational importance
    "rate":               1.5,
    "spkts":              1.4,
    "dpkts":              1.4,
    "sbytes":             1.3,
    "dbytes":             1.3,
    "sload":              1.4,
    "dload":              1.4,
    "sloss":              1.3,
    "dloss":              1.3,

    # Timing features — moderate importance
    "sinpkt":             1.2,
    "dinpkt":             1.2,
    "duration":           1.1,
    "tcprtt":             1.1,
    "synack":             1.2,
    "ackdat":             1.1,

    # Connection context — moderate importance
    "ct_srv_src":         1.2,
    "ct_state_ttl":       1.2,
    "ct_dst_src_ltm":     1.1,
    "ct_src_ltm":         1.1,
    "ct_srv_dst":         1.2,
    "is_sm_ips_ports":    1.3,

    # Protocol/state — context-dependent
    "proto":              1.0,
    "service":            1.0,
    "state":              1.1,
    "sttl":               1.0,
    "dttl":               1.0,

    # Window/TCP — lower triage value
    "swin":               0.8,
    "dwin":               0.8,
    "stcpb":              0.7,
    "dtcpb":              0.7,
    "sjit":               0.9,
    "djit":               0.9,
    "smean":              0.9,
    "dmean":              0.9,
    "trans_depth":        0.8,
    "response_body_len":  0.8,
    "is_ftp_login":       1.1,
    "ct_ftp_cmd":         1.0,
    "ct_flw_http_mthd":   0.9,
    "ct_dst_ltm":         0.9,
    "ct_src_dport_ltm":   0.9,
    "ct_dst_sport_ltm":   0.9,
}

# Scoring weights (α + β + γ = 1.0)
ALPHA = 0.40   # model confidence weight
BETA  = 0.40   # SHAP magnitude weight
GAMMA = 0.20   # critical feature boost weight


@dataclass
class PriorityBreakdown:
    """Full breakdown of how the priority score was computed."""
    confidence_score:     float    # raw model probability [0,1]
    shap_magnitude:       float    # normalised SHAP contribution [0,1]
    critical_boost:       float    # normalised criticality-weighted score [0,1]
    priority_score:       float    # final [0-100] score
    severity:             str      # CRITICAL / HIGH / MEDIUM / LOW
    top_features:         list     # [(feature_name, shap_val, criticality_wt), ...]
    confidence_component: float    # α * confidence_score
    shap_component:       float    # β * shap_magnitude
    critical_component:   float    # γ * critical_boost


def _get_severity(score: float) -> str:
    for level, cfg in SEVERITY_LEVELS.items():
        if score >= cfg["min"]:
            return level
    return "LOW"


def compute_priority(
    model,
    explainer: shap.TreeExplainer,
    X_row: pd.DataFrame,
    top_n: int = 10,
) -> Tuple[float, PriorityBreakdown]:
    """
    Compute the SHAP-weighted alert priority score for a single flow.

    Parameters
    ----------
    model       : trained XGBoost classifier
    explainer   : shap.TreeExplainer fitted on model
    X_row       : pd.DataFrame with one row (must match training feature order)
    top_n       : how many top contributing features to include in breakdown

    Returns
    -------
    (priority_score, PriorityBreakdown)
    """
    feature_names = list(X_row.columns)

    # ── 1. Model confidence ───────────────────────────────────────────────────
    prob = model.predict_proba(X_row)[0]
    # Handle binary (prob[1]) or multiclass (max non-normal class)
    if len(prob) == 2:
        confidence = float(prob[1])
    else:
        confidence = float(max(prob[1:]))  # exclude class 0 (normal)

    # ── 2. SHAP values ────────────────────────────────────────────────────────
    shap_vals = explainer.shap_values(X_row)

    # For multiclass, shap_vals is a list of arrays — use the attack class
    if isinstance(shap_vals, list):
        # Sum absolute SHAP across all attack classes
        sv = np.sum([np.abs(shap_vals[c][0]) for c in range(1, len(shap_vals))], axis=0)
    else:
        sv = np.abs(shap_vals[0])                  # binary

    # ── 3. Criticality-weighted SHAP magnitude ────────────────────────────────
    crit_weights = np.array([
        FEATURE_CRITICALITY.get(f, 1.0) for f in feature_names
    ])
    weighted_sv = sv * crit_weights

    # Normalise to [0, 1] using a sigmoid-like clamp
    raw_shap_mag  = float(np.sum(np.abs(sv)))
    raw_crit_mag  = float(np.sum(weighted_sv))

    # Calibrate: typical malicious flow has ~total |SHAP| around 5-15 depending on model
    # Normalise against a reference ceiling (tune if needed)
    SHAP_CEILING = 15.0
    CRIT_CEILING = 20.0

    shap_magnitude = min(raw_shap_mag  / SHAP_CEILING, 1.0)
    critical_boost = min(raw_crit_mag  / CRIT_CEILING, 1.0)

    # ── 4. Composite priority score ───────────────────────────────────────────
    conf_component  = ALPHA * confidence
    shap_component  = BETA  * shap_magnitude
    crit_component  = GAMMA * critical_boost

    priority_score = (conf_component + shap_component + crit_component) * 100.0
    priority_score = min(max(priority_score, 0.0), 100.0)

    # ── 5. Top contributing features ─────────────────────────────────────────
    feature_contributions = sorted(
        zip(feature_names, sv, crit_weights),
        key=lambda x: abs(x[1]) * x[2],
        reverse=True,
    )
    top_features = feature_contributions[:top_n]

    breakdown = PriorityBreakdown(
        confidence_score     = confidence,
        shap_magnitude       = shap_magnitude,
        critical_boost       = critical_boost,
        priority_score       = priority_score,
        severity             = _get_severity(priority_score),
        top_features         = top_features,
        confidence_component = conf_component  * 100,
        shap_component       = shap_component  * 100,
        critical_component   = crit_component  * 100,
    )
    return priority_score, breakdown


def render_triage_card(
    breakdown: PriorityBreakdown,
    flow_data: pd.DataFrame,
    show_waterfall: bool = True,
    show_gauge: bool = True,
) -> None:
    """
    Render a full SOC triage card in Streamlit.
    Call this after compute_priority().
    """
    sev  = breakdown.severity
    cfg  = SEVERITY_LEVELS[sev]
    score = breakdown.priority_score

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            border-left: 6px solid {cfg['color']};
            background: rgba(255,255,255,0.04);
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 16px;
        ">
            <h2 style="margin:0; color:{cfg['color']}">
                {cfg['emoji']} {sev} — Priority Score: {score:.1f} / 100
            </h2>
            <p style="margin:4px 0 0 0; color:#aaa;">
                <b>Recommended Action:</b> {cfg['action']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    # ── Gauge chart ───────────────────────────────────────────────────────────
    if show_gauge:
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": "Alert Priority Score", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar":  {"color": cfg["color"]},
                    "steps": [
                        {"range": [0,  40], "color": "#1e3a1e"},
                        {"range": [40, 60], "color": "#3a3a1e"},
                        {"range": [60, 80], "color": "#3a2a1e"},
                        {"range": [80, 100],"color": "#3a1e1e"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.75,
                        "value": score,
                    },
                },
                number={"suffix": "/100", "font": {"size": 28}},
            ))
            fig.update_layout(
                height=220,
                margin=dict(l=20, r=20, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Score breakdown bar ───────────────────────────────────────────────────
    with col2:
        st.markdown("**Score Breakdown**")
        components = {
            f"Confidence ({ALPHA*100:.0f}%)": breakdown.confidence_component,
            f"SHAP Magnitude ({BETA*100:.0f}%)": breakdown.shap_component,
            f"Critical Feature Boost ({GAMMA*100:.0f}%)": breakdown.critical_component,
        }
        fig2 = go.Figure(go.Bar(
            x=list(components.values()),
            y=list(components.keys()),
            orientation="h",
            marker_color=[cfg["color"], "#6C7BFF", "#34C759"],
            text=[f"{v:.1f}" for v in components.values()],
            textposition="outside",
        ))
        fig2.update_layout(
            height=200,
            margin=dict(l=0, r=40, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis=dict(range=[0, 50], showgrid=False),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            f"- **Model Confidence:** `{breakdown.confidence_score*100:.1f}%`  \n"
            f"- **SHAP Magnitude (norm):** `{breakdown.shap_magnitude:.3f}`  \n"
            f"- **Criticality Boost (norm):** `{breakdown.critical_boost:.3f}`"
        )

    # ── Top features waterfall ────────────────────────────────────────────────
    if show_waterfall and breakdown.top_features:
        st.markdown("---")
        st.markdown("#### 🔬 Top Contributing Features (SHAP × Criticality Weight)")

        names  = [f for f, _, _ in breakdown.top_features]
        sv     = [s for _, s, _ in breakdown.top_features]
        cw     = [c for _, _, c in breakdown.top_features]
        weighted = [abs(s) * c for s, c in zip(sv, cw)]

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            name="Raw |SHAP|",
            x=names, y=[abs(s) for s in sv],
            marker_color="#6C7BFF", opacity=0.7,
        ))
        fig3.add_trace(go.Bar(
            name="SHAP × Criticality",
            x=names, y=weighted,
            marker_color=cfg["color"], opacity=0.9,
        ))
        fig3.update_layout(
            barmode="group",
            height=320,
            margin=dict(l=10, r=10, t=10, b=80),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(tickangle=-35, showgrid=False),
            yaxis=dict(title="Score Contribution", showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Table
        tbl_data = pd.DataFrame({
            "Feature":              names,
            "SHAP Value":           [f"{s:+.4f}" for s in sv],
            "Criticality Weight":   [f"{c:.1f}" for c in cw],
            "Triage Impact":        [f"{abs(s)*c:.4f}" for s, c in zip(sv, cw)],
            "Direction":            ["↑ Attack" if s > 0 else "↓ Normal" for s in sv],
        })
        st.dataframe(tbl_data, use_container_width=True, hide_index=True)

    # ── Raw feature values ────────────────────────────────────────────────────
    with st.expander("📋 Raw Flow Features"):
        st.dataframe(flow_data.T.rename(columns={0: "Value"}), use_container_width=True)


# ── Alert queue manager ───────────────────────────────────────────────────────
class AlertQueue:
    """
    In-memory queue of analysed alerts. Used by the simulated stream mode
    to accumulate and display multiple flows with priority ordering.
    """

    def __init__(self, max_size: int = 100):
        self.max_size  = max_size
        self.alerts: list[dict] = []

    def add(
        self,
        flow_id: str,
        flow_data: pd.DataFrame,
        prediction: int,
        breakdown: PriorityBreakdown,
    ):
        self.alerts.append({
            "id":          flow_id,
            "timestamp":   pd.Timestamp.now(),
            "prediction":  "🔴 Attack" if prediction else "🟢 Normal",
            "severity":    breakdown.severity,
            "score":       round(breakdown.priority_score, 1),
            "confidence":  f"{breakdown.confidence_score*100:.1f}%",
            "top_feature": breakdown.top_features[0][0] if breakdown.top_features else "—",
        })
        if len(self.alerts) > self.max_size:
            self.alerts.pop(0)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.alerts:
            return pd.DataFrame()
        df = pd.DataFrame(self.alerts)
        # Sort by score descending (highest priority first)
        return df.sort_values("score", ascending=False).reset_index(drop=True)

    def render(self):
        df = self.to_dataframe()
        if df.empty:
            st.info("No alerts yet.")
            return

        st.markdown(f"**{len(df)} alerts in queue — sorted by priority**")

        # Colour-code severity column
        def colour_severity(val):
            colours = {
                "CRITICAL": "background-color: #FF2D5533",
                "HIGH":     "background-color: #FF950033",
                "MEDIUM":   "background-color: #FFCC0033",
                "LOW":      "background-color: #34C75933",
            }
            return colours.get(val, "")

        styled = df.style.map(colour_severity, subset=["severity"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
