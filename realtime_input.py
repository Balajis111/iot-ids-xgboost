"""
realtime_input.py
-----------------
Real-time network flow input for the IoT IDS.
Supports two modes:
  1. Manual entry  – SOC analyst types in feature values via Streamlit form
  2. Simulated stream – auto-generates synthetic IoT flows for demo/testing

Returns a pandas DataFrame row ready for model inference.
"""

import numpy as np
import pandas as pd
import streamlit as st
import time
import random

# ── Feature schema ────────────────────────────────────────────────────────────
# Align with UNSW-NB15 / CICIoT2023 feature subset used during training.
# Replace with your actual trained feature list if different.
FEATURES = [
    "duration",
    "proto",           # encoded: 0=tcp, 1=udp, 2=icmp, 3=other
    "service",         # encoded: 0=http,1=dns,2=ftp,3=ssh,4=smtp,5=other
    "state",           # encoded: 0=FIN,1=INT,2=CON,3=REQ,4=RST,5=other
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
]

PROTO_MAP  = {"tcp": 0, "udp": 1, "icmp": 2, "other": 3}
SERVICE_MAP = {"http": 0, "dns": 1, "ftp": 2, "ssh": 3, "smtp": 4, "other": 5}
STATE_MAP   = {"FIN": 0, "INT": 1, "CON": 2, "REQ": 3, "RST": 4, "other": 5}


# ── Attack-type simulation profiles ──────────────────────────────────────────
_NORMAL_PROFILE = dict(
    duration=(0.1, 2.0), spkts=(2, 20), dpkts=(2, 20),
    sbytes=(100, 5000), dbytes=(100, 5000),
    rate=(10, 500), sttl=(60, 128), dttl=(60, 128),
    sload=(0, 5000), dload=(0, 5000),
    sloss=(0, 1), dloss=(0, 1), sinpkt=(0.01, 0.5), dinpkt=(0.01, 0.5),
    sjit=(0, 10), djit=(0, 10), swin=(64, 65535), dwin=(64, 65535),
    smean=(50, 1500), dmean=(50, 1500),
)

_DOS_PROFILE = dict(
    duration=(0.001, 0.1), spkts=(500, 5000), dpkts=(0, 5),
    sbytes=(50000, 500000), dbytes=(0, 500),
    rate=(5000, 50000), sttl=(64, 64), dttl=(0, 10),
    sload=(100000, 1000000), dload=(0, 100),
    sloss=(100, 1000), dloss=(0, 5),
    sinpkt=(0.0001, 0.001), dinpkt=(0.5, 5),
    sjit=(0, 2), djit=(0, 2), swin=(0, 10), dwin=(0, 10),
    smean=(100, 200), dmean=(0, 50),
)

_SCAN_PROFILE = dict(
    duration=(0.001, 0.05), spkts=(1, 3), dpkts=(0, 1),
    sbytes=(40, 200), dbytes=(0, 100),
    rate=(1, 50), sttl=(64, 128), dttl=(64, 128),
    sload=(0, 500), dload=(0, 100),
    sloss=(0, 2), dloss=(0, 2), sinpkt=(0.001, 0.1), dinpkt=(0.001, 0.1),
    sjit=(0, 5), djit=(0, 5), swin=(0, 0), dwin=(0, 0),
    smean=(40, 100), dmean=(0, 50),
)

_EXPLOIT_PROFILE = dict(
    duration=(0.5, 10), spkts=(5, 50), dpkts=(5, 50),
    sbytes=(500, 50000), dbytes=(500, 50000),
    rate=(10, 200), sttl=(64, 128), dttl=(64, 128),
    sload=(500, 20000), dload=(500, 20000),
    sloss=(0, 5), dloss=(0, 5), sinpkt=(0.01, 0.3), dinpkt=(0.01, 0.3),
    sjit=(0, 20), djit=(0, 20), swin=(512, 65535), dwin=(512, 65535),
    smean=(100, 1000), dmean=(100, 1000),
)

PROFILES = {
    "Normal":  (_NORMAL_PROFILE,  0),
    "DoS":     (_DOS_PROFILE,     1),
    "Scan":    (_SCAN_PROFILE,    1),
    "Exploit": (_EXPLOIT_PROFILE, 1),
}


def _sample_profile(profile: dict) -> dict:
    return {k: random.uniform(v[0], v[1]) for k, v in profile.items()}


def generate_synthetic_flow(attack_type: str = "Normal") -> pd.DataFrame:
    """Generate a single synthetic network flow matching a given attack type."""
    profile_data, _ = PROFILES.get(attack_type, PROFILES["Normal"])
    base = _sample_profile(profile_data)

    row = {
        "duration":           base["duration"],
        "proto":              random.choice([0, 1, 2]),
        "service":            random.randint(0, 5),
        "state":              random.randint(0, 5),
        "spkts":              int(base["spkts"]),
        "dpkts":              int(base["dpkts"]),
        "sbytes":             int(base["sbytes"]),
        "dbytes":             int(base["dbytes"]),
        "rate":               base["rate"],
        "sttl":               int(base["sttl"]),
        "dttl":               int(base["dttl"]),
        "sload":              base["sload"],
        "dload":              base["dload"],
        "sloss":              int(base["sloss"]),
        "dloss":              int(base["dloss"]),
        "sinpkt":             base["sinpkt"],
        "dinpkt":             base["dinpkt"],
        "sjit":               base["sjit"],
        "djit":               base["djit"],
        "swin":               int(base["swin"]),
        "stcpb":              random.randint(0, 2**32),
        "dtcpb":              random.randint(0, 2**32),
        "dwin":               int(base["dwin"]),
        "tcprtt":             random.uniform(0, 0.5),
        "synack":             random.uniform(0, 0.3),
        "ackdat":             random.uniform(0, 0.3),
        "smean":              int(base["smean"]),
        "dmean":              int(base["dmean"]),
        "trans_depth":        random.randint(0, 5),
        "response_body_len":  random.randint(0, 10000),
        "ct_srv_src":         random.randint(1, 50),
        "ct_state_ttl":       random.randint(0, 6),
        "ct_dst_ltm":         random.randint(1, 50),
        "ct_src_dport_ltm":   random.randint(1, 50),
        "ct_dst_sport_ltm":   random.randint(1, 50),
        "ct_dst_src_ltm":     random.randint(1, 50),
        "is_ftp_login":       random.randint(0, 1),
        "ct_ftp_cmd":         random.randint(0, 5),
        "ct_flw_http_mthd":   random.randint(0, 10),
        "ct_src_ltm":         random.randint(1, 50),
        "ct_srv_dst":         random.randint(1, 50),
        "is_sm_ips_ports":    random.randint(0, 1),
    }
    return pd.DataFrame([row])


# ── Manual entry UI ───────────────────────────────────────────────────────────
def render_manual_input() -> pd.DataFrame | None:
    """
    Renders a Streamlit form for manual flow entry.
    Returns a DataFrame row on submit, else None.
    """
    st.subheader("📝 Manual Flow Entry")
    with st.form("manual_flow_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            duration   = st.number_input("Duration (s)",    min_value=0.0, value=1.0,  step=0.01)
            proto      = st.selectbox("Protocol",           list(PROTO_MAP.keys()))
            service    = st.selectbox("Service",            list(SERVICE_MAP.keys()))
            state      = st.selectbox("State",              list(STATE_MAP.keys()))
            spkts      = st.number_input("Src Packets",     min_value=0, value=10)
            dpkts      = st.number_input("Dst Packets",     min_value=0, value=10)
            sbytes     = st.number_input("Src Bytes",       min_value=0, value=500)
            dbytes     = st.number_input("Dst Bytes",       min_value=0, value=500)
            rate       = st.number_input("Rate (pps)",      min_value=0.0, value=50.0)
            sttl       = st.number_input("Src TTL",         min_value=0, max_value=255, value=64)
            dttl       = st.number_input("Dst TTL",         min_value=0, max_value=255, value=64)
            sload      = st.number_input("Src Load",        min_value=0.0, value=1000.0)
            dload      = st.number_input("Dst Load",        min_value=0.0, value=1000.0)

        with col2:
            sloss      = st.number_input("Src Loss",        min_value=0, value=0)
            dloss      = st.number_input("Dst Loss",        min_value=0, value=0)
            sinpkt     = st.number_input("Src Inter-pkt (s)", min_value=0.0, value=0.05, step=0.001, format="%.4f")
            dinpkt     = st.number_input("Dst Inter-pkt (s)", min_value=0.0, value=0.05, step=0.001, format="%.4f")
            sjit       = st.number_input("Src Jitter",      min_value=0.0, value=1.0)
            djit       = st.number_input("Dst Jitter",      min_value=0.0, value=1.0)
            swin       = st.number_input("Src Window",      min_value=0, value=8192)
            dwin       = st.number_input("Dst Window",      min_value=0, value=8192)
            tcprtt     = st.number_input("TCP RTT (s)",     min_value=0.0, value=0.05, step=0.001, format="%.4f")
            synack     = st.number_input("SYN-ACK time",    min_value=0.0, value=0.02, step=0.001, format="%.4f")
            ackdat     = st.number_input("ACK-data time",   min_value=0.0, value=0.01, step=0.001, format="%.4f")
            smean      = st.number_input("Src Mean pkt size", min_value=0, value=500)
            dmean      = st.number_input("Dst Mean pkt size", min_value=0, value=500)

        with col3:
            trans_depth        = st.number_input("Trans Depth",         min_value=0, value=1)
            response_body_len  = st.number_input("Response Body Len",   min_value=0, value=0)
            ct_srv_src         = st.number_input("ct_srv_src",          min_value=0, value=5)
            ct_state_ttl       = st.number_input("ct_state_ttl",        min_value=0, value=2)
            ct_dst_ltm         = st.number_input("ct_dst_ltm",          min_value=0, value=5)
            ct_src_dport_ltm   = st.number_input("ct_src_dport_ltm",    min_value=0, value=5)
            ct_dst_sport_ltm   = st.number_input("ct_dst_sport_ltm",    min_value=0, value=5)
            ct_dst_src_ltm     = st.number_input("ct_dst_src_ltm",      min_value=0, value=5)
            is_ftp_login       = st.selectbox("is_ftp_login",           [0, 1])
            ct_ftp_cmd         = st.number_input("ct_ftp_cmd",          min_value=0, value=0)
            ct_flw_http_mthd   = st.number_input("ct_flw_http_mthd",    min_value=0, value=0)
            ct_src_ltm         = st.number_input("ct_src_ltm",          min_value=0, value=5)
            ct_srv_dst         = st.number_input("ct_srv_dst",          min_value=0, value=5)
            is_sm_ips_ports    = st.selectbox("is_sm_ips_ports",        [0, 1])

        submitted = st.form_submit_button("🔍 Analyse Flow", type="primary")

    if not submitted:
        return None

    row = {
        "duration": duration, "proto": PROTO_MAP[proto],
        "service": SERVICE_MAP[service], "state": STATE_MAP[state],
        "spkts": spkts, "dpkts": dpkts, "sbytes": sbytes, "dbytes": dbytes,
        "rate": rate, "sttl": sttl, "dttl": dttl,
        "sload": sload, "dload": dload,
        "sloss": sloss, "dloss": dloss,
        "sinpkt": sinpkt, "dinpkt": dinpkt,
        "sjit": sjit, "djit": djit,
        "swin": swin, "stcpb": 0, "dtcpb": 0, "dwin": dwin,
        "tcprtt": tcprtt, "synack": synack, "ackdat": ackdat,
        "smean": smean, "dmean": dmean,
        "trans_depth": trans_depth, "response_body_len": response_body_len,
        "ct_srv_src": ct_srv_src, "ct_state_ttl": ct_state_ttl,
        "ct_dst_ltm": ct_dst_ltm, "ct_src_dport_ltm": ct_src_dport_ltm,
        "ct_dst_sport_ltm": ct_dst_sport_ltm, "ct_dst_src_ltm": ct_dst_src_ltm,
        "is_ftp_login": is_ftp_login, "ct_ftp_cmd": ct_ftp_cmd,
        "ct_flw_http_mthd": ct_flw_http_mthd, "ct_src_ltm": ct_src_ltm,
        "ct_srv_dst": ct_srv_dst, "is_sm_ips_ports": is_sm_ips_ports,
    }
    return pd.DataFrame([row])
