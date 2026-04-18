"""
realtime_input.py
-----------------
Real-time network flow input for the IoT IDS.
Features aligned to UNSW-NB15 training set (34 features).
"""

import numpy as np
import pandas as pd
import streamlit as st
import random

# ── Exact 34 features the model was trained on ────────────────────────────────
FEATURES = [
    "dur", "proto", "service", "state",
    "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sload", "dload",
    "sloss", "dloss",
    "sinpkt", "dinpkt",
    "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat",
    "smean", "dmean",
    "trans_depth", "response_body_len",
    "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "is_sm_ips_ports",
]

PROTO_MAP   = {"tcp": 0, "udp": 1, "icmp": 2, "other": 3}
SERVICE_MAP = {"http": 0, "dns": 1, "ftp": 2, "ssh": 3, "smtp": 4, "other": 5}
STATE_MAP   = {"FIN": 0, "INT": 1, "CON": 2, "REQ": 3, "RST": 4, "other": 5}


# ── Attack simulation profiles ────────────────────────────────────────────────
PROFILES = {
    "Normal": {
        "dur": (0.1, 2.0), "spkts": (2, 20), "dpkts": (2, 20),
        "sbytes": (100, 5000), "dbytes": (100, 5000),
        "rate": (10, 500), "sload": (0, 5000), "dload": (0, 5000),
        "sloss": (0, 1), "dloss": (0, 1),
        "sinpkt": (0.01, 0.5), "dinpkt": (0.01, 0.5),
        "sjit": (0, 10), "djit": (0, 10),
        "swin": (64, 65535), "dwin": (64, 65535),
        "smean": (50, 1500), "dmean": (50, 1500),
    },
    "DoS": {
        "dur": (0.001, 0.1), "spkts": (500, 5000), "dpkts": (0, 5),
        "sbytes": (50000, 500000), "dbytes": (0, 500),
        "rate": (5000, 50000), "sload": (100000, 1000000), "dload": (0, 100),
        "sloss": (100, 1000), "dloss": (0, 5),
        "sinpkt": (0.0001, 0.001), "dinpkt": (0.5, 5),
        "sjit": (0, 2), "djit": (0, 2),
        "swin": (0, 10), "dwin": (0, 10),
        "smean": (100, 200), "dmean": (0, 50),
    },
    "Scan": {
        "dur": (0.001, 0.05), "spkts": (1, 3), "dpkts": (0, 1),
        "sbytes": (40, 200), "dbytes": (0, 100),
        "rate": (1, 50), "sload": (0, 500), "dload": (0, 100),
        "sloss": (0, 2), "dloss": (0, 2),
        "sinpkt": (0.001, 0.1), "dinpkt": (0.001, 0.1),
        "sjit": (0, 5), "djit": (0, 5),
        "swin": (0, 0), "dwin": (0, 0),
        "smean": (40, 100), "dmean": (0, 50),
    },
    "Exploit": {
        "dur": (0.5, 10), "spkts": (5, 50), "dpkts": (5, 50),
        "sbytes": (500, 50000), "dbytes": (500, 50000),
        "rate": (10, 200), "sload": (500, 20000), "dload": (500, 20000),
        "sloss": (0, 5), "dloss": (0, 5),
        "sinpkt": (0.01, 0.3), "dinpkt": (0.01, 0.3),
        "sjit": (0, 20), "djit": (0, 20),
        "swin": (512, 65535), "dwin": (512, 65535),
        "smean": (100, 1000), "dmean": (100, 1000),
    },
}


def generate_synthetic_flow(attack_type: str = "Normal") -> pd.DataFrame:
    p = PROFILES.get(attack_type, PROFILES["Normal"])
    row = {
        "dur":                random.uniform(*p["dur"]),
        "proto":              random.choice([0, 1, 2]),
        "service":            random.randint(0, 5),
        "state":              random.randint(0, 5),
        "spkts":              int(random.uniform(*p["spkts"])),
        "dpkts":              int(random.uniform(*p["dpkts"])),
        "sbytes":             int(random.uniform(*p["sbytes"])),
        "dbytes":             int(random.uniform(*p["dbytes"])),
        "rate":               random.uniform(*p["rate"]),
        "sload":              random.uniform(*p["sload"]),
        "dload":              random.uniform(*p["dload"]),
        "sloss":              int(random.uniform(*p["sloss"])),
        "dloss":              int(random.uniform(*p["dloss"])),
        "sinpkt":             random.uniform(*p["sinpkt"]),
        "dinpkt":             random.uniform(*p["dinpkt"]),
        "sjit":               random.uniform(*p["sjit"]),
        "djit":               random.uniform(*p["djit"]),
        "swin":               int(random.uniform(*p["swin"])),
        "stcpb":              random.randint(0, 2**32),
        "dtcpb":              random.randint(0, 2**32),
        "dwin":               int(random.uniform(*p["dwin"])),
        "tcprtt":             random.uniform(0, 0.5),
        "synack":             random.uniform(0, 0.3),
        "ackdat":             random.uniform(0, 0.3),
        "smean":              int(random.uniform(*p["smean"])),
        "dmean":              int(random.uniform(*p["dmean"])),
        "trans_depth":        random.randint(0, 5),
        "response_body_len":  random.randint(0, 10000),
        "ct_src_dport_ltm":   random.randint(1, 50),
        "ct_dst_sport_ltm":   random.randint(1, 50),
        "is_ftp_login":       random.randint(0, 1),
        "ct_ftp_cmd":         random.randint(0, 5),
        "ct_flw_http_mthd":   random.randint(0, 10),
        "is_sm_ips_ports":    random.randint(0, 1),
    }
    return pd.DataFrame([row])[FEATURES]


def render_manual_input():
    st.subheader("📝 Manual Flow Entry")
    with st.form("manual_flow_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            dur        = st.number_input("dur (s)",           min_value=0.0, value=1.0,   step=0.01)
            proto      = st.selectbox("Protocol",             list(PROTO_MAP.keys()))
            service    = st.selectbox("Service",              list(SERVICE_MAP.keys()))
            state      = st.selectbox("State",                list(STATE_MAP.keys()))
            spkts      = st.number_input("spkts",             min_value=0, value=10)
            dpkts      = st.number_input("dpkts",             min_value=0, value=10)
            sbytes     = st.number_input("sbytes",            min_value=0, value=500)
            dbytes     = st.number_input("dbytes",            min_value=0, value=500)
            rate       = st.number_input("rate",              min_value=0.0, value=50.0)
            sload      = st.number_input("sload",             min_value=0.0, value=1000.0)
            dload      = st.number_input("dload",             min_value=0.0, value=1000.0)

        with col2:
            sloss      = st.number_input("sloss",             min_value=0, value=0)
            dloss      = st.number_input("dloss",             min_value=0, value=0)
            sinpkt     = st.number_input("sinpkt",            min_value=0.0, value=0.05,  step=0.001, format="%.4f")
            dinpkt     = st.number_input("dinpkt",            min_value=0.0, value=0.05,  step=0.001, format="%.4f")
            sjit       = st.number_input("sjit",              min_value=0.0, value=1.0)
            djit       = st.number_input("djit",              min_value=0.0, value=1.0)
            swin       = st.number_input("swin",              min_value=0, value=8192)
            dwin       = st.number_input("dwin",              min_value=0, value=8192)
            tcprtt     = st.number_input("tcprtt",            min_value=0.0, value=0.05,  step=0.001, format="%.4f")
            synack     = st.number_input("synack",            min_value=0.0, value=0.02,  step=0.001, format="%.4f")
            ackdat     = st.number_input("ackdat",            min_value=0.0, value=0.01,  step=0.001, format="%.4f")

        with col3:
            smean             = st.number_input("smean",             min_value=0, value=500)
            dmean             = st.number_input("dmean",             min_value=0, value=500)
            trans_depth       = st.number_input("trans_depth",       min_value=0, value=1)
            response_body_len = st.number_input("response_body_len", min_value=0, value=0)
            ct_src_dport_ltm  = st.number_input("ct_src_dport_ltm",  min_value=0, value=5)
            ct_dst_sport_ltm  = st.number_input("ct_dst_sport_ltm",  min_value=0, value=5)
            is_ftp_login      = st.selectbox("is_ftp_login",         [0, 1])
            ct_ftp_cmd        = st.number_input("ct_ftp_cmd",        min_value=0, value=0)
            ct_flw_http_mthd  = st.number_input("ct_flw_http_mthd",  min_value=0, value=0)
            is_sm_ips_ports   = st.selectbox("is_sm_ips_ports",      [0, 1])

        submitted = st.form_submit_button("🔍 Analyse Flow", type="primary")

    if not submitted:
        return None

    row = {
        "dur":                dur,
        "proto":              PROTO_MAP[proto],
        "service":            SERVICE_MAP[service],
        "state":              STATE_MAP[state],
        "spkts":              spkts,
        "dpkts":              dpkts,
        "sbytes":             sbytes,
        "dbytes":             dbytes,
        "rate":               rate,
        "sload":              sload,
        "dload":              dload,
        "sloss":              sloss,
        "dloss":              dloss,
        "sinpkt":             sinpkt,
        "dinpkt":             dinpkt,
        "sjit":               sjit,
        "djit":               djit,
        "swin":               swin,
        "stcpb":              0,
        "dtcpb":              0,
        "dwin":               dwin,
        "tcprtt":             tcprtt,
        "synack":             synack,
        "ackdat":             ackdat,
        "smean":              smean,
        "dmean":              dmean,
        "trans_depth":        trans_depth,
        "response_body_len":  response_body_len,
        "ct_src_dport_ltm":   ct_src_dport_ltm,
        "ct_dst_sport_ltm":   ct_dst_sport_ltm,
        "is_ftp_login":       is_ftp_login,
        "ct_ftp_cmd":         ct_ftp_cmd,
        "ct_flw_http_mthd":   ct_flw_http_mthd,
        "is_sm_ips_ports":    is_sm_ips_ports,
    }
    return pd.DataFrame([row])[FEATURES]