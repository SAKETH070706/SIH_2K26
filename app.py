import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import plotly.graph_objects as go
import os

# =========================================================
# 1. SYSTEM CONFIGURATION & MITRE STAGE TAXONOMY
# =========================================================
st.set_page_config(
    page_title="AI Network World Model | Cyber Defense",
    page_icon="🛡️",
    layout="wide"
)

DYNAMIC_FEATURES = [
    'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Mean',
    'Bwd Pkt Len Mean', 'Flow Byts/s', 'Flow Pkts/s',
    'Flow IAT Mean', 'Flow IAT Std', 'SYN Flag Cnt',
    'ACK Flag Cnt', 'RST Flag Cnt', 'Init Fwd Win Byts'
]

# NOTE: The model outputs 4 discrete stages. MITRE ATT&CK features granular tactics.
# Each label below denotes the PRIMARY tactic that class was trained to represent.
STAGE_CONFIG = {
    0: {
        "name": "Nominal Baseline",
        "label": "Phase 0: Baseline Normal",
        "mitre_id": "N/A (Benign)",
        "color": "#22c55e",
        "severity": "Low"
    },
    1: {
        "name": "Initial Access (Brute Force / Probe)",
        "label": "Phase 1: Initial Access",
        "mitre_id": "TA0001 (Primary)",
        "color": "#eab308",
        "severity": "Medium"
    },
    2: {
        "name": "Command and Control",
        "label": "Phase 2: Command & Control",
        "mitre_id": "TA0011",
        "color": "#f97316",
        "severity": "High"
    },
    3: {
        "name": "Impact (Resource Exhaustion / DoS)",
        "label": "Phase 3: Impact",
        "mitre_id": "TA0040 (Primary)",
        "color": "#ef4444",
        "severity": "Critical"
    }
}

# =========================================================
# 2. CAUSAL DYNAMICS WORLD MODEL ARCHITECTURE
# =========================================================
class NetworkWorldModel(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=64, num_classes=4, dropout=0.2):
        super(NetworkWorldModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.dynamics_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        h_t = lstm_out[:, -1, :]
        return self.dynamics_head(h_t), self.classifier_head(h_t)

# =========================================================
# 3. SYSTEM ARTIFACT LOADER
# =========================================================
@st.cache_resource
def load_system():
    required_files = [
        "scaler_2018.pkl", "network_world_model_2018.pth",
        "baseline_lr.pkl", "baseline_rf.pkl", "dashboard_test_stream_2018.npz"
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        return None, f"Missing system artifacts: {', '.join(missing)}"
    try:
        device = torch.device("cpu")
        scaler = joblib.load("scaler_2018.pkl")
        checkpoint = torch.load("network_world_model_2018.pth", map_location=device)
        model = NetworkWorldModel(input_dim=len(DYNAMIC_FEATURES), hidden_dim=64, num_classes=4)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        optimal_T = checkpoint.get('optimal_T', 1.3808)
        lr_model = joblib.load("baseline_lr.pkl")
        rf_model = joblib.load("baseline_rf.pkl")
        fallback_data = np.load("dashboard_test_stream_2018.npz")
        return {
            "model": model, "scaler": scaler, "optimal_T": optimal_T,
            "lr": lr_model, "rf": rf_model, "fallback": fallback_data
        }, None
    except Exception as err:
        return None, f"Initialization error: {str(err)}"

# =========================================================
# 4. REAL PCAP -> BIDIRECTIONAL FLOW RECONSTRUCTION
# =========================================================
FLOW_TIMEOUT_SEC = 120.0
MAX_PACKETS = 20000
MIN_FLOWS_REQUIRED = 20

def _flow_key(pkt_ip, pkt_l4, proto):
    a = (pkt_ip.src, getattr(pkt_l4, "sport", 0))
    b = (pkt_ip.dst, getattr(pkt_l4, "dport", 0))
    if a <= b:
        return (a, b, proto)
    return (b, a, proto)

def extract_pcap_features(uploaded_pcap):
    """
    Parses a raw PCAP/PCAPNG file into bidirectional flow-level features.
    Returns (dataframe, note) -- note is None on success, or an explicit error string.
    Safely handles Windows file locks on malformed or corrupted inputs.
    """
    try:
        from scapy.all import rdpcap, IP, TCP, UDP
    except ImportError:
        return None, "Scapy is not installed in this environment. Run: pip install scapy"

    pcap_path = f"_temp_{uploaded_pcap.name}"
    packets = None
    try:
        with open(pcap_path, "wb") as f:
            f.write(uploaded_pcap.getbuffer())
        packets = rdpcap(pcap_path)
    except Exception as e:
        return None, f"Could not read this file as a PCAP/PCAPNG capture: {e}"
    finally:
        # Prevent Windows file-lock PermissionError [WinError 32]
        try:
            if os.path.exists(pcap_path):
                os.remove(pcap_path)
        except Exception:
            pass

    if packets is None or len(packets) == 0:
        return None, "The uploaded capture contains 0 packets."

    flows = {}
    forward_src = {}
    n_used = 0

    for pkt in packets[:MAX_PACKETS]:
        if IP not in pkt:
            continue
        ip = pkt[IP]
        if TCP in pkt:
            l4 = pkt[TCP]
            proto = "TCP"
        elif UDP in pkt:
            l4 = pkt[UDP]
            proto = "UDP"
        else:
            continue

        key = _flow_key(ip, l4, proto)
        ts = float(pkt.time)
        pkt_len = float(len(pkt))

        if key not in flows or (ts - flows[key]["last_ts"]) > FLOW_TIMEOUT_SEC:
            flows[key] = {
                "first_ts": ts, "last_ts": ts,
                "fwd_lens": [], "bwd_lens": [],
                "iats": [], "prev_ts": ts,
                "syn": 0, "ack": 0, "rst": 0,
                "init_fwd_win": 0.0, "init_fwd_win_set": False,
            }
            forward_src[key] = (ip.src, getattr(l4, "sport", 0))

        f = flows[key]
        if ts > f["prev_ts"]:
            f["iats"].append(ts - f["prev_ts"])
        f["prev_ts"] = ts
        f["last_ts"] = ts

        is_forward = (ip.src, getattr(l4, "sport", 0)) == forward_src[key]
        if is_forward:
            f["fwd_lens"].append(pkt_len)
            if proto == "TCP" and not f["init_fwd_win_set"]:
                f["init_fwd_win"] = float(getattr(l4, "window", 0) or 0)
                f["init_fwd_win_set"] = True
        else:
            f["bwd_lens"].append(pkt_len)

        if proto == "TCP":
            flags = l4.flags
            if flags.S:
                f["syn"] += 1
            if flags.A:
                f["ack"] += 1
            if flags.R:
                f["rst"] += 1

        n_used += 1

    if n_used == 0:
        return None, "No IPv4 TCP/UDP packets found in capture."

    rows = []
    for key, f in flows.items():
        duration = max(f["last_ts"] - f["first_ts"], 1e-6)
        tot_fwd = len(f["fwd_lens"])
        tot_bwd = len(f["bwd_lens"])
        if tot_fwd + tot_bwd < 2:
            continue
        totlen_fwd = float(sum(f["fwd_lens"]))
        totlen_bwd = float(sum(f["bwd_lens"]))
        
        # Scaling alignment: duration and IAT fields converted to microseconds (* 1e6)
        rows.append({
            'Flow Duration': duration * 1e6,
            'Tot Fwd Pkts': tot_fwd,
            'Tot Bwd Pkts': tot_bwd,
            'TotLen Fwd Pkts': totlen_fwd,
            'TotLen Bwd Pkts': totlen_bwd,
            'Fwd Pkt Len Mean': (totlen_fwd / tot_fwd) if tot_fwd else 0.0,
            'Bwd Pkt Len Mean': (totlen_bwd / tot_bwd) if tot_bwd else 0.0,
            'Flow Byts/s': (totlen_fwd + totlen_bwd) / duration,
            'Flow Pkts/s': (tot_fwd + tot_bwd) / duration,
            'Flow IAT Mean': (float(np.mean(f["iats"])) * 1e6) if f["iats"] else 0.0,
            'Flow IAT Std': (float(np.std(f["iats"])) * 1e6) if len(f["iats"]) > 1 else 0.0,
            'SYN Flag Cnt': f["syn"],
            'ACK Flag Cnt': f["ack"],
            'RST Flag Cnt': f["rst"],
            'Init Fwd Win Byts': f["init_fwd_win"],
            '_first_ts': f["first_ts"],
        })

    if len(rows) < MIN_FLOWS_REQUIRED:
        return None, (f"Only reconstructed {len(rows)} usable flow(s) from this capture "
                       f"(need at least {MIN_FLOWS_REQUIRED} for a {MIN_FLOWS_REQUIRED}-step "
                       f"sequence window).")

    df = pd.DataFrame(rows).sort_values("_first_ts").drop(columns=["_first_ts"]).reset_index(drop=True)
    return df, None

def process_dataframe(df_raw, scaler, window_size=20):
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    col_lookup = {c.lower(): c for c in df.columns}

    schema_map = {
        'total fwd packets': 'Tot Fwd Pkts', 'tot fwd pkts': 'Tot Fwd Pkts',
        'total backward packets': 'Tot Bwd Pkts', 'tot bwd pkts': 'Tot Bwd Pkts',
        'total length of fwd packets': 'TotLen Fwd Pkts', 'totlen fwd pkts': 'TotLen Fwd Pkts',
        'total length of bwd packets': 'TotLen Bwd Pkts', 'totlen bwd pkts': 'TotLen Bwd Pkts',
        'fwd packet length mean': 'Fwd Pkt Len Mean', 'fwd pkt len mean': 'Fwd Pkt Len Mean',
        'bwd packet length mean': 'Bwd Pkt Len Mean', 'bwd pkt len mean': 'Bwd Pkt Len Mean',
        'flow bytes/s': 'Flow Byts/s', 'flow byts/s': 'Flow Byts/s',
        'flow packets/s': 'Flow Pkts/s', 'flow pkts/s': 'Flow Pkts/s',
        'flow duration': 'Flow Duration',
        'flow iat mean': 'Flow IAT Mean', 'flow iat std': 'Flow IAT Std',
        'syn flag count': 'SYN Flag Cnt', 'syn flag cnt': 'SYN Flag Cnt',
        'ack flag count': 'ACK Flag Cnt', 'ack flag cnt': 'ACK Flag Cnt',
        'rst flag count': 'RST Flag Cnt', 'rst flag cnt': 'RST Flag Cnt',
        'init_win_bytes_forward': 'Init Fwd Win Byts', 'init fwd win byts': 'Init Fwd Win Byts',
    }

    canonical_df = pd.DataFrame()
    for feat in DYNAMIC_FEATURES:
        matched_col = None
        for alias, standard in schema_map.items():
            if standard == feat and alias in col_lookup:
                matched_col = col_lookup[alias]
                break
        if matched_col is not None:
            canonical_df[feat] = pd.to_numeric(df[matched_col], errors='coerce')
        elif feat in df.columns:
            canonical_df[feat] = pd.to_numeric(df[feat], errors='coerce')
        else:
            canonical_df[feat] = 0.0

    # Cap extreme rate values at the 99.5th percentile of the batch
    for col in ['Flow Pkts/s', 'Flow Byts/s']:
        finite = canonical_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        cap = float(finite.quantile(0.995)) if len(finite) else 0.0
        canonical_df[col] = canonical_df[col].replace([np.inf, -np.inf], cap)
    canonical_df.fillna(0.0, inplace=True)

    if len(canonical_df) < window_size:
        return None, f"Dataset provides {len(canonical_df)} rows. Minimum sequence lookback requires {window_size} rows."

    scaled_array = scaler.transform(canonical_df.values)
    sequences = [scaled_array[i: i + window_size] for i in range(0, len(scaled_array) - window_size + 1)]
    return np.array(sequences, dtype=np.float32), None

# =========================================================
# 5. CORE SYSTEM INITIALIZATION
# =========================================================
sys_bundle, err_msg = load_system()
if err_msg:
    st.error(f"❌ {err_msg}")
    st.info("Ensure all 5 model artifacts are available in the local directory.")
    st.stop()

model = sys_bundle["model"]
scaler = sys_bundle["scaler"]
optimal_T = sys_bundle["optimal_T"]
lr_model = sys_bundle["lr"]
rf_model = sys_bundle["rf"]
fallback_stream = sys_bundle["fallback"]

# =========================================================
# 6. SIDEBAR WORKFLOW CONTROLS
# =========================================================
st.sidebar.title("🎮 Telemetry Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Network Telemetry (PCAP or CSV)",
    type=["csv", "pcap", "pcapng"],
    help="Accepts raw packet captures (.pcap/.pcapng, reconstructed into flows locally) "
         "or pre-computed bidirectional flow records (.csv)."
)

k_horizon = st.sidebar.slider(
    "Forecast Horizon (K Steps)", min_value=1, max_value=10, value=5,
    help="Autoregressive forward steps simulated by the dynamics head P(S_t+1 | S_t)."
)

if "jump_idx" not in st.session_state:
    st.session_state.jump_idx = 0

if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(('.pcap', '.pcapng')):
            pcap_df, pcap_err = extract_pcap_features(uploaded_file)
            if pcap_df is None:
                st.sidebar.error(f"PCAP parsing failed: {pcap_err}")
                st.sidebar.warning("Showing benchmark demo stream instead -- NOT your uploaded capture.")
                sequences = fallback_stream['X_test']
                data_source_msg = "Fallback Stream (PCAP parsing failed -- see error above)"
            else:
                sequences, proc_err = process_dataframe(pcap_df, scaler)
                if proc_err:
                    st.sidebar.error(proc_err)
                    st.sidebar.warning("Showing benchmark demo stream instead -- NOT your uploaded capture.")
                    sequences = fallback_stream['X_test']
                    data_source_msg = "Fallback Stream (Reconstructed flows insufficient)"
                else:
                    data_source_msg = f"PCAP Reconstructed: {uploaded_file.name} ({len(pcap_df)} flows -> {len(sequences)} windows)"
        else:
            raw_df = pd.read_csv(uploaded_file)
            raw_df.columns = [str(c).strip() for c in raw_df.columns]
            label_col = [c for c in raw_df.columns if 'label' in c.lower()]
            if label_col:
                lbl = label_col[0]
                is_attack = ~raw_df[lbl].astype(str).str.upper().str.contains('BENIGN|NORMAL')
                attack_indices = np.where(is_attack)[0]
                if len(attack_indices) > 0:
                    first_attack = attack_indices[0]
                    start_i = max(0, first_attack - 100)
                    end_i = min(len(raw_df), first_attack + 1500)
                    sampled_df = raw_df.iloc[start_i:end_i].reset_index(drop=True)
                    st.session_state.jump_idx = min(105, max(0, len(sampled_df) - 21))
                else:
                    sampled_df = raw_df.head(2000)
            else:
                if len(raw_df) > 5000:
                    mid_point = len(raw_df) // 2
                    sampled_df = raw_df.iloc[mid_point: mid_point + 2000].reset_index(drop=True)
                else:
                    sampled_df = raw_df

            sequences, proc_err = process_dataframe(sampled_df, scaler)
            if proc_err:
                st.sidebar.error(proc_err)
                st.sidebar.warning("Showing benchmark demo stream instead -- NOT your uploaded file.")
                sequences = fallback_stream['X_test']
                data_source_msg = "Fallback Stream (Upload Invalid)"
            else:
                data_source_msg = f"Live File: {uploaded_file.name} ({len(sequences)} windows)"
    except Exception as e:
        st.sidebar.error(f"Ingestion failed: {e}")
        st.sidebar.warning("Showing benchmark demo stream instead -- NOT your uploaded file.")
        sequences = fallback_stream['X_test']
        data_source_msg = "Fallback Stream (Read Error)"
else:
    sequences = fallback_stream['X_test']
    data_source_msg = "Benchmark Test Stream (Balanced 4-Stage)"

st.sidebar.caption(f"**Data Ingestion Pipeline:** {data_source_msg}")

if "y_test" in fallback_stream and uploaded_file is None:
    st.sidebar.markdown("---")
    st.sidebar.write("⚡ **Jump to Stage Scenario:**")
    y_test_labels = fallback_stream["y_test"]
    c0, c1, c2, c3 = st.sidebar.columns(4)

    def get_stage_center_index(target_stage):
        matches = np.where(y_test_labels == target_stage)[0]
        return int(matches[len(matches) // 2]) if len(matches) > 0 else 0

    if c0.button("Phase 0"):
        st.session_state.jump_idx = get_stage_center_index(0)
    if c1.button("Phase 1"):
        st.session_state.jump_idx = get_stage_center_index(1)
    if c2.button("Phase 2"):
        st.session_state.jump_idx = get_stage_center_index(2)
    if c3.button("Phase 3"):
        st.session_state.jump_idx = get_stage_center_index(3)

window_idx = st.sidebar.slider(
    "Temporal Window Index (t)", min_value=0, max_value=max(0, len(sequences) - 1),
    value=min(st.session_state.jump_idx, max(0, len(sequences) - 1)), step=1
)

# =========================================================
# 7. INFERENCE, CAUSAL ROLLOUT & SALIENCY
# =========================================================
current_window = sequences[window_idx:window_idx + 1]
x_tensor = torch.tensor(current_window, dtype=torch.float32, requires_grad=True)

next_s1, logits = model(x_tensor)
calibrated_probs = torch.softmax(logits / optimal_T, dim=1).detach().numpy()[0]
predicted_stage = int(np.argmax(calibrated_probs))
confidence_score = float(calibrated_probs[predicted_stage])

target_logit = logits[0, predicted_stage]
target_logit.backward()
saliency = x_tensor.grad.data.abs().numpy()[0]
feature_importance = np.mean(saliency, axis=0)
sorted_feat_idx = np.argsort(feature_importance)[::-1]

rollout_stages = [predicted_stage]
rollout_confs = [confidence_score]
sim_window = current_window.copy()

with torch.no_grad():
    for _ in range(k_horizon):
        inp = torch.tensor(sim_window, dtype=torch.float32)
        pred_state, pred_logit = model(inp)
        probs = torch.softmax(pred_logit / optimal_T, dim=1).numpy()[0]
        stg = int(np.argmax(probs))
        rollout_stages.append(stg)
        rollout_confs.append(float(probs[stg]))
        sim_window = np.roll(sim_window, -1, axis=1)
        sim_window[0, -1, :] = pred_state.numpy()[0]

# =========================================================
# 8. PRESENTATION DASHBOARD
# =========================================================
st.title("🛡️ AI Network World Model: Predictive Threat Operations")
st.caption("Temporal State Dynamics | K-Step Autoregressive Simulation | MITRE ATT&CK Staging | Baseline Architecture Audit")

stage_info = STAGE_CONFIG[predicted_stage]
terminal_stage = rollout_stages[-1]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Assessment (t)", stage_info["label"])
m2.metric("Calibrated Confidence", f"{confidence_score * 100:.2f}%", f"T = {optimal_T:.4f}")
m3.metric("Temporal Context (W)", "20 Consecutive Flows")
m4.metric(f"Forecast Horizon (t+{k_horizon})", STAGE_CONFIG[terminal_stage]["label"], f"{rollout_confs[-1]*100:.1f}% Conf")

st.divider()
st.markdown("### ⚔️ Architecture Audit: Static Classifiers vs. World Model")
static_input = current_window[0, -1, :].reshape(1, -1)
lr_pred = lr_model.predict(static_input)[0]
lr_conf = np.max(lr_model.predict_proba(static_input)[0])
rf_pred = rf_model.predict(static_input)[0]
rf_conf = np.max(rf_model.predict_proba(static_input)[0])

b1, b2, b3 = st.columns(3)
with b1:
    st.markdown("#### 🔹 Logistic Regression")
    st.caption("Mandated Linear Baseline")
    st.write(f"**Classification:** {STAGE_CONFIG[lr_pred]['label']}")
    st.write(f"**Confidence:** {lr_conf*100:.2f}%")
    st.error("❌ Forward Simulation: **Unsupported** (Static)")
with b2:
    st.markdown("#### 🌲 Random Forest")
    st.caption("Static SOTA Ensemble")
    st.write(f"**Classification:** {STAGE_CONFIG[rf_pred]['label']}")
    st.write(f"**Confidence:** {rf_conf*100:.2f}%")
    st.error("❌ Forward Simulation: **Unsupported** (Static)")
with b3:
    st.markdown("#### 🌐 AI World Model (Ours)")
    st.caption("Causal State Dynamics $P(S_{t+1}|S_t)$")
    st.write(f"**Classification:** {stage_info['label']}")
    st.write(f"**Calibrated Conf:** {confidence_score*100:.2f}%")
    st.success(f"✅ Forward Simulation: **Forecasted +{k_horizon} Transitions**")

st.divider()
g1, g2 = st.columns([6, 4])

with g1:
    st.markdown("### 📈 Attack Progression Trajectory")
    steps = [f"t+{i}" if i > 0 else "Current (t)" for i in range(len(rollout_stages))]
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=steps, y=rollout_stages, mode='lines+markers+text',
        text=[f"Phase {s}<br>({c*100:.1f}%)" for s, c in zip(rollout_stages, rollout_confs)],
        textposition="top center",
        line=dict(color=stage_info["color"], width=3),
        marker=dict(size=12, color=[STAGE_CONFIG[s]["color"] for s in rollout_stages])
    ))
    fig_line.update_layout(
        yaxis=dict(title="Predicted Phase", tickmode='array', tickvals=[0, 1, 2, 3],
                    ticktext=["Phase 0: Baseline", "Phase 1: Initial Access",
                              "Phase 2: C2 Beacon", "Phase 3: Impact"], range=[-0.5, 3.8]),
        xaxis=dict(title="Discrete Forward Rollout Steps"),
        template="plotly_dark", height=380, margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_line, use_container_width=True)

with g2:
    st.markdown("### 🔍 Feature Attribution (Gradient Saliency)")
    top_feats = [DYNAMIC_FEATURES[i] for i in sorted_feat_idx[:6]]
    top_scores = [feature_importance[i] for i in sorted_feat_idx[:6]]
    fig_bar = go.Figure(go.Bar(x=top_scores[::-1], y=top_feats[::-1], orientation='h',
                                 marker=dict(color="#38bdf8")))
    fig_bar.update_layout(xaxis=dict(title="Attribution Score |∂Logit/∂X|"),
                            template="plotly_dark", height=380, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("### 📋 Autonomous SOC Incident Triage")
st.info(
    f"**Predicted Phase:** {stage_info['label']} ({stage_info['mitre_id']}). "
    f"Dominant telemetry attributions: **`{top_feats[0]}`** and **`{top_feats[1]}`**. "
    f"Autoregressive forward rollout forecasts transition to **{STAGE_CONFIG[terminal_stage]['label']}** "
    f"within **+{k_horizon} state transitions**."
)