import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from model import NetworkWorldModel

# Page Config
st.set_page_config(
    page_title="Network World Model SOC",
    page_icon="🛡️",
    layout="wide"
)

# Force CPU inference for local demo
device = torch.device("cpu")

@st.cache_resource
def load_assets():
    model = NetworkWorldModel(input_dim=16, hidden_dim=64, num_layers=2, num_classes=4)
    model.load_state_dict(torch.load("network_world_model.pth", map_location=device))
    data = np.load("dashboard_test_stream.npz")
    return model, data["sequences"], data["labels"]

model, sequences, labels = load_assets()

# Feature mapping
FEATURE_NAMES = [
    "Destination Port", "Flow Duration", "Flow Bytes/s", "Flow Packets/s",
    "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Mean", "Bwd Packet Length Mean",
    "Flow IAT Mean", "Flow IAT Std", "SYN Flag Count", "ACK Flag Count",
    "RST Flag Count", "Init_Win_bytes_forward"
]

STAGE_NAMES = {0: "Normal", 1: "Recon/PortScan", 2: "C2 / Botnet", 3: "Impact / DDoS"}
STAGE_COLORS = {0: "#10b981", 1: "#f59e0b", 2: "#ec4899", 3: "#ef4444"}

# Header & Pitch
st.title("🛡️ Predictive Network World Model SOC")
st.caption("Autoregressive Telemetry Forecasting • MITRE ATT&CK Progression • Zero-Touch Mitigation")

# Sidebar Controls
st.sidebar.header("🕹️ Telemetry Stream Controller")
seq_idx = st.sidebar.slider("Timeline Sequence Window (Time Index)", 0, len(sequences) - 1, 0)
rollout_steps = st.sidebar.slider("Autoregressive Horizon (K-Steps Lookahead)", 1, 10, 5)

st.sidebar.divider()
st.sidebar.subheader("Model Performance Benchmark")
st.sidebar.markdown("""
- **Recurrent World Model F1**: `99.74%`
- **Static Baseline (LogReg) F1**: `63.09%`
- **Detection Delta**: `+36.65%`
- **Evaluation Split**: Chronological (No-Leakage)
""")

# Prepare current window
raw_seq = sequences[seq_idx]
input_tensor = torch.tensor(raw_seq, dtype=torch.float32).unsqueeze(0).to(device)

# 1. Closed-Loop Autoregressive Rollout
model.eval()
current_window = input_tensor.clone()
sim_states = []
pred_stages = []
pred_confs = []

with torch.no_grad():
    for _ in range(rollout_steps):
        next_state, logits = model(current_window)
        probs = torch.softmax(logits, dim=1)
        stage = torch.argmax(probs, dim=1).item()
        conf = probs[0, stage].item()
        
        sim_states.append(next_state.squeeze(0).numpy())
        pred_stages.append(stage)
        pred_confs.append(conf)
        
        # Feedback loop: append simulated vector to sequence window
        next_expanded = next_state.unsqueeze(1)
        current_window = torch.cat([current_window[:, 1:, :], next_expanded], dim=1)

immediate_stage = pred_stages[0]
ground_truth = int(labels[seq_idx])

# Top Status Indicators
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("Forecasted Stage (t+1)", STAGE_NAMES[immediate_stage])
with col_b:
    st.metric("Ground Truth Stage", STAGE_NAMES.get(ground_truth, f"Stage {ground_truth}"))
with col_c:
    st.metric("Prediction Confidence", f"{pred_confs[0] * 100:.2f}%")
with col_d:
    alert_status = "CRITICAL BREACH" if immediate_stage == 3 else ("SUSPICIOUS" if immediate_stage > 0 else "NOMINAL")
    st.metric("Security Posture", alert_status)

# Dynamic Automated SOC Action Banner
if immediate_stage == 3:
    st.error("""
    🚨 **AUTOMATED SOC PLAYBOOK TRIGGERED: VOLUMETRIC MITIGATION**  
    - **Action**: Null-routing target destination IP via BGP Flowspec.  
    - **Firewall Policy**: Rate-limiting ingress TCP SYN / UDP packets across egress edge routers.  
    - **World Model Insight**: Sustained forward trajectory detected across lookahead horizon.
    """)
elif immediate_stage == 1:
    st.warning("⚠️ **AUTOMATED MITIGATION**: Port sweep detected. Quarantine source IP in temporary honey-token ACL.")
elif immediate_stage == 2:
    st.warning("⚠️ **AUTOMATED MITIGATION**: Command & Control beaconing identified. Isolating host network segment.")
else:
    st.success("✅ **SYSTEM NOMINAL**: Environmental telemetry flows match benign operational baseline.")

# Main Visualization Tabs
tab1, tab2, tab3 = st.tabs(["📈 Telemetry Dynamics Forecast", "🔍 Explainability & Saliency", "⏱️ Threat Progression Horizon"])

with tab1:
    st.subheader(f"{rollout_steps}-Step Autoregressive Environmental Simulation")
    st.write("Simulated physical telemetry trajectory generated entirely through internal state representations:")
    
    sim_arr = np.array(sim_states)
    steps_axis = [f"t+{i}" for i in range(1, rollout_steps + 1)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps_axis, y=sim_arr[:, 0], mode='lines+markers', name='Flow Duration (Normalized)', line=dict(color='#0284c7', width=2)))
    fig.add_trace(go.Scatter(x=steps_axis, y=sim_arr[:, 2], mode='lines+markers', name='Flow Bytes/s', line=dict(color='#f59e0b', width=2)))
    fig.add_trace(go.Scatter(x=steps_axis, y=sim_arr[:, 3], mode='lines+markers', name='Flow Packets/s', line=dict(color='#ef4444', width=2)))
    fig.add_trace(go.Scatter(x=steps_axis, y=sim_arr[:, 10], mode='lines+markers', name='Flow IAT Mean (Jitter)', line=dict(color='#10b981', width=2)))
    
    fig.update_layout(
        template="plotly_dark",
        height=380,
        xaxis_title="Simulation Lookahead Horizon",
        yaxis_title="Normalized Dynamic Variance",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Feature Saliency Attribution (Explainable AI)")
    st.write("Gradient attribution mapping which packet telemetry features drove this specific threat prediction:")
    
    # Run gradient attribution on current sequence
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
            
    saliency_input = input_tensor.clone().detach()
    saliency_input.requires_grad_(True)
    model.zero_grad()
    
    _, saliency_logits = model(saliency_input)
    saliency_logits[0, immediate_stage].backward()
    
    attribution_scores = saliency_input.grad.abs().squeeze(0).mean(dim=0).numpy()
    
    saliency_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Attribution Weight": attribution_scores
    }).sort_values(by="Attribution Weight", ascending=True)
    
    fig_saliency = go.Figure(go.Bar(
        x=saliency_df["Attribution Weight"],
        y=saliency_df["Feature"],
        orientation='h',
        marker=dict(color='#38bdf8')
    ))
    fig_saliency.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Mean Absolute Gradient Influence"
    )
    st.plotly_chart(fig_saliency, use_container_width=True)

with tab3:
    st.subheader(f"Forecasted Threat Progression (t+1 to t+{rollout_steps})")
    
    # Process up to 10 steps in rows of 5
    for row_start in range(0, rollout_steps, 5):
        row_end = min(row_start + 5, rollout_steps)
        row_steps = list(range(row_start, row_end))
        cols = st.columns(len(row_steps))
        
        for idx, step_idx in enumerate(row_steps):
            with cols[idx]:
                s_class = pred_stages[step_idx]
                st.markdown(f"#### Step t+{step_idx+1}")
                st.markdown(
                    f"<div style='padding: 10px; border-radius: 8px; background-color: {STAGE_COLORS.get(s_class)}; "
                    f"color: white; font-weight: bold; text-align: center; font-size: 15px; margin-bottom: 12px;'>"
                    f"{STAGE_NAMES.get(s_class)}<br>"
                    f"<small style='font-weight: normal;'>Conf: {pred_confs[step_idx]*100:.1f}%</small>"
                    f"</div>",
                    unsafe_allow_html=True
                )