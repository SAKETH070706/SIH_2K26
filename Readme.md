# AI Network World Model for Autonomous Threat Assessment

An offline, specification-compliant Network Intrusion and Threat Dynamics World Model. Rather than classifying traffic frames as static independent events, this system models temporal network environment dynamics $P(S_{t+1} \mid S_t)$ over a sliding lookback window ($W=20$). It performs $K$-step autoregressive rollout forecasting along the MITRE ATT&CK kill-chain, uses temperature calibration ($T=1.3808$) for reliable confidence, and provides gradient saliency attribution for SOC explainability.

---

## 1. System Architecture & Features

* **Dual-Head Recurrent Architecture:** Shared 2-layer LSTM backbone ($H=64$, $p=0.2$) with two heads:
  * **Dynamics Head:** Predicts the next-step feature state vector $\hat{S}_{t+1}$.
  * **Classification Head:** Classifies into 4 MITRE ATT&CK attack phases.
* **Autoregressive State Rollout:** Forecasts attack trajectory escalation across a user-defined horizon ($t+1 \dots t+K$).
* **Dual Ingestion Engine:**
  * **PCAP/PCAPNG Mode:** Extracts bidirectional flows via an undirected 5-tuple canonical key, computing 15 dynamic features (microsecond-aligned durations, IATs, flag distributions, and initial forward window bytes).
  * **CSV Telemetry Mode:** Flexible column mapper that cleans schema variations and handles rate asymptotes via batch-level 99.5th percentile capping.
* **Comparative Baseline Audit:** Real-time side-by-side verification against standard baselines (Logistic Regression and Random Forest), illustrating why memoryless models cannot forecast future transitions.
* **Fully Offline Operations:** Operates without external API calls or internet dependencies.

---

## 2. Directory Structure & Required Artifacts

Ensure the following files are present in the application directory:

```text
├── app.py                           # Core Streamlit dashboard & inference engine
├── network_world_model_2018.pth     # Trained PyTorch dual-head model checkpoint
├── scaler_2018.pkl                  # Fitted StandardScaler (15 dynamic features)
├── baseline_lr.pkl                  # Mandated Logistic Regression baseline
├── baseline_rf.pkl                  # SOTA Random Forest ensemble baseline
├── dashboard_test_stream_2018.npz   # Balanced 4-phase test stream (500 samples/phase)
├── README.md                        # Setup and operational instructions
└── architecture_document.md         # 2-Page technical architecture specification