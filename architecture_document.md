```markdown
# Autonomous Threat Dynamics: A Causal Dual-Head Network World Model
**Technical Architecture & Scientific Specification | 2-Page Brief**

---

## 1. Executive Problem Statement & Mathematical Formulation
Traditional Intrusion Detection Systems (NIDS) frame threat detection as static, memoryless classification:
$$\hat{Y}_t = f(X_t)$$
This formulation fails to track sequential multi-step kill chains and cannot anticipate future transitions. 

Our system implements a **Causal Network World Model** that models the underlying environment transition dynamics:
$$P(S_{t+1} \mid S_t, \dots, S_{t-W+1})$$
where an observation sequence $X_{t-W+1:t} \in \mathbb{R}^{W \times D}$ ($W=20$ consecutive bidirectional flows, $D=15$ dynamic features) is mapped into a shared temporal latent representation $h_t \in \mathbb{R}^{64}$.

             ┌──────────────────────────────────────────────┐
             │  Input Sequence Window: X_{t-W+1:t} (20x15)  │
             └──────────────────────┬───────────────────────┘
                                    │
                                    ▼
             ┌──────────────────────────────────────────────┐
             │    2-Layer Recurrent Backbone (LSTM, H=64)   │
             └──────────────────────┬───────────────────────┘
                                    │ Latent State h_t
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
     ┌──────────────────────────────┐ ┌──────────────────────────────┐
     │ Dynamics Head (State Reg.)   │ │ Classifier Head (Kill-Chain) │
     │   Linear(64,32) -> ReLU      │ │   Linear(64,32) -> ReLU      │
     │   Linear(32,15)              │ │   Linear(32,4)               │
     └──────────────┬───────────────┘ └──────────────┬───────────────┘
                    │                                │
                    ▼                                ▼
             Forecast \hat{S}_{t+1}          Logits -> Calibrated P(Y_t)


### Multi-Task Objective
The model is trained end-to-end minimizing a composite multi-task objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{classification}}(Y_t, \hat{Y}_t) + \lambda \cdot \mathcal{L}_{\text{dynamics}}(S_{t+1}, \hat{S}_{t+1})$$
where $\mathcal{L}_{\text{classification}}$ is Cross-Entropy over MITRE phases, $\mathcal{L}_{\text{dynamics}}$ is Mean Squared Error (MSE) over continuous flow telemetry, and $\lambda = 0.5$ balances regularized latent representation learning.

---

## 2. Dynamic Telemetry Vector Formulation ($D=15$)
The system unifies packet-level TCP header dynamics and flow-level rate metrics:

1. **Flow Duration ($\mu\text{s}$):** Microsecond-scale active flow length.
2. **Tot Fwd Pkts / Tot Bwd Pkts:** Directional flow packet counts.
3. **TotLen Fwd Pkts / TotLen Bwd Pkts:** Directional byte volume.
4. **Fwd Pkt Len Mean / Bwd Pkt Len Mean:** Directional average packet size.
5. **Flow Byts/s / Flow Pkts/s:** Transmission rate intensity (asymptote-capped).
6. **Flow IAT Mean / Flow IAT Std ($\mu\text{s}$):** Packet inter-arrival timing statistics.
7. **SYN / ACK / RST Flag Cnt:** Packet-level control signal distributions.
8. **Init Fwd Win Byts:** Initial client TCP advertised window size.

---

## 3. Autoregressive Rollout & Uncertainty Calibration

### K-Step Autoregressive Simulation
To evaluate attack trajectory progression, the engine performs closed-loop forward simulation across horizon $K$:
1. Given sequence $X_t \in \mathbb{R}^{W \times D}$, compute $\hat{S}_{t+1} = \text{DynamicsHead}(\text{Backbone}(X_t))$.
2. Shift the lookback buffer left by 1 index: drop $S_{t-W+1}$, append predicted state $\hat{S}_{t+1}$ at index $-1$.
3. Feed the updated simulated window back into the model to predict $\hat{S}_{t+2}$ and phase probability $\hat{P}_{t+1}$.
4. Repeat up to step $t+K$, charting transition trajectories across phases before downstream systems are impacted.

### Post-Hoc Temperature Calibration
Raw classification logits $z_t$ are calibrated using optimal temperature scaling on validation partitions ($T = 1.3808$):
$$\hat{P}(Y_t = c \mid X_t) = \frac{\exp(z_{t,c} / T)}{\sum_{j=0}^{3} \exp(z_{t,j} / T)}$$
This reduces overconfidence on out-of-distribution transitions, improving reliability for autonomous triage.

---

## 4. Ingestion Engines & Real-Time Dissection

### Bidirectional PCAP Reconstruction
The Scapy-based network engine processes raw captures via:
* **Canonical 5-Tuple Key:** $\text{FlowKey} = (\min(A, B), \max(A, B), \text{Proto})$, where $A = (\text{IP}_{\text{src}}, \text{Port}_{\text{sport}})$.
* **Directional Attribution:** Sets the first-seen endpoint as the forward source (`forward_src`). Subsequent reverse-direction packets map to backward accumulators.
* **Microsecond Time Scaling:** Explicitly aligns timestamp deltas with the $10^6$ multiplier expected by the CSE-CIC-IDS-2018 scaler.
* **Fault-Tolerant IO:** Protected cleanup handlers isolate Scapy file descriptors, preventing Windows file-lock exceptions (`PermissionError [WinError 32]`).

---

## 5. Comparative Audit & Empirical Findings

| Architecture | Model Family | Sequential Memory ($W=20$) | Forward Projection ($t+K$) | Explanatory Interface |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Linear | ❌ No ($W=1$) | ❌ Unsupported | Static Coefficients |
| **Random Forest** | Tree Ensemble | ❌ No ($W=1$) | ❌ Unsupported | Impurity Metrics |
| **AI World Model** | Dual-Head LSTM | ✅ Yes ($W=20$) | ✅ Autoregressive ($t+K$) | Gradient Saliency $\left\|\frac{\partial \text{Logit}}{\partial X}\right\|$ |

## Empirical Boundary & Known Limitations

* **Single-Packet Reconnaissance:** In cross-dataset evaluations against CIC-IDS-2017 single-packet SYN sweeps (`Tot Fwd Pkts = 1`, `TotLen Fwd Pkts = 0`), the model classifies the flows as Phase 0 (Baseline Normal).
* **Root Cause:** The model was trained on CSE-CIC-IDS-2018 enterprise infiltration profiles (e.g., SSH/FTP brute-force and botnet beaconing), which consist of multi-packet, stateful TCP interactions. Single-packet exploratory probes lack payload exchange and inter-arrival momentum, falling below the sequence activation threshold ($z \approx -0.7$).
* **Mitigation Roadmap:** Future iterations require augmenting the training corpus with stateless packet-level scan captures and deploying a lightweight stateless pre-filter (e.g., eBPF/XDP) ahead of the recurrent World Model.


## Benchmark Performance Evaluation

Evaluated on the balanced 4-phase CSE-CIC-IDS-2018 test partition ($N=2,000$ flows across Baseline, Initial Access, C2, and Impact). All models were audited under identical feature distributions:

| Model Architecture | Input Scope | Precision | Recall | Macro F1 | False Positive Rate (FPR) | Autoregressive Rollout ($t+K$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Mandated)** | Static ($W=1$) | 96.74% | 96.45% | 96.40% | 14.00% | ❌ Unsupported |
| **Random Forest (Static Ensemble)** | Static ($W=1$) | 99.90% | 99.90% | 99.90% | 0.00% | ❌ Unsupported |
| **AI Network World Model (Ours)** | Recurrent ($W=20$) | **99.80%** | **99.80%** | **99.80%** | **0.00%** | **✅ Supported (+K Steps)** |

### Key Benchmark Takeaways
1. **False Positive Suppression:** The mandated Logistic Regression baseline exhibits an unacceptable 14.00% FPR on normal baseline traffic. The World Model eliminates these false alarms (0.00% FPR) by leveraging temporal sequence context.
2. **Beyond Static Detection:** While static ensembles (Random Forest) achieve parity on single-frame classification, they cannot forecast trajectory evolution. The World Model provides equivalent discriminative performance while regressing continuous state dynamics $\hat{S}_{t+1}$.