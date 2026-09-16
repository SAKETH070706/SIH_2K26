import joblib
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from app import DYNAMIC_FEATURES, NetworkWorldModel

# 1. Load artifacts
data = np.load('dashboard_test_stream_2018.npz')
X_test = data['X_test']
y_test = data['y_test']

lr = joblib.load('baseline_lr.pkl')
rf = joblib.load('baseline_rf.pkl')

checkpoint = torch.load('network_world_model_2018.pth', map_location='cpu')
model = NetworkWorldModel(
    input_dim=len(DYNAMIC_FEATURES), hidden_dim=64, num_classes=4
)
model.load_state_dict(checkpoint['model_state'])
model.eval()
T = checkpoint.get('optimal_T', 1.3808)

# 2. World Model Inference (Recurrent Sequence Window W=20)
with torch.no_grad():
  x_tensor = torch.tensor(X_test, dtype=torch.float32)
  _, logits = model(x_tensor)
  wm_preds = torch.argmax(logits / T, dim=1).numpy()

# 3. Static Baselines Inference (Single-step static frame W=1)
X_static = X_test[:, -1, :]
lr_preds = lr.predict(X_static)
rf_preds = rf.predict(X_static)


# 4. Metric Computation
def evaluate_model(name, y_true, y_pred):
  rep = classification_report(
      y_true, y_pred, output_dict=True, zero_division=0
  )

  # Binary False Positive Rate: Class 0 (Normal) vs Classes 1, 2, 3 (Attack)
  y_true_bin = (y_true > 0).astype(int)
  y_pred_bin = (y_pred > 0).astype(int)
  cm_bin = confusion_matrix(y_true_bin, y_pred_bin)
  tn, fp, fn, tp = cm_bin.ravel()
  fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0

  macro_p = rep['macro avg']['precision'] * 100
  macro_r = rep['macro avg']['recall'] * 100
  macro_f1 = rep['macro avg']['f1-score'] * 100
  fpr_pct = fpr * 100

  print(f'=== {name} ===')
  print(f'Precision: {macro_p:.2f}%')
  print(f'Recall:    {macro_r:.2f}%')
  print(f'F1-Score:  {macro_f1:.2f}%')
  print(f'FPR:       {fpr_pct:.2f}%\n')
  return {
      'model': name,
      'precision': macro_p,
      'recall': macro_r,
      'f1': macro_f1,
      'fpr': fpr_pct,
  }


res_lr = evaluate_model(
    'Logistic Regression (Mandated Linear Baseline)', y_test, lr_preds
)
res_rf = evaluate_model('Random Forest (Static SOTA Ensemble)', y_test, rf_preds)
res_wm = evaluate_model(
    'AI Network World Model (Ours - Recurrent Dynamics)', y_test, wm_preds
)