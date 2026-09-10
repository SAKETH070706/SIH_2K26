import torch
import torch.nn as nn

class NetworkWorldModel(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, num_layers=2, num_classes=4):
        super(NetworkWorldModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
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
        pred_next_state = self.dynamics_head(h_t)
        pred_mitre_logits = self.classifier_head(h_t)
        return pred_next_state, pred_mitre_logits