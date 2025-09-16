import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings('ignore')


class ProfessionalLSTM(nn.Module):
    def __init__(self, input_size=20, hidden_size=128, num_layers=3, dropout=0.3):
        super(ProfessionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size // 2, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        device = x.device
        # Bidirectional LSTM requires 2 * num_layers
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=device)

        # LSTM output
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Multi-head attention mechanism
        lstm_out_transposed = lstm_out.transpose(0, 1)  # (seq_len, batch, features)
        attn_out, _ = self.attention(lstm_out_transposed, lstm_out_transposed, lstm_out_transposed)
        attn_out = attn_out.transpose(0, 1)  # Back to (batch, seq_len, features)

        # Layer normalization and residual connection
        lstm_out = self.layer_norm(lstm_out + attn_out)

        # Global average pooling
        context = torch.mean(lstm_out, dim=1)

        # Dense prediction head
        out = self.relu(self.fc1(context))
        out = self.dropout1(out)
        out = self.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.relu(self.fc3(out))
        out = self.fc4(out)

        return out

    @staticmethod
    def train_enhanced_model(X_train, y_train, X_test, y_test):
        print("\n"+"*"* 15 + "TRAINING MODEL" + "*"* 15)
        """Train enhanced LSTM model with attention and dropout"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        input_size = X_train.shape[2]
        model = ProfessionalLSTM(input_size=input_size).to(device)

        criterion = nn.HuberLoss(delta=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model.train()
        epochs = 100

        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}')

            scheduler.step(avg_loss)
            if epoch > 10 and avg_loss < 0.00005:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Predictions
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_tensor).squeeze().cpu().numpy()
            test_pred = model(X_test_tensor).squeeze().cpu().numpy()

        return model, train_pred, test_pred, y_train_tensor.cpu().numpy(), y_test_tensor.cpu().numpy()


    def evaluate_and_plot(self, train_pred, y_train, test_pred, y_test):
        print("\n"+"*"* 15 + "CALCULATING RESULTS" + "*"* 15)
        """Evaluate model and plot predictions"""
        # Metrics
        metrics = {
            "Train R²": r2_score(y_train, train_pred),
            "Train MSE": mean_squared_error(y_train, train_pred),
            "Train MAE": mean_absolute_error(y_train, train_pred),
            "Test R²": r2_score(y_test, test_pred),
            "Test MSE": mean_squared_error(y_test, test_pred),
            "Test MAE": mean_absolute_error(y_test, test_pred),
        }

        print("\n📊 Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}")

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label="Actual", color="black")
        plt.plot(test_pred, label="Predicted", color="red", alpha=0.7)
        plt.title("Predicted vs Actual (Test Set)")
        plt.xlabel("Time")
        plt.ylabel("Gold Price")
        plt.legend()
        plt.show()

        return metrics
