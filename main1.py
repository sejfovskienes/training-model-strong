#!/usr/bin/env python3
"""
Enhanced AI-Based GLD Price Prediction using LSTM + Attention
"""

import warnings
warnings.filterwarnings('ignore')

import time
import os
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# --- LSTM model remains the same
class ProfessionalLSTM(nn.Module):
    def __init__(self, input_size=20, hidden_size=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_size*2, num_heads=8, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size//2, 32)
        self.fc4 = nn.Linear(32,1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size*2)

    def forward(self, x):
        device = x.device
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size, device=device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out_t = lstm_out.transpose(0,1)
        attn_out,_ = self.attention(lstm_out_t, lstm_out_t, lstm_out_t)
        attn_out = attn_out.transpose(0,1)
        lstm_out = self.layer_norm(lstm_out + attn_out)
        context = torch.mean(lstm_out, dim=1)
        out = self.relu(self.fc1(context))
        out = self.dropout1(out)
        out = self.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        return out

# --- Fetch GLD Data
def fetch_gld_data(days_back=730):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    df = yf.download('GLD', start=start_date, end=end_date, interval='1h', progress=False)
    df = df[['Open','High','Low','Close','Volume']]
    df.dropna(inplace=True)
    return df

# --- Calculate indicators (same as before)
def calculate_enhanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    print("2. entered calculate_enhanced_indicators() function")

    # --- Moving Averages
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # --- RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    RS = roll_up / roll_down
    df['RSI'] = 100 - (100 / (1 + RS))

    # --- MACD (12-26 EMA + Signal 9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # --- Bollinger Bands
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = sma20 + 2 * std20
    df['BB_Lower'] = sma20 - 2 * std20
    df['BB_Mid'] = sma20
    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']).fillna(0.1)

    bb_position_calc = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    if isinstance(bb_position_calc, pd.DataFrame):
        bb_position_calc = bb_position_calc.iloc[:, 0]
    df['BB_Position'] = bb_position_calc.fillna(0.5).clip(0, 1)

    # --- ATR (Average True Range)
    high_low = (df['High'] - df['Low']).abs()
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    atr_pct = (df['ATR'] / df['Close'] * 100)
    if isinstance(atr_pct, pd.DataFrame):
        atr_pct = atr_pct.iloc[:, 0]
    df['ATR_Pct'] = atr_pct.fillna(2.0)

    # --- Stochastic Oscillator
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    stoch_k = 100 * (df['Close'] - low14) / (high14 - low14)
    if isinstance(stoch_k, pd.DataFrame):
        stoch_k = stoch_k.iloc[:, 0]
    df['Stoch_K'] = stoch_k
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # --- Momentum
    df['Momentum'] = df['Close'] / df['Close'].shift(10) - 1

    # --- Volatility
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()

    # --- Fill any remaining NaNs
    df = df.fillna(0)

    print("3. finished calculate_enhanced_indicators() function")
    return df

# --- Create sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        prev = data[i-1,0]
        curr = data[i,0]
        y.append((curr - prev)/prev)
    return np.array(X), np.array(y)

# --- Train model
def train_model(df):
    df = calculate_enhanced_indicators(df)
    features = ['Close','SMA7','SMA20','EMA21','RSI14']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    split = int(len(df)*0.8)
    X_train, y_train = create_sequences(scaled[:split], seq_length=60)
    X_test, y_test = create_sequences(scaled[split:], seq_length=60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProfessionalLSTM(input_size=X_train.shape[2]).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    for epoch in range(50):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X).squeeze()
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}')

    # Predict test set
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_pred_pct = model(X_test_tensor).squeeze().cpu().detach().numpy()
    
    # Convert pct change back to prices
    df_test = df.iloc[split+60:]
    y_test_prices = df_test['Close'].values
    y_pred_prices = []
    for i in range(len(y_pred_pct)):
        base = df_test['Close'].iloc[i-1] if i>0 else df['Close'].iloc[split+59]
        y_pred_prices.append(base*(1+y_pred_pct[i]))
    y_pred_prices = np.array(y_pred_prices)

    # Save model and scaler
    torch.save(model.state_dict(), 'gld_lstm.pth')
    joblib.dump(scaler,'gld_scaler.pkl')

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(df_test.index, y_test_prices, label='Actual Close')
    plt.plot(df_test.index, y_pred_prices, label='Predicted Close')
    plt.legend()
    plt.title('GLD Price Prediction')
    plt.show()
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test_prices, y_pred_prices))
    print(f'Test RMSE: {rmse:.2f}')

# --- Run
if __name__ == "__main__":
    df = fetch_gld_data()
    train_model(df)
