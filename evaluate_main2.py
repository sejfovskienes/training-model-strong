"""
🎯 GLD MODEL EVALUATOR - LOAD & ANALYZE ONLY
============================================

This script loads your pre-trained GLD model and data to:
✅ Calculate accuracy metrics (no retraining)
✅ Generate professional plots
✅ Show prediction vs actual comparisons
✅ Export results for analysis

USAGE:
1. Make sure your training script saved these files:
   - gld_professional_model_[timestamp].pth
   - gld_scaler_[timestamp].pkl
   - gld_data_1h.npy (cached data)
   
2. Run: python gld_evaluator.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import os
import glob

class ProfessionalLSTM(nn.Module):
    """Same model architecture as training script"""
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
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        lstm_out_transposed = lstm_out.transpose(0, 1)
        attn_out, _ = self.attention(lstm_out_transposed, lstm_out_transposed, lstm_out_transposed)
        attn_out = attn_out.transpose(0, 1)
        
        lstm_out = self.layer_norm(lstm_out + attn_out)
        context = torch.mean(lstm_out, dim=1)
        
        out = self.relu(self.fc1(context))
        out = self.dropout1(out)
        out = self.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        
        return out

def load_saved_model():
    """Load the most recent trained model"""
    print("🔍 Looking for saved GLD models...")
    
    # Look for both naming patterns
    model_files = glob.glob("gld_professional_model_*.pth")  # With timestamp
    model_files.extend(glob.glob("gld_professional_model.pth"))  # Without timestamp
    
    if not model_files:
        raise FileNotFoundError("❌ No saved GLD model found! Please run training first.")
    
    # Prefer the most recent file
    if len(model_files) == 1:
        latest_model = model_files[0]
    else:
        latest_model = max(model_files, key=os.path.getctime)
        
    print(f"📁 Found model: {latest_model}")
    
    # Load model data
    try:
        checkpoint = torch.load(latest_model, map_location='cpu')
        print(f"✅ Model file loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model file: {e}")
        raise
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format with metadata
        model_config = checkpoint.get('model_architecture', {})
        input_size = model_config.get('input_size', 20)
        hidden_size = model_config.get('hidden_size', 128)
        num_layers = model_config.get('num_layers', 3)
        dropout = model_config.get('dropout', 0.3)
        
        model = ProfessionalLSTM(input_size, hidden_size, num_layers, dropout)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        features = checkpoint.get('features', [])
        sequence_length = checkpoint.get('sequence_length', 60)
        
        print(f"✅ Model loaded with {input_size} features")
        print(f"✅ Architecture: {num_layers} layers, {hidden_size} hidden units")
        
    else:
        # Old format - just state dict (your case)
        try:
            # Try to determine input size from the model state
            first_layer_key = [k for k in checkpoint.keys() if 'lstm.weight_ih_l0' in k]
            if first_layer_key:
                input_size = checkpoint[first_layer_key[0]].shape[1]
                print(f"🔍 Detected input size: {input_size}")
            else:
                input_size = 20  # Default
                print(f"⚠️ Using default input size: {input_size}")
            
            model = ProfessionalLSTM(input_size=input_size)  # Use detected/default parameters
            model.load_state_dict(checkpoint)
            features = []  # Will use default features
            sequence_length = 60  # Default
            print("✅ Model loaded (using default/detected architecture)")
            
        except Exception as e:
            print(f"❌ Error loading model state dict: {e}")
            raise
    
    model.eval()
    return model, features, sequence_length, latest_model

def load_saved_scaler():
    """Load the most recent scaler"""
    # Look for both naming patterns
    scaler_files = glob.glob("gld_scaler_*.pkl")  # With timestamp
    scaler_files.extend(glob.glob("gld_scaler.pkl"))  # Without timestamp
    
    if not scaler_files:
        print("⚠️ No saved scaler found, using default MinMaxScaler")
        return MinMaxScaler()
    
    # Prefer the most recent file
    if len(scaler_files) == 1:
        latest_scaler = scaler_files[0]
    else:
        latest_scaler = max(scaler_files, key=os.path.getctime)
        
    print(f"📁 Found scaler: {latest_scaler}")
    
    scaler = joblib.load(latest_scaler)
    print("✅ Scaler loaded")
    return scaler

def load_cached_data():
    """Load cached GLD data"""
    cache_file = 'gld_data_1h.npy'
    dates_file = 'gld_dates_1h.npy'
    
    if not (os.path.exists(cache_file) and os.path.exists(dates_file)):
        raise FileNotFoundError("❌ No cached GLD data found! Please run training first to generate cache.")
    
    print("📊 Loading cached GLD data...")
    data_array = np.load(cache_file)
    dates_array = np.load(dates_file, allow_pickle=True)
    
    dates = pd.to_datetime(dates_array)
    gld_data = pd.DataFrame(data_array, 
                           columns=['Open', 'High', 'Low', 'Close', 'Volume'],
                           index=dates)
    
    print(f"✅ Data loaded: {len(gld_data):,} samples")
    print(f"📅 Period: {gld_data.index[0]} → {gld_data.index[-1]}")
    print(f"💰 Price range: ${gld_data['Close'].min():.2f} - ${gld_data['Close'].max():.2f}")
    
    return gld_data

def calculate_technical_indicators(df):
    """Calculate same indicators as training (simplified version)"""
    print("🔧 Calculating technical indicators...")
    
    # Core indicators (same as training)
    df['SMA7'] = df['Close'].rolling(window=7).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['EMA21'] = df['Close'].ewm(span=21).mean()
    df['EMA50'] = df['Close'].ewm(span=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain_14 = gain.rolling(window=14).mean()
    avg_loss_14 = loss.rolling(window=14).mean()
    rs_14 = avg_gain_14 / avg_loss_14
    df['RSI14'] = 100 - (100 / (1 + rs_14))
    
    avg_gain_30 = gain.rolling(window=30).mean()
    avg_loss_30 = loss.rolling(window=30).mean()
    rs_30 = avg_gain_30 / avg_loss_30
    df['RSI30'] = 100 - (100 / (1 + rs_30))
    
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    df['MACD'] = macd_line - signal_line
    df['MACD_Hist'] = macd_line - signal_line
    
    # Bollinger Bands
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma20 + (std20 * 2)
    df['BB_Lower'] = sma20 - (std20 * 2)
    df['BB_Mid'] = sma20
    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']).fillna(0.1)
    bb_position_calc = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['BB_Position'] = bb_position_calc.fillna(0.5).clip(0, 1)
    
    # Additional indicators (simplified)
    low14 = df['Low'].rolling(window=14).min()
    high14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = (100 * ((df['Close'] - low14) / (high14 - low14))).fillna(50).clip(0, 100)
    df['Williams_R'] = (-100 * ((high14 - df['Close']) / (high14 - low14))).fillna(-50).clip(-100, 0)
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    df['ATR_Pct'] = (df['ATR'] / df['Close'] * 100).fillna(2.0)
    
    # Ichimoku
    high9 = df['High'].rolling(window=9).max()
    low9 = df['Low'].rolling(window=9).min()
    df['Tenkan'] = (high9 + low9) / 2
    
    high26 = df['High'].rolling(window=26).max()
    low26 = df['Low'].rolling(window=26).min()
    df['Kijun'] = (high26 + low26) / 2
    
    # Volume
    df['Vol_SMA10'] = df['Volume'].rolling(window=10).mean()
    df['Vol_SMA30'] = df['Volume'].rolling(window=30).mean()
    df['Vol_Ratio'] = (df['Volume'] / df['Vol_SMA30']).fillna(1.0).replace([np.inf, -np.inf], 1.0).clip(0, 10)
    
    # Price momentum
    df['Price_Momentum_5'] = (df['Close'] / df['Close'].shift(5) - 1).fillna(0.0)
    
    # Support/Resistance
    df['Resistance'] = df['High'].rolling(window=20).max()
    df['Support'] = df['Low'].rolling(window=20).min()
    df['SR_Ratio'] = ((df['Close'] - df['Support']) / (df['Resistance'] - df['Support'])).fillna(0.5).clip(0, 1)
    
    return df

def create_sequences_for_evaluation(data, seq_length=60):
    """Create sequences for evaluation (same as training)"""
    X, y = [], []
    
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        
        current_price = data[i, 0]
        prev_price = data[i-1, 0]
        
        if prev_price > 0:
            pct_change = (current_price - prev_price) / prev_price
            pct_change = np.clip(pct_change, -0.1, 0.1)  # ±10% max for gold
            y.append(pct_change)
        else:
            y.append(0.0)
    
    return np.array(X), np.array(y)

def evaluate_model_performance(model, X, y, sequence_dates, scaler):
    """Evaluate model and return predictions"""
    print("🎯 Evaluating model performance...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X).to(device)
    
    # Get predictions in batches to avoid memory issues
    batch_size = 1000
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            batch_pred = model(batch).squeeze().cpu().numpy()
            if len(batch_pred.shape) == 0:  # Single prediction
                predictions.append(float(batch_pred))
            else:
                predictions.extend(batch_pred)
    
    predictions = np.array(predictions)
    
    # Calculate metrics
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mse)
    
    # Direction accuracy
    direction_accuracy = np.mean((np.sign(y) == np.sign(predictions))) * 100
    
    # Correlation
    correlation = np.corrcoef(y, predictions)[0, 1]
    
    print(f"📊 EVALUATION RESULTS:")
    print(f"   MSE: {mse:.6f}")
    print(f"   MAE: {mae:.6f}")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   Direction Accuracy: {direction_accuracy:.2f}%")
    print(f"   Correlation: {correlation:.4f}")
    
    return predictions, {
        'mse': mse, 'mae': mae, 'rmse': rmse,
        'direction_accuracy': direction_accuracy,
        'correlation': correlation
    }

def create_evaluation_plots(y_actual, y_pred, sequence_dates, gld_data, metrics):
    """Create comprehensive evaluation plots"""
    print("📈 Creating evaluation plots...")
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('🏆 GLD Model Evaluation Results', fontsize=20, color='gold', fontweight='bold')
    
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Prediction vs Actual Scatter
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_actual, y_pred, alpha=0.6, color='gold', s=20)
    
    # Perfect prediction line
    min_val, max_val = min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual % Change', color='white')
    ax1.set_ylabel('Predicted % Change', color='white')
    ax1.set_title('Prediction Accuracy', fontsize=14, color='white')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add R² score
    r_squared = metrics['correlation'] ** 2
    ax1.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax1.transAxes, 
             fontsize=12, color='yellow', bbox=dict(boxstyle="round", facecolor='black', alpha=0.8))
    
    # 2. Time Series of Predictions vs Actual
    ax2 = fig.add_subplot(gs[0, 1:])
    recent_indices = slice(-2000, None)  # Last 2000 predictions
    
    ax2.plot(sequence_dates[recent_indices], y_actual[recent_indices], 
             label='Actual % Change', color='lime', linewidth=1.5, alpha=0.8)
    ax2.plot(sequence_dates[recent_indices], y_pred[recent_indices], 
             label='Predicted % Change', color='orange', linewidth=1.5, alpha=0.8)
    
    ax2.set_title('Recent Predictions vs Actual', fontsize=14, color='white')
    ax2.set_ylabel('% Change', color='white')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. GLD Price Chart
    ax3 = fig.add_subplot(gs[1, :])
    recent_data = gld_data.tail(2000)
    
    ax3.plot(recent_data.index, recent_data['Close'], color='white', linewidth=2, label='GLD Price')
    ax3.fill_between(recent_data.index, recent_data['Close'], alpha=0.1, color='gold')
    
    ax3.set_title('GLD Price Movement (Recent Period)', fontsize=14, color='white')
    ax3.set_ylabel('Price ($)', color='white')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Residual Analysis
    ax4 = fig.add_subplot(gs[2, 0])
    residuals = y_actual - y_pred
    ax4.scatter(y_pred, residuals, alpha=0.5, color='lightcoral', s=20)
    ax4.axhline(y=0, color='white', linestyle='--', alpha=0.8)
    ax4.set_xlabel('Predicted Values', color='white')
    ax4.set_ylabel('Residuals', color='white')
    ax4.set_title('Residual Analysis', fontsize=14, color='white')
    ax4.grid(True, alpha=0.3)
    
    # 5. Direction Accuracy Over Time
    ax5 = fig.add_subplot(gs[2, 1])
    
    correct_directions = (np.sign(y_actual) == np.sign(y_pred)).astype(int)
    window_size = min(100, len(correct_directions) // 10)
    rolling_accuracy = pd.Series(correct_directions).rolling(window_size).mean() * 100
    
    ax5.plot(rolling_accuracy.values, color='lime', linewidth=2)
    ax5.axhline(y=50, color='red', linestyle='--', alpha=0.8, label='Random Chance')
    ax5.set_xlabel('Sample Index', color='white')
    ax5.set_ylabel('Rolling Accuracy (%)', color='white')
    ax5.set_title(f'Direction Accuracy ({window_size}-sample rolling)', fontsize=14, color='white')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance Summary
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    summary_text = f"""
📊 MODEL EVALUATION SUMMARY

Direction Accuracy: {metrics['direction_accuracy']:.1f}%
Correlation: {metrics['correlation']:.4f}
RMSE: {metrics['rmse']:.6f}
MAE: {metrics['mae']:.6f}

📈 Total Samples: {len(y_actual):,}
📅 Data Period: {gld_data.index[0].date()} to {gld_data.index[-1].date()}
💰 Price Range: ${gld_data['Close'].min():.2f} - ${gld_data['Close'].max():.2f}

🏆 Status: {"🟢 EXCELLENT" if metrics['direction_accuracy'] > 55 else "🟡 GOOD" if metrics['direction_accuracy'] > 52 else "🔴 NEEDS WORK"}
    """
    
    ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes, fontsize=12,
             ha='center', va='center', color='white',
             bbox=dict(boxstyle="round,pad=1", facecolor='darkblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'gld_evaluation_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"📊 Evaluation plot saved: {filename}")
    
    plt.show()

def main():
    """Main evaluation function"""
    print("🎯 GLD MODEL EVALUATOR - LOAD & ANALYZE")
    print("=" * 50)
    
    try:
        # Load saved components
        model, features, sequence_length, model_file = load_saved_model()
        scaler = load_saved_scaler()
        gld_data = load_cached_data()
        
        # Prepare data (same as training)
        enhanced_data = calculate_technical_indicators(gld_data)
        enhanced_data = enhanced_data.dropna()
        
        # Use same features as training (or default if not available)
        if not features:
            features = ['Close', 'SMA7', 'SMA20', 'SMA50', 'EMA21', 'EMA50', 'RSI14', 'RSI30', 
                       'MACD', 'MACD_Hist', 'BB_Width', 'BB_Position', 'Stoch_K', 'Williams_R', 
                       'ATR_Pct', 'Tenkan', 'Kijun', 'Vol_Ratio', 'Price_Momentum_5', 'SR_Ratio']
        
        print(f"🔧 Using {len(features)} features: {features[:5]}...")
        
        # Scale data
        scaled_data = scaler.transform(enhanced_data[features])
        
        # Create sequences
        X, y = create_sequences_for_evaluation(scaled_data, sequence_length)
        sequence_dates = enhanced_data.index[sequence_length:]
        
        print(f"✅ Created {len(X):,} evaluation sequences")
        
        # Evaluate model
        predictions, metrics = evaluate_model_performance(model, X, y, sequence_dates, scaler)
        
        # Create plots
        create_evaluation_plots(y, predictions, sequence_dates, enhanced_data, metrics)
        
        # Save results
        results = {
            'model_file': model_file,
            'data_period': f"{gld_data.index[0]} to {gld_data.index[-1]}",
            'total_samples': len(X),
            'metrics': metrics,
            'features_used': features
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'gld_evaluation_results_{timestamp}.txt'
        
        with open(results_file, 'w') as f:
            f.write("GLD MODEL EVALUATION RESULTS\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Model File: {results['model_file']}\n")
            f.write(f"Data Period: {results['data_period']}\n")
            f.write(f"Total Samples: {results['total_samples']:,}\n")
            f.write(f"Sequence Length: {sequence_length}\n")
            f.write(f"Features: {len(features)}\n\n")
            f.write("PERFORMANCE METRICS:\n")
            f.write(f"Direction Accuracy: {metrics['direction_accuracy']:.2f}%\n")
            f.write(f"Correlation: {metrics['correlation']:.4f}\n")
            f.write(f"RMSE: {metrics['rmse']:.6f}\n")
            f.write(f"MAE: {metrics['mae']:.6f}\n")
            f.write(f"MSE: {metrics['mse']:.6f}\n\n")
            f.write("FEATURES USED:\n")
            for i, feature in enumerate(features, 1):
                f.write(f"{i:2d}. {feature}\n")
        
        print(f"📄 Results saved: {results_file}")
        
        print("\n🏆 EVALUATION COMPLETE!")
        print(f"📊 Direction Accuracy: {metrics['direction_accuracy']:.1f}%")
        print(f"🔗 Correlation: {metrics['correlation']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()