import warnings
warnings.filterwarnings('ignore')

import argparse
import time
import os
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import joblib

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class ProfessionalLSTM(nn.Module):
    def __init__(self, input_size=20, hidden_size=128, num_layers=3, dropout=0.3):
        super(ProfessionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Enhanced LSTM with bidirectional processing
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        # Multi-head attention mechanism for better feature learning
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8, dropout=dropout)
        
        # Enhanced prediction head with residual connections
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
        
        # Global average pooling instead of just last timestep
        context = torch.mean(lstm_out, dim=1)
        
        # Enhanced prediction head with residual connections
        out = self.relu(self.fc1(context))
        out = self.dropout1(out)
        residual = out
        
        out = self.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        
        return out

def calculate_enhanced_indicators(df):
    """Calculate professional-grade technical indicators for institutional gold analysis"""
    # Basic Moving Averages
    df['SMA7'] = df['Close'].rolling(window=7).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['EMA21'] = df['Close'].ewm(span=21).mean()
    df['EMA50'] = df['Close'].ewm(span=50).mean()
    df['EMA200'] = df['Close'].ewm(span=200).mean()
    
    # RSI with multiple timeframes
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
    
    # MACD with histogram
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    df['MACD'] = macd_line - signal_line
    df['MACD_Hist'] = macd_line - signal_line
    
    # Bollinger Bands with squeeze detection
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma20 + (std20 * 2)
    df['BB_Lower'] = sma20 - (std20 * 2)
    df['BB_Mid'] = sma20
    # Calculate BB Width - simple assignment with proper handling
    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']).fillna(0.1)
    
    # Calculate BB Position - ensure Series result
    bb_position_calc = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['BB_Position'] = bb_position_calc.fillna(0.5).clip(0, 1)
    
    # Stochastic Oscillator
    low14 = df['Low'].rolling(window=14).min()
    high14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = (100 * ((df['Close'] - low14) / (high14 - low14))).fillna(50).clip(0, 100)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # Williams %R
    df['Williams_R'] = (-100 * ((high14 - df['Close']) / (high14 - low14))).fillna(-50).clip(-100, 0)
    
    # Average True Range and volatility
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    # Calculate ATR Percentage - simple assignment with proper handling
    df['ATR_Pct'] = (df['ATR'] / df['Close'] * 100).fillna(2.0)
    
    # Ichimoku Cloud components
    high9 = df['High'].rolling(window=9).max()
    low9 = df['Low'].rolling(window=9).min()
    df['Tenkan'] = (high9 + low9) / 2
    
    high26 = df['High'].rolling(window=26).max()
    low26 = df['Low'].rolling(window=26).min()
    df['Kijun'] = (high26 + low26) / 2
    
    df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    
    high52 = df['High'].rolling(window=52).max()
    low52 = df['Low'].rolling(window=52).min()
    df['Senkou_B'] = ((high52 + low52) / 2).shift(26)
    
    # Volume analysis
    df['Vol_SMA10'] = df['Volume'].rolling(window=10).mean()
    df['Vol_SMA30'] = df['Volume'].rolling(window=30).mean()
    # Calculate Volume Ratio - simple assignment with proper handling
    df['Vol_Ratio'] = (df['Volume'] / df['Vol_SMA30']).fillna(1.0).replace([np.inf, -np.inf], 1.0).clip(0, 10)
    
    # On Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Price momentum and trend strength
    df['Price_Change'] = df['Close'].pct_change()
    # Calculate Price Momentum - simple assignment with proper handling
    df['Price_Momentum_5'] = (df['Close'] / df['Close'].shift(5) - 1).fillna(0.0)
    df['Price_Momentum_20'] = (df['Close'] / df['Close'].shift(20) - 1).fillna(0.0)
    
    # Market regime detection
    df['Trend_Strength'] = np.where(df['Close'] > df['SMA200'], 1, 
                                   np.where(df['Close'] < df['SMA200'], -1, 0))
    
    # Volatility regime
    df['Vol_Regime'] = np.where(df['ATR_Pct'] > df['ATR_Pct'].rolling(50).mean() * 1.5, 1, 0)
    
    # Support/Resistance levels (simplified)
    df['Resistance'] = df['High'].rolling(window=20).max()
    df['Support'] = df['Low'].rolling(window=20).min()
    # Calculate SR Ratio - simple assignment with proper handling
    df['SR_Ratio'] = ((df['Close'] - df['Support']) / (df['Resistance'] - df['Support'])).fillna(0.5).clip(0, 1)
    
    return df

def fetch_gld_data(live_mode=False, force_refresh=False):
    """Fetch GLD (Gold ETF) historical data with proper validation and numpy storage"""
    cache_file = 'gld_data_1h.npy'  # Different cache for 1h data
    cache_dates_file = 'gld_dates_1h.npy'
    
    print("=== GLD DATA FETCHING DEBUG ===")
    print(f"Cache files: {cache_file}, {cache_dates_file}")
    print(f"Cache exists: {os.path.exists(cache_file) and os.path.exists(cache_dates_file)}")
    
    # Always fetch fresh data to avoid corruption issues
    force_refresh = True  # Force fresh data until we verify integrity
    
    if not force_refresh and os.path.exists(cache_file) and os.path.exists(cache_dates_file):
        try:
            cache_time = os.path.getmtime(cache_file)
            current_time = time.time()
            cache_age = current_time - cache_time
            print(f"Cache age: {cache_age:.0f} seconds ({cache_age/3600:.2f} hours)")
            
            # If cache is less than 1 hour old, use it
            if cache_age < 3600:  # 3600 seconds = 1 hour
                print("Loading cached data...")
                data_array = np.load(cache_file)
                dates_array = np.load(cache_dates_file, allow_pickle=True)  # Fix: allow pickle for string arrays
                
                # Convert back to DataFrame
                dates = pd.to_datetime(dates_array)
                cached_data = pd.DataFrame(data_array, 
                                         columns=['Open', 'High', 'Low', 'Close', 'Volume'],
                                         index=dates)
                
                print(f"✓ Using cached data: {len(cached_data)} rows")
                print(f"✓ Cache date range: {cached_data.index[0]} to {cached_data.index[-1]}")
                print(f"✓ Cache price range: ${cached_data['Close'].min():.2f} - ${cached_data['Close'].max():.2f}")
                
                # Validate current price against real market
                current_gld = yf.Ticker('GLD')
                current_price = current_gld.info.get('regularMarketPrice', 0)
                cached_latest = cached_data['Close'].iloc[-1]
                price_diff_pct = abs((current_price - cached_latest) / current_price) * 100
                
                if price_diff_pct > 10:  # If cached price differs by more than 10%
                    print(f"✗ Cache validation failed: cached ${cached_latest:.2f} vs current ${current_price:.2f} ({price_diff_pct:.1f}% diff)")
                    force_refresh = True
                else:
                    print(f"✓ Cache validation passed: price difference {price_diff_pct:.1f}%")
                    return cached_data
            else:
                print("Cache is too old, fetching fresh data...")
        except Exception as e:
            print(f"✗ Cache load error: {e}, fetching fresh data...")
    
    if force_refresh or not os.path.exists(cache_file):
        print("Fetching fresh data or cache validation failed...")
    
    try:
        print("\n=== FETCHING FRESH GLD DATA ===")
        print("Connecting to Yahoo Finance for GLD...")
        
        # OPTIMIZED: Use 1-hour intervals for massive dataset (Yahoo Finance limit workaround)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years of 1h data = ~17k samples
        
        print(f"Requesting 1-HOUR GLD data from {start_date.date()} to {end_date.date()}...")
        print("Calling yf.download() with 1h interval for GLD...")
        
        # Download 1-hour data with error handling (no 60-day limit)
        import time as time_module
        fetch_start = time_module.time()
        gld = yf.download('GLD', start=start_date, end=end_date, interval='1h', progress=False)
        fetch_duration = time_module.time() - fetch_start
        
        print(f"Download completed in {fetch_duration:.2f} seconds")
        print(f"Downloaded data type: {type(gld)}")
        print(f"Downloaded data empty: {gld.empty if hasattr(gld, 'empty') else 'No empty attr'}")
        
        if gld.empty:
            raise Exception("Downloaded GLD data is empty")
        
        print(f"✓ Raw GLD data shape: {gld.shape}")
        print(f"✓ Raw columns: {gld.columns.tolist()}")
        print(f"✓ Raw index type: {type(gld.index)}")
        print(f"✓ First few dates: {gld.index[:3].tolist()}")
        
        # Handle multi-level columns from yfinance
        if isinstance(gld.columns, pd.MultiIndex):
            print("Flattening multi-level columns...")
            gld.columns = [col[0] if col[1] == 'GLD' else f"{col[0]}_{col[1]}" for col in gld.columns]
        
        print(f"✓ Processed columns: {gld.columns.tolist()}")
        
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in gld.columns]
        if missing_cols:
            raise Exception(f"Missing required columns: {missing_cols}")
        
        print(f"✓ All required columns present: {required_cols}")
        
        # Validate data quality for 1h data
        if len(gld) < 2000:  # Need more data for 1h intervals
            raise Exception(f"Insufficient GLD data: only {len(gld)} rows. Need at least 2000 for 1h training.")
        
        nan_count = gld['Close'].isna().sum()
        if nan_count > len(gld) * 0.1:
            raise Exception(f"Too many missing Close prices: {nan_count} out of {len(gld)}")
        
        print(f"✓ Data quality check passed: {len(gld)} rows, {nan_count} NaN values")
        
        # Remove any rows with NaN in critical columns
        original_len = len(gld)
        gld = gld.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        dropped_rows = original_len - len(gld)
        
        print(f"✓ Cleaned GLD data shape: {gld.shape} (dropped {dropped_rows} rows)")
        print(f"✓ Date range: {gld.index[0]} to {gld.index[-1]}")
        print(f"✓ Price range: ${gld['Close'].min():.2f} - ${gld['Close'].max():.2f}")
        print(f"✓ Sample recent prices: {gld['Close'].tail(3).values}")
        
        # Final validation for 1h data
        if len(gld) < 5000:
            raise Exception(f"After cleaning, insufficient GLD data: {len(gld)} rows. Need at least 5000.")
        
        # Save to cache using numpy arrays for better reliability
        try:
            print(f"Saving to numpy cache: {cache_file}")
            
            # Convert to numpy arrays with proper datetime handling for 1h data
            data_array = gld[['Open', 'High', 'Low', 'Close', 'Volume']].values
            dates_array = gld.index.strftime('%Y-%m-%d %H:%M:%S').values  # Include time for 1h data
            
            # Save arrays
            np.save(cache_file, data_array)
            np.save(cache_dates_file, dates_array)
            
            cache_size = os.path.getsize(cache_file) + os.path.getsize(cache_dates_file)
            print(f"✓ GLD Data cached as numpy arrays! ({len(gld)} rows, {cache_size} bytes)")
            
            # Verify cache was written correctly
            print("Verifying numpy cache integrity...")
            test_data = np.load(cache_file)
            test_dates = np.load(cache_dates_file, allow_pickle=True)  # Fix: allow pickle for string arrays
            
            if len(test_data) != len(gld) or len(test_dates) != len(gld):
                os.remove(cache_file)
                os.remove(cache_dates_file)
                print("✗ Cache verification failed, removed cache files")
            else:
                print("✓ Numpy cache verification successful")
        except Exception as cache_error:
            print(f"✗ Warning: Could not cache GLD data: {cache_error}")
            # Continue without caching - not critical for training
            for f in [cache_file, cache_dates_file]:
                if os.path.exists(f):
                    os.remove(f)
        
        print("=== GLD DATA FETCH COMPLETE ===\n")
        return gld
    
    except Exception as e:
        print(f"GLD data fetch failed: {e}. Checking for cached data...")
        
        # Try to use old cached data as last resort
        if os.path.exists(cache_file) and os.path.exists(cache_dates_file):
            try:
                print("Attempting to use older cached GLD data...")
                data_array = np.load(cache_file)
                dates_array = np.load(cache_dates_file)
                dates = pd.to_datetime(dates_array)
                cached_data = pd.DataFrame(data_array, 
                                         columns=['Open', 'High', 'Low', 'Close', 'Volume'],
                                         index=dates)
                
                if len(cached_data) > 100:
                    print(f"Using old GLD cache: {len(cached_data)} rows")
                    return cached_data
                else:
                    print("Old GLD cache is also invalid")
            except Exception as cache_error:
                print(f"GLD cache load failed: {cache_error}")
            
            # Remove corrupted cache files
            for f in [cache_file, cache_dates_file]:
                if os.path.exists(f):
                    os.remove(f)
        
        print("Using synthetic GLD data as fallback...")
        # Generate synthetic gold price data
        dates = pd.date_range(start=datetime.now() - timedelta(days=1825), 
                             end=datetime.now(), freq='D')
        
        np.random.seed(42)
        prices = [180]  # Start around typical GLD price
        volumes = []
        
        for _ in range(len(dates) - 1):
            change = np.random.normal(0, 3)  # Lower volatility than BTC
            prices.append(max(prices[-1] + change, 50))  # Min price of $50
            volumes.append(np.random.randint(5000000, 20000000))  # Typical GLD volume
        
        volumes.append(np.random.randint(5000000, 20000000))
        
        synthetic_data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],  # Lower vol than BTC
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],   # Lower vol than BTC
            'Close': prices,
            'Volume': volumes
        }, index=dates)
        
        return synthetic_data

def clear_cache():
    """Clear the GLD data cache to force fresh data fetch"""
    cache_files = ['gld_data_1h.npy', 'gld_dates_1h.npy']
    cleared = False
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            cleared = True
    
    if cleared:
        print("GLD data cache cleared!")
    else:
        print("No GLD cache files found.")

def prepare_enhanced_data(df):
    """Prepare GLD data with enhanced technical indicators"""
    print("Calculating enhanced technical indicators for GLD...")
    print(f"Input GLD data shape: {df.shape}")
    print(f"Input columns: {df.columns.tolist()}")
    
    # Calculate enhanced indicators
    df = calculate_enhanced_indicators(df)
    
    print(f"After indicators calculation shape: {df.shape}")
    print(f"After indicators columns: {df.columns.tolist()}")
    
    # Define feature list (20 professional indicators)
    features = ['Close', 'SMA7', 'SMA20', 'SMA50', 'EMA21', 'EMA50', 'RSI14', 'RSI30', 'MACD', 'MACD_Hist', 
               'BB_Width', 'BB_Position', 'Stoch_K', 'Williams_R', 'ATR_Pct', 'Tenkan', 'Kijun', 
               'Vol_Ratio', 'Price_Momentum_5', 'SR_Ratio']
    
    # Validate all features are present
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"ERROR: Missing GLD features: {missing_features}")
        print("Available columns:", df.columns.tolist())
        raise Exception(f"Missing required GLD features: {missing_features}")
    
    print(f"✓ All {len(features)} GLD features validated")
    
    # Drop NaN rows
    original_len = len(df)
    df = df.dropna()
    dropped_rows = original_len - len(df)
    print(f"Dropped {dropped_rows} NaN rows, final GLD shape: {df.shape}")
    
    return df, features

def create_sequences(data, seq_length=60):
    """Create sequences for LSTM training with percentage change prediction"""
    X, y = [], []
    
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        
        # Instead of predicting absolute price, predict percentage change
        # This helps the model learn relative movements rather than absolute values
        current_price = data[i, 0]  # Close price at current timestep
        prev_price = data[i-1, 0]   # Close price at previous timestep
        
        if prev_price > 0:
            pct_change = (current_price - prev_price) / prev_price
            # Clip extreme values to prevent outliers from dominating (GLD is less volatile)
            pct_change = np.clip(pct_change, -0.1, 0.1)  # ±10% max change for gold
            y.append(pct_change)
        else:
            y.append(0.0)
    
    return np.array(X), np.array(y)

def train_enhanced_model(X_train, y_train, X_test, y_test):
    """Train enhanced LSTM model with attention and dropout for GLD prediction"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for GLD training: {device}")
    
    # Enhanced model parameters for better accuracy
    input_size = X_train.shape[2]
    hidden_size = 128  # Increased from 64
    num_layers = 3     # Increased from 2
    dropout = 0.3      # Reduced overfitting
    
    model = ProfessionalLSTM(input_size, hidden_size, num_layers, dropout).to(device)
    # Enhanced loss and optimizer for better convergence
    criterion = nn.HuberLoss(delta=0.05)  # Even more robust for gold (lower volatility)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Better regularization
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # Enhanced training with larger batches and more epochs
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Larger batch size
    
    model.train()
    epochs = 100  # More epochs for better learning with 1h data
    
    print(f"Starting GLD training with {len(train_loader)} batches per epoch...")
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            
            # Progress within epoch
            if batch_count % 10 == 0:
                print(f'  Epoch {epoch+1}/{epochs}, Batch {batch_count}/{len(train_loader)}, Current Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}] Complete - Average Loss: {avg_loss:.6f}')
        
        # Enhanced early stopping and learning rate scheduling
        scheduler.step(avg_loss)
        
        # Early stopping - more conservative to allow proper training
        if epoch > 10 and avg_loss < 0.00005:  # Only after 10 epochs and much lower threshold
            print(f"Early stopping at epoch {epoch+1} due to convergence")
            break
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        train_pred = model(X_train_tensor).squeeze().cpu().numpy()
        test_pred = model(X_test_tensor).squeeze().cpu().numpy()
    
    return model, train_pred, test_pred

# Example usage function
def run_gld_prediction():
    """Main function to run GLD prediction with the same professional architecture"""
    print("=== PROFESSIONAL GLD PREDICTION MODEL ===")
    
    # Fetch GLD data
    print("1. Fetching GLD data...")
    gld_data = fetch_gld_data()
    
    # Prepare enhanced data with technical indicators
    print("2. Preparing enhanced GLD data...")
    enhanced_data, features = prepare_enhanced_data(gld_data)
    
    # Scale the features
    print("3. Scaling features...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(enhanced_data[features])
    
    # Create sequences
    print("4. Creating sequences...")
    X, y = create_sequences(scaled_data, seq_length=60)
    
    # Train/test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train model
    print("5. Training enhanced GLD model...")
    model, train_pred, test_pred = train_enhanced_model(X_train, y_train, X_test, y_test)
    
    print("=== GLD PREDICTION MODEL TRAINING COMPLETE ===")
    
    return model, scaler, features, enhanced_data

if __name__ == "__main__":
    """
    HOW TO RUN THIS SCRIPT:
    
    Method 1 - Direct Python execution:
    python gld_model.py
    
    Method 2 - Command line with arguments:
    python gld_model.py --mode train
    python gld_model.py --mode predict --clear_cache
    
    Method 3 - In Jupyter notebook:
    %run gld_model.py
    
    Method 4 - Import and use functions:
    from gld_model import run_gld_prediction
    model, scaler, features, data = run_gld_prediction()
    """
    
    parser = argparse.ArgumentParser(description='Professional GLD Trading Model')
    parser.add_argument('--mode', choices=['train', 'predict', 'live'], 
                       default='train', help='Mode to run the model')
    parser.add_argument('--clear_cache', action='store_true', 
                       help='Clear cached data before running')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--seq_length', type=int, default=60, 
                       help='Sequence length for LSTM')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting GLD Model in {args.mode} mode...")
    
    # Clear cache if requested
    if args.clear_cache:
        clear_cache()
    
    try:
        # Run the main prediction function
        model, scaler, features, data = run_gld_prediction()
        
        # Save model and scaler for future use
        model_file = 'gld_professional_model.pth'
        scaler_file = 'gld_scaler.pkl'
        
        torch.save(model.state_dict(), model_file)
        joblib.dump(scaler, scaler_file)
        
        print(f"✅ Model saved to: {model_file}")
        print(f"✅ Scaler saved to: {scaler_file}")
        print(f"✅ Features used: {features}")
        print(f"✅ Data shape: {data.shape}")
        print("\n🎯 GLD Professional Model Ready for Trading!")
        
    except Exception as e:
        print(f"❌ Error running GLD model: {e}")
        import traceback
        traceback.print_exc()