# import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, List
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

class FetchGoldDataClass:
    def __init__(self, seq_length: int = 60, target_col: str = "Close"):
        self.seq_length = seq_length
        self.target_col = target_col

    def fetch_gld_hourly(self, days_back: int = 365, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch GLD 1-hour historical data from Yahoo Finance for the past `days_back` days,
        starting from the oldest date up to today.
        Returns a DataFrame with columns: ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        print("1. entered fetch_gld_hourly() function")
        chunks = []

        # start from earliest date
        s = datetime.now() - timedelta(days=days_back)
        e = s + timedelta(days=59)

        today = datetime.now()

        while s < today:
            if e > today:
                e = today

            df_chunk = yf.download('GLD', start=s, end=e, interval='1h', progress=False)

            if isinstance(df_chunk.columns, pd.MultiIndex):
                df_chunk.columns = [col[0] for col in df_chunk.columns]

            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df_chunk = df_chunk.dropna(subset=[col for col in required_cols if col in df_chunk.columns])
            df_chunk = df_chunk.reset_index()

            if not df_chunk.empty:
                chunks.append(df_chunk)

            # move the window forward
            s = e + timedelta(days=1)
            e = s + timedelta(days=59)

        res_df = pd.concat(chunks, ignore_index=True)
        return res_df
    
    def save_dataframe_to_csv(self, df: pd.DataFrame, path: str) -> bool:
        print("3. entered save_dataframe_to_csv() function")
        try:
            df.to_csv(path, index=False)
            return True
        except Exception as e:
            print(f"Error occured while saving the data, {e}")
            return False
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Simple cleaning the dataset before feature engineering.
        """
        print("4. entered preprocess() function")
        drop = [col for col in df.columns if col in ['High','Low','Open','Volume']]
        df = df.drop(columns=drop)
        df = df.ffill().bfill()
        print(f"\nNan count:\n{df.isnull().sum()}")
        df = df.dropna()
        print(f"\nNan count:\n{df.isnull().sum()}")
        print(f"Succesfully writed dataframe to csv file: {self.save_dataframe_to_csv(df, 'dataset.csv')}")
        return df
    
    def calculate_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate professional-grade technical indicators for institutional analysis"""
        print("2. entered calculate_enhanced_indicators() function")
        
        df = df.copy()
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
    
    def prepare_enhanced_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
            Prepare data with enhanced technical indicators.
            Enko: maybe no need this function.
        
        """
        print("Calculating enhanced technical indicators...")
        print(f"Input data shape: {df.shape}")
        print(f"Input columns: {df.columns.tolist()}")
        
        # Calculate enhanced indicators
        #--- Enko: the argument dataset is already with indicators, no need to calculate
        # df = self.calculate_enhanced_indicators(df)
        
        print(f"After indicators calculation shape: {df.shape}")
        print(f"After indicators columns: {df.columns.tolist()}")
        
        # Define feature list (20 professional indicators)
        features = ['Close', 'SMA7', 'SMA20', 'SMA50', 'EMA21', 'EMA50', 'RSI14', 'RSI30', 'MACD', 'MACD_Hist', 
                'BB_Width', 'BB_Position', 'Stoch_K', 'Williams_R', 'ATR_Pct', 'Tenkan', 'Kijun', 
                'Vol_Ratio', 'Price_Momentum_5', 'SR_Ratio']
        
        # Validate all features are present
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"ERROR: Missing features: {missing_features}")
            print("Available columns:", df.columns.tolist())
            raise Exception(f"Missing required features: {missing_features}")
        
        print(f"✓ All {len(features)} features validated")
        
        # Drop NaN rows
        original_len = len(df)
        df = df.dropna()
        dropped_rows = original_len - len(df)
        print(f"Dropped {dropped_rows} NaN rows, final shape: {df.shape}")
        
        return df, features
    
    def create_sequences(self, data: pd.DataFrame, seq_length: int = 60, price_col: str = "Gold_Price") -> Tuple[np.array, np.array]:
        print("5. making sequences for LSTM(X, y)")
        X, y = [], []

        for i in range(seq_length, len(data)):
            # Drop non-numeric/date columns
            X.append(data.iloc[i-seq_length:i].drop(columns=["Date"]).values)
            
            current_price = data.iloc[i][price_col]
            prev_price = data.iloc[i-1][price_col]
            
            if prev_price > 0:
                pct_change = (current_price - prev_price) / prev_price
                pct_change = np.clip(pct_change, -0.2, 0.2)
                y.append(pct_change)
            else:
                y.append(0.0)

        return np.array(X), np.array(y)

    def create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
            """
            Create LSTM sequences from the data.

            X: sequences of past timesteps (all numeric columns except Datetime)
            y: percentage change of target_col
            """
            print("5. making sequences for LSTM(X, y)")
            X, y = [], []

            # Ensure we only use numeric columns except Datetime
            feature_cols = data.select_dtypes(include=np.number).columns.tolist()

            for i in range(self.seq_length, len(data)):
                # Past seq_length rows for features
                X.append(data[feature_cols].iloc[i-self.seq_length:i].values)

                # Percentage change for target column
                current_price = data[self.target_col].iloc[i]
                prev_price = data[self.target_col].iloc[i-1]

                if prev_price > 0:
                    pct_change = (current_price - prev_price) / prev_price
                    pct_change = np.clip(pct_change, -0.2, 0.2)
                    y.append(pct_change)
                else:
                    y.append(0.0)

            return np.array(X), np.array(y)

    @staticmethod
    def train_test_split_sequences(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split sequences into train and test sets for time series forecasting.
        """
        print("6. dividing dataset for training")
        num_samples = len(X)
        train_size = int(num_samples * train_ratio)

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]

        return X_train, y_train, X_test, y_test
    
    def scale_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        feature_s = MinMaxScaler()
        target_s = MinMaxScaler()

        features = [col for col in df.columns if col not in ['Datetime','Close']]
        target = 'Close'

        df[features] = feature_s.fit_transform(df[features])
        df[target] = target_s.fit_transform(df[[target]])

        joblib.dump(feature_s ,'scalers/feature_s.pkl')
        joblib.dump(target_s ,'scalers/target_s.pkl')

        try: 
            df.to_csv('scaled_dataset.csv', index=False)
            saved = self.save_dataframe_to_csv(df, 'scaled_dataset.csv')
            print(f"scaled dataset saved: {saved}")
        except Exception as e:
            print(f"scaled dataset saved: {saved}, error message: {e}")
    
        return df