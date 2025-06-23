import sqlite3 as sq
import pandas as pd
import numpy as np

def load_data():
    """Loads data from the SQLite database."""
    with sq.connect("oister.db") as conn:
        query = "SELECT stonks.*, symbol_types.tag FROM stonks JOIN symbol_types USING (symbol)"
        return pd.read_sql(query, conn)

def engineer_features(df, price_col, windows):
    """
    Engineers a comprehensive set of technical analysis and momentum features.

    Args:
        df (pd.DataFrame): Input dataframe, grouped by symbol.
        price_col (str): The price column to use for calculations.
        windows (dict): A dictionary of window sizes for various indicators.

    Returns:
        pd.DataFrame: DataFrame with new features.
    """
    # --- Momentum Indicators ---
    # 1. Moving Average Ratio (captures trend)
    short_ma = df.groupby('symbol')[price_col].rolling(window=windows['short_ma']).mean()
    long_ma = df.groupby('symbol')[price_col].rolling(window=windows['long_ma']).mean()
    df['ma_ratio'] = (short_ma / long_ma).reset_index(level=0, drop=True)

    # 2. Relative Strength Index (RSI - momentum oscillator)
    delta = df.groupby('symbol')[price_col].diff()
    gain = delta.where(delta > 0, 0).rolling(window=windows['rsi']).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=windows['rsi']).mean()
    rs = gain / (loss + 1e-9) # Add epsilon to avoid division by zero
    df['rsi'] = (100 - (100 / (1 + rs))).reset_index(level=0, drop=True)
    
    # 3. MACD (trend-following momentum)
    ema_short = df.groupby('symbol')[price_col].ewm(span=windows['macd_fast'], adjust=False).mean()
    ema_long = df.groupby('symbol')[price_col].ewm(span=windows['macd_slow'], adjust=False).mean()
    df['macd'] = (ema_short - ema_long).reset_index(level=0, drop=True)
    df['macd_signal'] = df.groupby('symbol')['macd'].ewm(span=windows['macd_sign'], adjust=False).mean().reset_index(level=0, drop=True)

    # --- Volatility and Lag Features ---
    # Using historical pct_change, which is already calculated
    # 4. Rolling Volatility
    df['volatility'] = df.groupby('symbol')['pct_change'].rolling(window=windows['vol']).std().reset_index(level=0, drop=True)

    # 5. Lagged Returns
    for lag in range(1, windows['lags'] + 1):
        df[f'lag_{lag}'] = df.groupby('symbol')['pct_change'].shift(lag)
        
    return df

def create_ml_dataset(df, price_col, target_window, feature_windows):
    """
    Creates a full dataset with engineered features and a forward-looking target.

    Args:
        df (pd.DataFrame): Input dataframe sorted by symbol and timestamp.
        price_col (str): The price column for calculations.
        target_window (int): The rolling window for calculating target thresholds.
        feature_windows (dict): Dictionary of window sizes for feature engineering.

    Returns:
        pd.DataFrame: A clean DataFrame ready for modeling.
    """
    df = df.sort_values(['symbol', 'timestamp']).copy()

    # --- 1. Calculate base percentage change (used for features and target) ---
    df['pct_change'] = df.groupby('symbol')[price_col].pct_change()

    # --- 2. Engineer all predictive features from past and current data ---
    df = engineer_features(df, price_col, feature_windows)
    
    # --- 3. Calculate DYNAMIC thresholds for the target variable ---
    rolling_quantiles = df.groupby('symbol')['pct_change'].rolling(target_window, min_periods=target_window//2)
    df['lower_th'] = rolling_quantiles.quantile(0.25).reset_index(level=0, drop=True)
    df['upper_th'] = rolling_quantiles.quantile(0.75).reset_index(level=0, drop=True)

    # --- 4. Create the TARGET variable (future direction) ---
    future_pct_change = df.groupby('symbol')[price_col].pct_change().shift(-1)
    
    conditions = [
        future_pct_change <= df['lower_th'],
        future_pct_change >= df['upper_th'],
    ]
    choices = ['FALL', 'RISE']
    df['direction'] = np.select(conditions, choices, default='NEUTRAL')

    # --- 5. Clean up and finalize ---
    # Drop rows with NaNs created by all the rolling windows
    final_df = df.dropna().copy()
    
    # Drop intermediate columns used only for target calculation
    return final_df.drop(columns=['pct_change', 'lower_th', 'upper_th'])


def process_minute_data(df):
    """
    Processes minute-level data to create a feature-rich dataset.
    """
    df = df.drop(columns=['insertion_time', 'is_test'])
    df['mid_price'] = (df['bid_close'] + df['ask_close']) / 2
    df['spread'] = df['ask_close'] - df['bid_close']

    # Define windows appropriate for high-frequency minute data
    feature_windows = {
        'short_ma': 60, 'long_ma': 240, 'rsi': 120, 'vol': 252, 'lags': 5,
        'macd_fast': 48, 'macd_slow': 104, 'macd_sign': 36 # Scaled from common 12/26/9
    }
    
    df_processed = create_ml_dataset(df, 'mid_price', target_window=252, feature_windows=feature_windows)
    return df_processed

def create_daily_data(df):
    """
    Aggregates to daily level and creates a feature-rich dataset.
    """
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
    df['mid_price'] = (df['bid_close'] + df['ask_close']) / 2

    daily = df.groupby(['symbol', 'date']).agg(
        mean_price=('mid_price', 'mean'),
        max_price=('mid_price', 'max'),
        min_price=('mid_price', 'min'),
        last_price=('mid_price', 'last'),
        tag=('tag', 'first')
    ).reset_index()

    # Engineer daily-specific features
    daily['price_range'] = daily['max_price'] - daily['min_price']
    daily['timestamp'] = pd.to_datetime(daily['date']).view('int64') // 10**9

    # Define windows appropriate for daily data
    feature_windows = {
        'short_ma': 10, 'long_ma': 30, 'rsi': 14, 'vol': 21, 'lags': 3,
        'macd_fast': 12, 'macd_slow': 26, 'macd_sign': 9
    }
    
    daily_processed = create_ml_dataset(daily, 'mean_price', target_window=60, feature_windows=feature_windows)
    return daily_processed

if __name__ == "__main__":
    df_raw = load_data()
    
    # --- Generate new filenames for the feature-rich datasets ---
    minute_fname = "oister_minute_features.csv"
    daily_fname = "oister_daily_features.csv"

    print("Processing minute-level data with feature engineering...")
    df_minute_features = process_minute_data(df_raw.copy())
    df_minute_features.to_csv(minute_fname, index=False)
    print(f"Minute-level data saved to '{minute_fname}'. Shape: {df_minute_features.shape}")
    print(f"Features: {df_minute_features.columns.tolist()}")

    print("\nProcessing daily-level data with feature engineering...")
    daily_df_features = create_daily_data(df_raw.copy())
    daily_df_features.to_csv(daily_fname, index=False)
    print(f"Daily-level data saved to '{daily_fname}'. Shape: {daily_df_features.shape}")
    print(f"Features: {daily_df_features.columns.tolist()}")
