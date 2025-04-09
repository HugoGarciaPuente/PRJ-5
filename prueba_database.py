import sqlite3 as sq
import pandas as pd
import numpy as np
import os
import json

def load_data():
    """Loads data from the SQLite database."""
    with sq.connect("oister.db") as conn:
        query = """
            SELECT stonks.*, symbol_types.tag 
            FROM stonks
            JOIN symbol_types USING (symbol)
        """
        return pd.read_sql(query, conn)

def calculate_direction(df, column_name, low_threshold, high_threshold):
    """
    Calculates granular directional movement based on two thresholds:
    'DOWN' if the change is below -high_threshold,
    'MODERATE_DOWN' if between -high_threshold and -low_threshold,
    'SAME' if within the interval (-low_threshold, low_threshold),
    'MODERATE_UP' if between low_threshold and high_threshold,
    'UP' if above high_threshold.
    """
    # Ensure the data is ordered by symbol and time.
    df = df.sort_values(['symbol', 'timestamp'])
    
    # Calculate percentage change compared to the next close
    df['next_close'] = df.groupby('symbol')[column_name].shift(-1)
    df['change_pct'] = (df['next_close'] - df[column_name]) / df[column_name]
    
    # Define the conditions
    conditions = [
        df['change_pct'] <= -high_threshold,  # Strong downward movement
        (df['change_pct'] > -high_threshold) & (df['change_pct'] <= -low_threshold),  # Moderate downward
        (df['change_pct'] > -low_threshold) & (df['change_pct'] < low_threshold),   # No significant change
        (df['change_pct'] >= low_threshold) & (df['change_pct'] < high_threshold),  # Moderate upward
        df['change_pct'] >= high_threshold   # Strong upward movement
    ]
    choices = ['DOWN', 'MODERATE_DOWN', 'SAME', 'MODERATE_UP', 'UP']
    
    # Use np.select to assign the corresponding labels
    df['direction'] = np.select(conditions, choices, default='SAME')
    
    # Drop the temporary columns used for calculation
    return df.drop(columns=['next_close', 'change_pct'])

def process_minute_data(df):
    """Processes minute-level data to calculate direction and clean up the dataset."""
    df = df.drop(columns=['insertion_time', 'is_test']) 
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date

    # Calculate a baseline threshold based on the mean absolute percentage change per symbol
    base_threshold = df.groupby('symbol')['bid_close'].pct_change().abs().mean()
    # Here we use the base threshold as the low threshold and twice that as the high threshold.
    low_threshold = base_threshold
    high_threshold = base_threshold * 2

    df = calculate_direction(df, 'bid_close', low_threshold, high_threshold)

    return df, (low_threshold, high_threshold)

def create_daily_data(df):
    """Aggregates minute-level data into daily-level data."""
    daily = df.groupby(['symbol', 'date']).agg({
        'bid_close': ['mean', 'max', 'min', 'last'],
        'ask_close': ['mean', 'max', 'min'],
        'tag': 'first'
    }).reset_index()

    # Rename columns for clarity
    daily.columns = [
        'symbol', 'date',
        'mean_bid', 'max_bid', 'min_bid', 'last_bid',
        'mean_ask', 'max_ask', 'min_ask',
        'tag'
    ]

    # Calculate the baseline threshold for daily data
    base_threshold = daily.groupby('symbol')['last_bid'].pct_change().abs().mean()
    low_threshold = base_threshold
    high_threshold = base_threshold * 2

    # Add daily timestamp (convert date to POSIX timestamp)
    daily['timestamp'] = pd.to_datetime(daily['date']).astype('int64') // 10**9

    # Use our granular calculate_direction for daily data
    daily = calculate_direction(daily, 'last_bid', low_threshold, high_threshold)

    return daily, (low_threshold, high_threshold)

if __name__ == "__main__":
    # Process minute data
    df, (minute_low, minute_high) = process_minute_data(load_data())
    df.to_csv("oister_data.csv", index=False)
    print(f"Minute-level data saved. Thresholds - Low: {minute_low:.6f}, High: {minute_high:.6f}")

    # Process daily data
    daily_df, (daily_low, daily_high) = create_daily_data(df)
    daily_df.to_csv("oister_daily_data.csv", index=False)
    print(f"Daily-level data saved. Thresholds - Low: {daily_low:.6f}, High: {daily_high:.6f}")
