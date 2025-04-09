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

def calculate_direction(df, column_name, threshold):
    """Calculates the directional movement (UP, DOWN, SAME) based on a percentage threshold."""
    df = df.sort_values(['symbol', 'timestamp'])
    
    # Calculate percentage changes
    df['next_close'] = df.groupby('symbol')[column_name].shift(-1)
    df['change_pct'] = (df['next_close'] - df[column_name]) / df[column_name]

    # Assign directions based on the threshold
    conditions = [
        df['change_pct'] > threshold,
        df['change_pct'] < -threshold
    ]
    choices = ['UP', 'DOWN']
    
    df['direction'] = np.select(conditions, choices, default='SAME')
    return df.drop(columns=['next_close', 'change_pct'])

def process_minute_data(df):
    """Processes minute-level data to calculate direction and clean up the dataset."""
    df = df.drop(columns=['insertion_time', 'is_test']) 
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date

    # Calculate threshold for minute data
    threshold = df.groupby('symbol')['bid_close'].pct_change().abs().mean()
    df = calculate_direction(df, 'bid_close', threshold)

    return df, threshold

def create_daily_data(df):
    """Aggregates minute-level data into daily-level data."""
    daily = df.groupby(['symbol', 'date']).agg({
        'bid_close': ['mean', 'max', 'min', 'last'],
        'ask_close': ['mean', 'max', 'min'],
        'tag': 'first'
    }).reset_index()

    # Rename columns for better clarity
    daily.columns = [
        'symbol', 'date',
        'mean_bid', 'max_bid', 'min_bid', 'last_bid',
        'mean_ask', 'max_ask', 'min_ask',
        'tag'
    ]

    # Calculate daily threshold
    daily_threshold = daily.groupby('symbol')['last_bid'].pct_change().abs().mean()

    # Add daily timestamp
    daily['timestamp'] = pd.to_datetime(daily['date']).astype('int64') // 10**9

    # Calculate daily direction
    daily = calculate_direction(daily, 'last_bid', daily_threshold)

    return daily, daily_threshold

   
if __name__ == "__main__":
    # Process minute data
    df, minute_threshold = process_minute_data(load_data())
    df.to_csv("oister_data.csv", index=False)
    print(f"Minute-level data saved. Threshold: {minute_threshold:.6f}")

    # Process daily data
    daily_df, daily_threshold = create_daily_data(df)
    daily_df.to_csv("oister_daily_data.csv", index=False)
    print(f"Daily-level data saved. Threshold: {daily_threshold:.6f}")

