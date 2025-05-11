import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


def load_data(minute_path: str, daily_path: str):
    df_min = pd.read_csv(minute_path)
    df_day = pd.read_csv(daily_path)
    return df_min, df_day


def preprocess(df_min: pd.DataFrame, df_day: pd.DataFrame):
    df_min['datetime'] = pd.to_datetime(df_min['timestamp'], unit='s')
    df_min.set_index('datetime', inplace=True)
    df_min.sort_index(inplace=True)

    df_day['datetime'] = pd.to_datetime(df_day['timestamp'], unit='s')
    df_day.set_index('datetime', inplace=True)
    df_day.sort_index(inplace=True)
    return df_min, df_day


def detect_anomalies_zscore(series: pd.Series, threshold: float = 3.0):
    z = np.abs(stats.zscore(series.dropna()))
    return series.index[z > threshold]


def detect_anomalies_iqr(series: pd.Series, factor: float = 1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - factor * iqr, q3 + factor * iqr
    return series.index[(series < low) | (series > high)]


def detect_anomalies_iso(df: pd.DataFrame, cols: list, contamination: float):
    iso = IsolationForest(contamination=contamination, random_state=42)
    data = df[cols].dropna()
    iso.fit(data)
    preds = iso.predict(data)
    return data.index[preds == -1]


def analyze_tag(df_min: pd.DataFrame, df_day: pd.DataFrame, tag: str):
    # Filter by company tag
    m = df_min[df_min['tag'] == tag].copy()
    d = df_day[df_day['tag'] == tag].copy()
    # Compute spreads
    m['bid_spread'] = m['bid_max'] - m['bid_min']
    m['ask_spread'] = m['ask_max'] - m['ask_min']

    # Detect anomalies
    methods = {
        'Z-Score': detect_anomalies_zscore(m['bid_spread']),
        'IQR': detect_anomalies_iqr(m['ask_spread']),
        'IsolationForest': detect_anomalies_iso(m, ['bid_spread', 'ask_spread'], 0.01)
    }

    # Prepare daily anomaly counts
    all_idx = np.concatenate(list(methods.values()))
    counts = pd.Series(1, index=all_idx)
    daily_counts = counts.resample('D').sum().fillna(0)

    # Create multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: price time-series
    ax = axes[0, 0]
    ax.plot(m.index, m['bid_min'], label='bid_min', alpha=0.6)
    ax.plot(m.index, m['bid_max'], label='bid_max', alpha=0.6)
    ax.plot(m.index, m['ask_min'], label='ask_min', alpha=0.6)
    ax.plot(m.index, m['ask_max'], label='ask_max', alpha=0.6)
    ax.set_title(f'{tag} Prices (minute)')
    ax.legend(fontsize='small')

    # Top-right: spreads with anomalies overlay
    ax = axes[0, 1]
    ax.plot(m.index, m['bid_spread'], label='Bid Spread', alpha=0.8)
    ax.plot(m.index, m['ask_spread'], label='Ask Spread', alpha=0.8)
    colors = {'Z-Score': 'red', 'IQR': 'orange', 'IsolationForest': 'purple'}
    markers = {'Z-Score': 'o', 'IQR': 's', 'IsolationForest': 'x'}
    for label, idx in methods.items():
        ax.scatter(idx, m.loc[idx, 'bid_spread'], color=colors[label], marker=markers[label], s=20, label=f'{label} (Bid)')
        ax.scatter(idx, m.loc[idx, 'ask_spread'], color=colors[label], marker=markers[label], s=20, label=f'{label} (Ask)', alpha=0.7)
    # dedupe legend
    handles, labels_ = ax.get_legend_handles_labels()
    unique = dict(zip(labels_, handles))
    ax.legend(unique.values(), unique.keys(), fontsize='small')
    ax.set_title(f'{tag} Spreads & Anomalies')

    # Bottom-left: daily metrics
    ax = axes[1, 0]
    ax.plot(d.index, d['mean_bid'], label='mean_bid')
    ax.plot(d.index, d['max_bid'], label='max_bid')
    ax.plot(d.index, d['min_bid'], label='min_bid')
    ax.plot(d.index, d['mean_ask'], label='mean_ask')
    ax.plot(d.index, d['max_ask'], label='max_ask')
    ax.plot(d.index, d['min_ask'], label='min_ask')
    ax.set_title(f'{tag} Daily Stats')
    ax.legend(fontsize='small')

    # Bottom-right: daily anomaly counts
    ax = axes[1, 1]
    ax.bar(daily_counts.index, daily_counts.values, width=0.8, color='steelblue', edgecolor='steelblue')
    ax.set_title(f'{tag} Daily Anomaly Counts')
    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    for label in ax.get_xticklabels():
        label.set_rotation(45)

    plt.tight_layout()
    fname = f'{tag}_summary.png'
    plt.savefig(fname)
    plt.close()
    print(f"[{tag}] Saved combined summary figure to {fname}")

    # Save anomaly timestamps to CSV
    records = []
    for label, idx in methods.items():
        for ts in idx:
            records.append({'tag': tag, 'method': label, 'timestamp': ts})
    pd.DataFrame(records).to_csv(f'{tag}_anomalies.csv', index=False)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--minute_csv', default='oister_data.csv')
    p.add_argument('--daily_csv', default='oister_daily_data.csv')
    args = p.parse_args()
    df_min, df_day = load_data(args.minute_csv, args.daily_csv)
    df_min, df_day = preprocess(df_min, df_day)
    for tag in sorted(df_min['tag'].unique()):
        analyze_tag(df_min, df_day, tag)

