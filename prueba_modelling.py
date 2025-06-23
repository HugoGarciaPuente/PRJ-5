import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings('ignore')

def time_series_split(df, test_size=0.2):
    """Splits the dataset into training and testing sets based on time."""
    # Assumes df is already sorted by time
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]

def analyze_feature_importance(model, features, title, model_name):
    """Displays and saves feature importance charts."""
    if not hasattr(model, 'feature_importances_'):
        print(f"No feature importance for {model_name}")
        return

    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(20)
    print(f"\nFeature importance ({title}, {model_name}):\n", imp)

    plt.figure(figsize=(10, 8))
    imp.plot(kind='barh')
    plt.title(f"Top 20 Features: {title} - {model_name}")
    plt.tight_layout()
    plt.savefig(f"fi_{title.lower()}_{model_name.lower()}.png")
    plt.close()

def optimize_model(X_train, y_train, freq, models_dict):
    """Tunes models via time-series CV and returns best estimators."""
    best, scores = {}, {}
    tscv = TimeSeriesSplit(n_splits=5)

    for name, (mdl, params) in models_dict.items():
        print(f"Optimizing {name} ({freq})...")
        search = RandomizedSearchCV(
            mdl, params, n_iter=10, cv=tscv, scoring='f1_weighted', n_jobs=-1, random_state=42, verbose=1
        )
        search.fit(X_train, y_train)
        best[name] = search.best_estimator_
        scores[name] = search.best_score_
        print(f"Best {name} params: {search.best_params_}")
        print(f"Best F1 (validation) for {name}: {scores[name]:.4f}\n")

    best_name = max(scores, key=scores.get)
    print(f"==> Best model for {freq}: {best_name} (F1={scores[best_name]:.4f})\n")
    return best, best_name

def evaluate_models(models, X_test, y_test, label_encoder, freq):
    """Runs classification report and returns metrics dict."""
    res = {}
    classes = label_encoder.classes_
    print("--- Test Set Evaluation ---")
    for name, mdl in models.items():
        preds = mdl.predict(X_test)
        print(f"\n{name} evaluation ({freq}):")
        print(classification_report(y_test, preds, target_names=classes))
        rpt = classification_report(y_test, preds, output_dict=True)
        res[name] = {'f1_weighted': rpt['weighted avg']['f1-score'], 'accuracy': rpt['accuracy']}
    return res

def plot_comparison(results, title):
    """Plots and saves model comparison."""
    df = pd.DataFrame(results).T
    ax = df.plot(kind='bar', figsize=(10,6), rot=0)
    ax.set_title(f"Model Comparison on Test Set: {title}")
    ax.set_ylabel('Score')
    plt.tight_layout()
    plt.savefig(f"cmp_{title.lower().replace(' ', '_')}.png")
    plt.close()

def main():
    models = {
        'RandomForest': (RandomForestClassifier(class_weight='balanced_subsample', random_state=42),
                         {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_leaf': [5, 10]}),
        
        # --- XGBoost: Set verbosity to 0 to hide training info ---
        'XGBoost': (XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=42, verbosity=0),
                    {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [5, 7], 'subsample': [0.7], 'colsample_bytree': [0.7]}),
        
        # --- LightGBM: Set verbose to -1 to hide training info ---
        'LightGBM': (LGBMClassifier(objective='multiclass', class_weight='balanced', random_state=42, verbose=-1),
                     {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'num_leaves': [20, 40], 'max_depth': [7, 10]})
    }

    le = LabelEncoder()
    
    # Use the feature-rich data files
    for freq, fname in [('Minute', 'oister_minute_features.csv'), ('Daily', 'oister_daily_features.csv')]:
        print('='*80)
        print(f"Processing {freq} data from {fname}")
        
        try:
            df = pd.read_csv(fname)
        except FileNotFoundError:
            print(f"ERROR: The required file '{fname}' was not found.")
            print("Please run the updated data generation script first.")
            continue

        df = df.sort_values('timestamp')
        df = pd.get_dummies(df, columns=['symbol'], prefix='symbol')
        
        train, test = time_series_split(df)
        
        y_train = le.fit_transform(train['direction'])
        y_test = le.transform(test['direction'])
        
        non_feature_cols = ['direction', 'date', 'tag']
        feature_names = [col for col in train.columns if col not in non_feature_cols]

        X_train = train[feature_names]
        X_test = test[feature_names]
        X_test = X_test[X_train.columns]

        print(f"Training with {len(X_train.columns)} features on {len(X_train)} samples.")
        print(f"Testing with {len(X_test.columns)} features on {len(X_test)} samples.")

        best_models, best_name = optimize_model(X_train, y_train, freq, models)
        for nm, m in best_models.items():
            analyze_feature_importance(m, X_train.columns, freq, nm)
        
        results = evaluate_models(best_models, X_test, y_test, le, freq)
        plot_comparison(results, f"{freq} Data")
        
        joblib.dump(best_models[best_name], f"best_{freq.lower()}_{best_name}.pkl")
        print(f"Saved best model for {freq}: {best_name}\n")

if __name__ == '__main__':
    main()
