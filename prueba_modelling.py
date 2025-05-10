import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')


def preprocess_data(df):
    """Adds temporal features"""
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    return df


def time_series_split(df, test_size=0.2):
    """Splits the dataset into training and testing sets"""
    sorted_df = df.sort_values('date')
    split_idx = int(len(sorted_df) * (1 - test_size))
    return sorted_df.iloc[:split_idx], sorted_df.iloc[split_idx:]


def select_features(frequency):
    """Defines the feature set based on data frequency (minute or daily)."""
    return ['day_of_week', 'month', 'bid_close', 'ask_close'] if frequency == 'minute' else \
           ['mean_bid', 'max_bid', 'min_bid', 'last_bid']


def analyze_feature_importance(model, features, title, model_name):
    """Displays feature importance for the trained model."""
    
    if model_name == "IsolationForest":
        print(f"\nFeature Importance not available for {model_name}")
        return
        
    importance = pd.Series(model.feature_importances_, index=features)
    print(f"\nFeature Importance ({title} - {model_name}):")
    print(importance.sort_values(ascending=False))
    
    plt.figure(figsize=(10, 6))
    importance.sort_values().plot(kind='barh')
    plt.title(f'Feature Importance ({title} - {model_name})')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{title.lower()}_{model_name.lower()}.png')
    plt.close()


def optimize_model(X_train, y_train, freq, models_dict):
    """Return the best one."""
    best_models = {}
    best_scores = {}
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    for model_name, (model, param_dist) in models_dict.items():
        print(f"Optimizing {model_name} for {freq}-level data...")
        
        # Special handling for IsolationForest
        if model_name == "IsolationForest":
            model.fit(X_train)
            y_pred = model.predict(X_train)
            # In IsolationForest, -1 is anomaly, 1 is normal
            y_pred = np.where(y_pred == -1, 1, 0)  # Assuming class 1 is our anomaly class
            score = f1_score(y_train, y_pred, average='weighted')
            best_models[model_name] = model
            best_scores[model_name] = score
            print(f"F1 Score for {model_name} ({freq}): {score:.4f}")
            continue
        
        
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=5,  # reduced iterations due to hardware limitations
            cv=tscv,
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        search.fit(X_train, y_train)
        best_models[model_name] = search.best_estimator_
        best_scores[model_name] = search.best_score_
        
        print(f"\nBest Parameters ({model_name} - {freq}): {search.best_params_}")
        print(f"Best F1 Score ({model_name} - {freq}): {search.best_score_:.4f}")
    
    # Find the best model
    best_model_name = max(best_scores, key=best_scores.get)
    print(f"\nBest model for {freq}-level data: {best_model_name} with F1 score {best_scores[best_model_name]:.4f}")
    
    return best_models, best_model_name


def evaluate_models(models, X_test, y_test, label_encoder, freq):
    """Evaluate all models on test data."""
    results = {}
    
    # Use the new five directional classes:
    target_names = ['DOWN', 'MODERATE_DOWN', 'SAME', 'MODERATE_UP', 'UP']
    
    for model_name, model in models.items():
        if model_name == "IsolationForest":
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == -1, 1, 0)
        else:
            y_pred = model.predict(X_test)
        
        print(f"\n{model_name} Model Evaluation ({freq}):")
        report = classification_report(y_test, y_pred, 
                                       target_names=target_names,
                                       output_dict=True)
        print(classification_report(y_test, y_pred, 
                                    target_names=target_names))
        
        results[model_name] = {
            'f1_weighted': report['weighted avg']['f1-score'],
            'accuracy': report['accuracy']
        }
    
    return results


def plot_model_comparison(results, title):
    """Plot comparison of model performance."""
    df_results = pd.DataFrame(results).T
    
    plt.figure(figsize=(12, 6))
    df_results.plot(kind='bar')
    plt.title(f'Model Performance Comparison - {title}')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'model_comparison_{title.lower()}.png')
    plt.close()


def main():
    # Define models with minimal parameter grids (also because of hardware limitations)
    models = {
        "RandomForest": (
            RandomForestClassifier(
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1,
                criterion='entropy'
            ),
            {
                'n_estimators': [200],
                'max_depth': [None],
                'min_samples_leaf': [1],
                'min_samples_split': [2],
                'max_features': ['sqrt']
            }
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(
                random_state=42,
                loss='log_loss'
            ),
            {
                'n_estimators': [200],
                'learning_rate': [0.1],
                'subsample': [0.8],
                'max_depth': [3],
                'min_samples_leaf': [1],
                'max_features': ['sqrt']
            }
        ),
        "XGBoost": (
            XGBClassifier(
                eval_metric='mlogloss',
                random_state=42,
                objective='multi:softprob',
                tree_method='hist',
                num_class=5  # adjust num_class to 5
            ),
            {
                'n_estimators': [100],
                'max_depth': [3],
                'min_child_weight': [1],
                'gamma': [0],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
                'learning_rate': [0.1]
            }
        ),
        "LightGBM": (
            LGBMClassifier(
                random_state=42,
                objective='multiclass',
                class_weight='balanced',
                num_class=5,  # adjust num_class to 5
                verbose=-1,
                is_unbalance=True
            ),
            {
                'n_estimators': [100],
                'num_leaves': [31],
                'learning_rate': [0.1],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
                'reg_alpha': [0],
                'reg_lambda': [0],
                'min_child_samples': [5]
            }
        ),
        "IsolationForest": (
            IsolationForest(
                contamination=0.2,
                random_state=42,
                n_jobs=-1,
                max_samples='auto'
            ),
            {}  # No hyperparameter tuning for IsolationForest
        )
    }
    
    label_encoder = LabelEncoder()

    # Train models for minute-level data
    print("=" * 80)
    print("Training minute-level models...")
    print("=" * 80)
    minute_data = preprocess_data(pd.read_csv("oister_data.csv"))
    train_min, test_min = time_series_split(minute_data)

    features_min = select_features('minute')
    X_train_min = train_min[features_min]
    # Fit and transform the new five-level 'direction' column
    y_train_min = label_encoder.fit_transform(train_min['direction'])
    X_test_min = test_min[features_min]
    y_test_min = label_encoder.transform(test_min['direction'])

    best_models_min, best_model_name_min = optimize_model(X_train_min, y_train_min, 'minute', models)
    
    for model_name, model in best_models_min.items():
        analyze_feature_importance(model, features_min, 'Minute', model_name)

    results_min = evaluate_models(best_models_min, X_test_min, y_test_min, label_encoder, 'minute')
    plot_model_comparison(results_min, 'Minute-Level')
    
    joblib.dump(best_models_min[best_model_name_min], f'best_model_minute_{best_model_name_min.lower()}.pkl')
    for model_name, model in best_models_min.items():
        joblib.dump(model, f'model_minute_{model_name.lower()}.pkl')

    # Train models for daily-level data
    print("\n" + "=" * 80)
    print("Training daily-level models...")
    print("=" * 80)
    daily_data = preprocess_data(pd.read_csv("oister_daily_data.csv"))
    train_daily, test_daily = time_series_split(daily_data)

    features_daily = select_features('daily')
    X_train_daily = train_daily[features_daily]
    y_train_daily = label_encoder.fit_transform(train_daily['direction'])
    X_test_daily = test_daily[features_daily]
    y_test_daily = label_encoder.transform(test_daily['direction'])

    best_models_daily, best_model_name_daily = optimize_model(X_train_daily, y_train_daily, 'daily', models)
    
    for model_name, model in best_models_daily.items():
        analyze_feature_importance(model, features_daily, 'Daily', model_name)

    results_daily = evaluate_models(best_models_daily, X_test_daily, y_test_daily, label_encoder, 'daily')
    plot_model_comparison(results_daily, 'Daily-Level')
    
    joblib.dump(best_models_daily[best_model_name_daily], f'best_model_daily_{best_model_name_daily.lower()}.pkl')
    
    
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)
    print(f"Best minute-level model: {best_model_name_min} (F1 Score: {results_min[best_model_name_min]['f1_weighted']:.4f})")
    print(f"Best daily-level model: {best_model_name_daily} (F1 Score: {results_daily[best_model_name_daily]['f1_weighted']:.4f})")


if __name__ == "__main__":
    main()

