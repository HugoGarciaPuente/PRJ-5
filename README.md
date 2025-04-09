# Directional Movement Classification & Model Training

This repository contains code to perform time-series prediction on financial data using multiple machine learning models. The code has been updated to handle new databases with enhanced directional levels. Instead of simply classifying price movements as `DOWN`, `SAME`, or `UP`, the updated code supports five distinct classes:
- **DOWN**
- **MODERATE_DOWN**
- **SAME**
- **MODERATE_UP**
- **UP**

## Overview

The project performs the following steps:

1. **Data Preprocessing:**  
   - Loads the dataset(s) from CSV files.
   - Adds temporal features like day of the week and month.
   - Splits the data into training and testing sets based on time series order.

2. **Feature Selection:**  
   - Selects different feature sets for minute-level data and daily-level data.
   - Uses features such as bid/ask prices and aggregated daily statistics.

3. **Model Optimization and Training:**  
   - Uses a set of machine learning models (RandomForest, GradientBoosting, XGBoost, LightGBM, and IsolationForest).
   - Performs hyperparameter tuning using `RandomizedSearchCV` with a time-series cross-validation strategy.
   - Optimizes and selects the best model based on F1 score.

4. **Feature Importance Analysis:**  
   - Analyzes and plots the feature importances for models that support this metric.

5. **Evaluation and Comparison:**  
   - Evaluates models against the test dataset.
   - Adjusts the evaluation to work with the five new directional classes.
   - Plots a comparison chart of the model performances.

6. **Model Saving:**  
   - Serializes (using `joblib`) the best models from both minute-level and daily-level data to disk.

## Requirements

- Python 3.7+
- Libraries:
  - pandas
  - matplotlib
  - numpy
  - scikit-learn
  - joblib
  - xgboost
  - lightgbm
  - warnings (builtin)
  
You can install the required packages using `pip`:

```bash
pip install pandas matplotlib numpy scikit-learn joblib xgboost lightgbm









![model_comparison_minute-level](https://github.com/user-attachments/assets/5ce33ff2-6d7c-417a-a881-221e885e9349)
![model_comparison_daily-level](https://github.com/user-attachments/assets/40328647-710e-4bce-baab-07bffc59bfb4)
![feature_importance_minute_xgboost](https://github.com/user-attachments/assets/4f7587a2-888b-453e-90ad-1c6e02efaae8)
![feature_importance_minute_randomforest](https://github.com/user-attachments/assets/c49f7bb7-15fe-419b-9275-375783b1b5ef)
![feature_importance_minute_lightgbm](https://github.com/user-attachments/assets/65bb83b4-f551-49d0-990c-953addc472a9)
![feature_importance_minute_gradientboosting](https://github.com/user-attachments/assets/fc471a8d-169a-43b4-b778-48c3b08f00c9)
![feature_importance_daily_xgboost](https://github.com/user-attachments/assets/38b46873-52e8-4af1-8f91-2c03d68022dc)
![feature_importance_daily_randomforest](https://github.com/user-attachments/assets/2f58fa18-4277-4265-b6d9-72f561a674d1)
![feature_importance_daily_lightgbm](https://github.com/user-attachments/assets/8bb9eaef-8d23-4280-8144-c7748c620aeb)
![feature_importance_daily_gradientboosting](https://github.com/user-attachments/assets/e59bfd5c-60e5-41c7-af90-fb7c176c5ce0)
