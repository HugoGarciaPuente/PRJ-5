# Financial Time Series Forecasting

This repository contains a machine learning project aimed at predicting the directional movement of stock prices. The project has evolved significantly from its initial version to address poor model performance and create a more robust and insightful workflow.

## Project Evolution & Rationale

The project has undergone several key changes to improve model performance and clarity:

1.  **Simplified Target Variable**: The initial model used a granular 4-class system. This proved too complex for the models, resulting in poor performance. The project has been refactored to predict a simpler, more distinct 3-class target:
    *   **FALL**: A significant downward price movement.
    *   **NEUTRAL**: A stable price movement or insignificant change.
    *   **RISE**: A significant upward price movement.
    *   **Reasoning**: This simplification makes the classification task clearer and helps the models learn more distinct patterns, aiming for higher predictive accuracy on significant movements.

2.  **Centralized Feature Engineering**: All feature creation logic has been moved from the modeling script into a dedicated data preparation script (`database_creation.py`).
    *   **Reasoning**: This creates a single source of truth for features, ensures consistency, and separates the concerns of data preparation from modeling. It allows for a richer and more complex set of features to be engineered once and used efficiently.

3.  **Enhanced Feature Set**: The project now incorporates a wide array of technical indicators and momentum-based features.
    *   **Reasoning**: To provide the models with more context than just raw price data, features like RSI, MACD, moving average ratios, and historical volatility have been added. This aims to capture market trends, momentum, and risk.

## Project Workflow

The project is structured into two main scripts:

### 1. Data Preparation & Feature Engineering (`database_creation.py`)

This script is responsible for loading the raw price data and creating two feature-rich datasets (`oister_minute_features.csv` and `oister_daily_features.csv`) ready for modeling. Its key tasks include:
- Loading raw data from the database.
- Engineering a comprehensive feature set, including:
  - **Momentum Indicators**: Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD).
  - **Trend Indicators**: Ratio of short-term to long-term Moving Averages (MA Ratio).
  - **Volatility Measures**: Rolling standard deviation of returns.
  - **Lagged Features**: Price returns from previous time steps.
- Calculating the 3-class target variable (`FALL`, `NEUTRAL`, `RISE`) based on future price movements.
- Saving the final, processed datasets to CSV files.

### 2. Model Training & Evaluation (`modelling.py`)

This script takes the prepared datasets and runs the entire modeling pipeline:
- **Loads** the feature-rich CSV files.
- **Splits** the data into training and testing sets using a time-series-aware approach to prevent data leakage.
- **Performs Hyperparameter Tuning** on multiple models (RandomForest, XGBoost, LightGBM) using `RandomizedSearchCV` with time-series cross-validation.
- **Evaluates** the best models on the test set, reporting key metrics like F1-score, precision, and recall.
- **Analyzes Feature Importance** to understand which market signals are most influential.
- **Saves** the best-performing model for both minute and daily frequencies using `joblib`.

## Explored Avenues & Challenges

-   **Sentiment Analysis**: An attempt was made to incorporate sentiment from financial news and Twitter. However, this was abandoned due to a lack of comprehensive datasets that matched the project's specific companies and historical timeframes.

## Current Status & Results

Despite the significant feature engineering and simplification of the problem, model performance remains low. The results indicate that the technical indicators derived from price data alone may not be sufficient to consistently predict market movements in this complex environment. Further research into alternative data sources or more advanced modeling techniques (e.g., deep learning for time series) may be required.


## Requirements
- Python 3.7+
- pandas
- matplotlib
- numpy
- scikit-learn
- joblib
- xgboost
- lightgbm

  
![cmp_minute_data](https://github.com/user-attachments/assets/cc2ede9f-6211-4e28-b899-db8df2225ded)
![cmp_daily_data](https://github.com/user-attachments/assets/a3116b96-b7e1-4206-86c7-fb9de797c773)
![fi_minute_randomforest](https://github.com/user-attachments/assets/491521e6-1800-4e5f-9c23-f9d8e53bcd8e)
![fi_minute_lightgbm](https://github.com/user-attachments/assets/0f3df6c7-987f-4c8c-b877-9b24a3c8a3b8)
![fi_daily_xgboost](https://github.com/user-attachments/assets/54b757d6-d0f2-4c70-b180-c905c4640258)
![fi_daily_randomforest](https://github.com/user-attachments/assets/8a847011-a770-474a-8a3f-8dcd9ef42e9a)
![fi_daily_lightgbm](https://github.com/user-attachments/assets/a824031c-c5a8-49f7-a5de-825e96f9e66c)
![fi_minute_xgboost](https://github.com/user-attachments/assets/78ad47cb-2eef-4c01-a63c-08679681d3fd)









