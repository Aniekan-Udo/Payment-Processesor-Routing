# Payment Processor Routing Model

This repository houses a machine learning pipeline designed to intelligently route payment transactions to the optimal processor. By leveraging historical data, the model predicts the **lowest possible transaction latency** for various processors, enabling efficient and fast payment processing.


## Features

This project streamlines the process of training and deploying a latency prediction model with the following key features:

* **Automated Data Preparation**: Loads raw transaction data, performs essential feature engineering (e.g., extracting `day_of_week`, `hour_of_day`, and `time_of_day` from timestamps), and handles column drops for irrelevant features.

* **Robust Data Preprocessing**:
    * Splits data into training and testing sets.
    * Applies `log1p` transformation to the `latency_ms` target variable to normalize its distribution, improving model performance. Predictions are then inverse-transformed (`expm1`) back to the original scale.
    * Utilizes `StandardScaler` for numerical features and `OneHotEncoder` for categorical features to prepare data for model consumption.

* **Advanced Model Training**: Employs a **Random Forest Regressor** and leverages `GridSearchCV` for comprehensive hyperparameter tuning, ensuring the selection of the best-performing model.

* **Comprehensive Model Evaluation**: Assesses model accuracy using industry-standard regression metrics: **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R-squared ($R^2$)**. Visualizes performance with residual plots and true vs. predicted latency graphs.

* **Persistent Model Artifacts**: Securely saves the trained model, feature preprocessor, and other critical metadata (like column names and data types) using `joblib` for seamless future loading and inference.

* **Intelligent Prediction Engine**: Provides a dedicated function to predict the best payment processor for new, unseen transactions by estimating latency for each available processor and recommending the one with the lowest predicted value.

