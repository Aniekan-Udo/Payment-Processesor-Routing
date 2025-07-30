import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
import logging 
import os 
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, make_scorer

# --- Configuration ---
DATA_FILE = "processor_data.csv"
MODEL_DIR = "model_artifacts"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
TARGET_METRIC = 'neg_mean_absolute_error'

# Hyperparameter grid for RandomForestRegressor
PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [10, 20, None],
    'min_samples_leaf': [1, 2, 4]
}

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)

def load_data_and_engineer_features(file_path):
    """
    Loads the dataset and performs feature engineering.
    """
    logging.info(f"Attempting to load data from: {file_path}")
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Error: The file '{file_path}' was not found.")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading the data: {e}")
        raise

    logging.info("Starting feature engineering...")
    data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.day_name()
    data['hour_of_day'] = pd.to_datetime(data['timestamp']).dt.hour

    bins = [0, 6, 12, 18, 24]
    labels = ['Late Night', 'Morning', 'Afternoon', 'Evening']
    data['time_of_day'] = pd.cut(data['hour_of_day'], bins=bins, labels=labels, right=False, include_lowest=True)

    
    # Make sure these dropped columns are consistent with X_train_columns for app.py
    df = data.drop(columns=["transaction_id", "error_code", "hour_of_day", "timestamp", "country", "currency"], axis=1)
    logging.info("Dropped unnecessary columns including 'country' and 'currency'.")
    return df

def preprocess_data(X, y):
    """
    Splits data, identifies column types, and creates preprocessors.
    Applies log1p transformation to the target variable (y).
    """
    logging.info("Performing Train-Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE) 
    logging.info(f"Train-Test split complete. Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Apply log1p transformation to target variable
    logging.info("Applying log1p transformation to target variable (latency_ms)...")
    y_train_transformed = np.log1p(y_train)
    y_test_transformed = np.log1p(y_test)
    logging.info("Target variable transformed successfully.")
    

    cat_cols = [col for col in X_train.columns if X_train[col].dtype == 'object' or isinstance(X_train[col].dtype, pd.CategoricalDtype)]
    num_cols = [col for col in X_train.columns if pd.api.types.is_numeric_dtype(X_train[col])]

    logging.info(f"Numerical columns for scaling: {num_cols}")
    logging.info(f"Categorical columns for OneHot Encoding: {cat_cols}")

    logging.info("Creating feature preprocessor (StandardScaler and OneHotEncoder)...")
    feature_preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), num_cols),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='drop'
    )

    logging.info("Applying feature transformation to training and test sets...")
    X_train_transformed = feature_preprocessor.fit_transform(X_train)
    X_test_transformed = feature_preprocessor.transform(X_test)
    logging.info(f"X_train_transformed shape: {X_train_transformed.shape}")
    logging.info(f"X_test_transformed shape: {X_test_transformed.shape}")

    return X_train_transformed, X_test_transformed, y_train_transformed, y_test_transformed, \
           feature_preprocessor, X_train.columns, X_train.dtypes

def train_and_evaluate_model(X_train_transformed, y_train_transformed, X_test_transformed, y_test_transformed, y_original_test):
    """
    Trains the RandomForestRegressor with GridSearchCV and evaluates its performance.
    Predictions are inverse transformed before evaluation.
    """
    logging.info("Setting up RandomForestRegressor for hyperparameter tuning...")
    model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    grid_search = GridSearchCV(estimator=model,
                               param_grid=PARAM_GRID,
                               cv=CV_FOLDS,
                               scoring=TARGET_METRIC,
                               verbose=2,
                               n_jobs=-1)

    logging.info("Starting Grid Search for best model parameters...")
    try:
        grid_search.fit(X_train_transformed, y_train_transformed)
        best_model = grid_search.best_estimator_
        logging.info(f"Grid Search completed. Best parameters: {grid_search.best_params_}")
        logging.info(f"Best cross-validation score ({TARGET_METRIC}): {grid_search.best_score_:.4f}")
    except Exception as e:
        logging.error(f"An error occurred during Grid Search: {e}")
        raise

    logging.info("Evaluating the best model...")
    y_pred_transformed = best_model.predict(X_test_transformed)
    
    #Inverse transform predictions for evaluation
    y_pred = np.expm1(y_pred_transformed)
    logging.info("Predictions inverse transformed for evaluation.")
    

    # Regression Metrics (using original y_test and inverse-transformed y_pred)
    mae = mean_absolute_error(y_original_test, y_pred) # Use y_original_test here
    rmse = root_mean_squared_error(y_original_test, y_pred) # Use y_original_test here
    r2 = r2_score(y_original_test, y_pred) # Use y_original_test here

    logging.info(f"### Mean Absolute Error (MAE) of Best Model: {mae:.4f} ###")
    logging.info(f"### Root Mean Squared Error (RMSE) of Best Model: {rmse:.4f} ###")
    logging.info(f"### R-squared (R2) of Best Model: {r2:.4f} ###")

    # Residual Plot
    residuals = y_original_test - y_pred # Using original y_test for residuals
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Latency (Original Scale)')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title('Residual Plot (After Target Transformation)')
    os.makedirs('plots', exist_ok=True)
    plt.savefig(os.path.join('plots', 'residual_plot_transformed_target.png')) 
    logging.info("Residual plot (transformed target) saved to plots/residual_plot_transformed_target.png")
    plt.close()

    # Predicted vs True Values Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_original_test, y=y_pred, alpha=0.6) 
    plt.plot([y_original_test.min(), y_original_test.max()], [y_original_test.min(), y_original_test.max()], 'r--', lw=2)
    plt.xlabel('True Latency (Original Scale)')
    plt.ylabel('Predicted Latency (Original Scale)')
    plt.title('True vs. Predicted Latency (After Target Transformation)')
    plt.grid(True)
    plt.savefig(os.path.join('plots', 'true_vs_predicted_transformed_target.png')) 
    logging.info("True vs. Predicted Latency plot (transformed target) saved to plots/true_vs_predicted_transformed_target.png")
    plt.close()

    return best_model

def save_model_artifacts(model, feature_preprocessor, X_train_columns, X_train_dtypes, model_dir):
    """
    Saves the trained model and preprocessors to disk.
    """
    logging.info(f"Saving model artifacts to {model_dir}...")
    os.makedirs(model_dir, exist_ok=True)

    try:
        joblib.dump(model, os.path.join(model_dir, 'random_forest_regressor_model.joblib'))
        joblib.dump(feature_preprocessor, os.path.join(model_dir, 'feature_preprocessor.joblib'))
        joblib.dump(X_train_columns, os.path.join(model_dir, 'X_train_columns.joblib'))
        joblib.dump(X_train_dtypes, os.path.join(model_dir, 'X_train_dtypes.joblib'))
        logging.info("Model, feature preprocessor, columns, and dtypes saved successfully.")
    except Exception as e:
        logging.error(f"Error saving model artifacts: {e}")
        raise

def load_model_artifacts(model_dir):
    """
    Loads the trained model and preprocessors from disk.
    """
    logging.info(f"Loading model artifacts from {model_dir}...")
    try:
        loaded_model = joblib.load(os.path.join(model_dir, 'random_forest_regressor_model.joblib'))
        loaded_feature_preprocessor = joblib.load(os.path.join(model_dir, 'feature_preprocessor.joblib'))
        loaded_X_train_columns = joblib.load(os.path.join(model_dir, 'X_train_columns.joblib'))
        loaded_X_train_dtypes = joblib.load(os.path.join(model_dir, 'X_train_dtypes.joblib'))
        logging.info("Model artifacts loaded successfully.")
        return loaded_model, loaded_feature_preprocessor, \
               loaded_X_train_columns, loaded_X_train_dtypes
    except FileNotFoundError:
        logging.error(f"Error: Model artifacts not found in '{model_dir}'. Please ensure they are trained and saved.")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading model artifacts: {e}")
        raise

def predict_best_processor(new_transaction_data, model, feature_preprocessor, all_processors, X_train_columns, X_train_dtypes):
    """
    Predicts the best processor for a new transaction based on the *lowest predicted latency*.
    Predictions are inverse transformed back to the original scale.
    """
    processor_latencies = {}

    transaction_row = pd.DataFrame(index=[0], columns=X_train_columns)
    for col in X_train_columns:
        if isinstance(X_train_dtypes[col], pd.CategoricalDtype):
            transaction_row[col] = pd.Series([np.nan], dtype=pd.CategoricalDtype(
                categories=X_train_dtypes[col].categories, ordered=X_train_dtypes[col].ordered
            ))
        elif pd.api.types.is_numeric_dtype(X_train_dtypes[col]):
            transaction_row[col] = pd.Series([np.nan], dtype=float)
        else:
            transaction_row[col] = pd.Series([np.nan], dtype=X_train_dtypes[col])

    for key, value in new_transaction_data.items():
        if key in transaction_row.columns:
            try:
                transaction_row.loc[0, key] = value
            except Exception as e:
                logging.warning(f"Could not assign value '{value}' to column '{key}': {e}. Setting to NaN.")
                transaction_row.loc[0, key] = np.nan
        else:
            logging.warning(f"New transaction data contains unknown column: {key}. It will be ignored.")

    for processor_name in all_processors:
        transaction_df_for_processor = transaction_row.copy()
        transaction_df_for_processor.loc[0, 'processor'] = processor_name 

        transaction_df_for_processor = transaction_df_for_processor[X_train_columns]

        logging.debug(f"DataFrame for {processor_name} before transform:\n{transaction_df_for_processor}")
        logging.debug(f"Shape of DataFrame for {processor_name} before transform: {transaction_df_for_processor.shape}")
        logging.debug(f"Dtypes for {processor_name} before transform:\n{transaction_df_for_processor.dtypes}")
        logging.debug(f"Is NaN present in DataFrame for {processor_name} before transform? {transaction_df_for_processor.isnull().any().any()}")

        try:
            transformed_transaction = feature_preprocessor.transform(transaction_df_for_processor)
            # Predict in the transformed scale
            predicted_latency_transformed = model.predict(transformed_transaction)[0]
            # Inverse transform to get original latency scale
            predicted_latency = np.expm1(predicted_latency_transformed) 
            processor_latencies[processor_name] = predicted_latency
            logging.debug(f"Processor: {processor_name}, Predicted Latency: {predicted_latency:.4f}")
        except Exception as e:
            logging.error(f"Error predicting for processor {processor_name}: {e}")
            processor_latencies[processor_name] = float('inf')

    if not processor_latencies:
        logging.warning("No processor latencies could be calculated.")
        return None, None, {}

    best_processor = min(processor_latencies, key=processor_latencies.get)
    lowest_predicted_latency = processor_latencies[best_processor]
    logging.info(f"Best processor: {best_processor} with predicted latency {lowest_predicted_latency:.4f}")

    return best_processor, lowest_predicted_latency, processor_latencies

def main():
    """
    Main function to orchestrate the ML pipeline:
    1. Load data and engineer features.
    2. Preprocess data (split, scale, encode).
    3. Train and evaluate the model.
    4. Save model artifacts.
    """
    logging.info("Starting payment processor routing model training pipeline.")

    df = load_data_and_engineer_features(DATA_FILE)

    X = df.drop('latency_ms', axis=1)
    y = df['latency_ms']

    # Store original y_test for evaluation after inverse transformation
    X_train, X_test, y_original_train, y_original_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE) 

    all_processors = df['processor'].unique().tolist()
    logging.info(f"All unique processors identified: {all_processors}")

    X_train_transformed, X_test_transformed, y_train_transformed, y_test_transformed, \
    feature_preprocessor, X_train_columns, X_train_dtypes = preprocess_data(X, y)

    # Pass original y_test for evaluation
    best_model = train_and_evaluate_model(X_train_transformed, y_train_transformed,
                                          X_test_transformed, y_test_transformed, y_original_test)

    save_model_artifacts(best_model, feature_preprocessor, X_train_columns, X_train_dtypes, MODEL_DIR)
    
    joblib.dump(all_processors, os.path.join(MODEL_DIR, 'all_processors.joblib'))
    logging.info("All unique processors list saved successfully as an artifact.")
    logging.info("Payment processor routing model training pipeline finished successfully.")
    return all_processors

if __name__ == "__main__":
    # Run the main training pipeline
    all_processors_from_training = main()

    # --- Example Usage of Predict Function ---
    try:
        loaded_model, loaded_feature_preprocessor, \
        loaded_X_train_columns, loaded_X_train_dtypes = load_model_artifacts(MODEL_DIR)

        loaded_all_processors = joblib.load(os.path.join(MODEL_DIR, 'all_processors.joblib'))

        new_transaction = {
            'amount': 150.0,
            'payment_method': 'Credit Card', 
            'merchant_category': 'Retail',
            'day_of_week': 'Monday',
            'time_of_day': 'Morning'
        }

        best_processor, lowest_latency, all_latencies = predict_best_processor(
            new_transaction,
            loaded_model,
            loaded_feature_preprocessor,
            loaded_all_processors,
            loaded_X_train_columns,
            loaded_X_train_dtypes
        )

        print("\n--- Prediction Results ---")
        if best_processor:
            print(f"Recommended Best Processor (lowest latency): {best_processor}")
            print(f"Predicted Lowest Latency: {lowest_latency:.4f} ms")
            print("Predicted Latencies for all processors:")
            for proc, lat in all_latencies.items():
                print(f"   {proc}: {lat:.4f} ms")
        else:
            print("Could not recommend a processor.")

    except Exception as e:
        logging.error(f"Failed to load artifacts or make a prediction: {e}")