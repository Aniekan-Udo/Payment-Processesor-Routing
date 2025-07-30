import pandas as pd
import numpy as np
import joblib
import logging
import os
from datetime import datetime # Import datetime module
# --- Configuration ---
MODEL_DIR = "model_artifacts"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Log to console
    ]
)

def load_model_artifacts(model_dir):
    """
    Loads the trained model, preprocessors, and all necessary metadata from disk.

    Args:
        model_dir (str): Directory where the artifacts are saved.

    Returns:
        tuple: (loaded_model, loaded_feature_preprocessor, loaded_target_transformer,
                loaded_X_train_columns, loaded_X_train_dtypes, loaded_all_processors)
    """
    logging.info(f"Loading model artifacts from {model_dir}...")
    try:
        loaded_model = joblib.load(os.path.join(model_dir, 'random_forest_model.joblib'))
        loaded_feature_preprocessor = joblib.load(os.path.join(model_dir, 'feature_preprocessor.joblib'))
        loaded_target_transformer = joblib.load(os.path.join(model_dir, 'target_transformer.joblib'))
        loaded_X_train_columns = joblib.load(os.path.join(model_dir, 'X_train_columns.joblib'))
        loaded_X_train_dtypes = joblib.load(os.path.join(model_dir, 'X_train_dtypes.joblib'))
        
        # --- NEW: Load the all_processors list ---
        loaded_all_processors = joblib.load(os.path.join(model_dir, 'all_processors.joblib'))
        
        logging.info("Model artifacts loaded successfully.")
        return loaded_model, loaded_feature_preprocessor, loaded_target_transformer, \
               loaded_X_train_columns, loaded_X_train_dtypes, loaded_all_processors
    except FileNotFoundError as e:
        logging.error(f"Error: Model artifact not found. Please ensure all artifacts are trained and saved in '{model_dir}'. Missing file: {e.filename}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading model artifacts: {e}")
        raise

def predict_best_processor(new_transaction_data, model, feature_preprocessor, target_transformer, all_processors, X_train_columns, X_train_dtypes):
    """
    Predicts the best processor for a new transaction based on the success probability.

    Args:
        new_transaction_data (dict): Features of the new transaction excluding 'processor','latency' and 'status'.
                                     Example: {'amount': 150.0, 'currency': 'USD', 'time_of_day': 'Morning'}
        model: The trained machine learning model.
        feature_preprocessor: The fitted ColumnTransformer.
        target_transformer: The fitted LabelEncoder for the target variable.
        all_processors (list): A list of all unique processor names seen during training. This is crucial
                               to ensure consistency in predictions.
        X_train_columns (pd.Index): The columns of the original X_train DataFrame,
                                     used to ensure consistent input structure.
        X_train_dtypes (pd.Series): The dtypes of the original X_train DataFrame,
                                     used to ensure consistent input types.

    Returns:
        tuple: (best_processor_name, highest_probability, all_probabilities_dict)
               Returns (None, None, {}) if 'success' class is not found or no probabilities can be calculated.
    """
    logging.info("Predicting best processor for new transaction...")
    processor_probabilities = {}

    # Check if 'success' class is present in the target transformer
    if 'success' not in target_transformer.classes_:
        logging.warning("Warning: 'success' class not found in target classes. Cannot predict success probability.")
        return None, None, {}

    # Get the index of the 'success' class
    success_class_idx = np.where(target_transformer.classes_ == 'success')[0][0]

    # Create an empty DataFrame with the correct columns and dtypes based on training data
    transaction_row = pd.DataFrame(index=[0], columns=X_train_columns)
    for col in X_train_columns:
        if isinstance(X_train_dtypes[col], pd.CategoricalDtype):
            # For categorical columns, ensure the dtype includes the original categories
            transaction_row[col] = pd.Series([np.nan], dtype=pd.CategoricalDtype(
                categories=X_train_dtypes[col].categories, ordered=X_train_dtypes[col].ordered
            ))
        elif pd.api.types.is_numeric_dtype(X_train_dtypes[col]):
            # Ensure numerical columns are float-compatible for NaNs
            transaction_row[col] = pd.Series([np.nan], dtype=float)
        else:
            # For other types, assign the original dtype
            transaction_row[col] = pd.Series([np.nan], dtype=X_train_dtypes[col])

    # Populate the single row with new_transaction_data
    for key, value in new_transaction_data.items():
        if key in transaction_row.columns:
            try:
                transaction_row.loc[0, key] = value
            except Exception as e:
                logging.warning(f"Could not assign value '{value}' to column '{key}' in transaction_row: {e}. "
                                f"Ensure value is compatible with dtype/categories. Setting to NaN.")
                transaction_row.loc[0, key] = np.nan # Set to NaN if assignment fails
        else:
            logging.warning(f"New transaction data contains unknown column: '{key}'. It will be ignored.")

    # Iterate through all known processors to predict success probability for each
    if not all_processors:
        logging.error("No 'all_processors' list provided or loaded. Cannot perform predictions.")
        return None, None, {}

    for processor_name in all_processors:
        transaction_df_for_processor = transaction_row.copy()
        # Set the current processor name for the prediction
        transaction_df_for_processor.loc[0, 'processor'] = processor_name 

        # Ensure the order of columns matches the training data before transformation
        # This is critical for ColumnTransformer to work correctly
        transaction_df_for_processor = transaction_df_for_processor[X_train_columns]

        try:
            # Transform the single transaction row using the fitted preprocessor
            transformed_transaction = feature_preprocessor.transform(transaction_df_for_processor)
            
            # Get probability predictions from the model
            probabilities = model.predict_proba(transformed_transaction)
            
            # Extract the success probability for the current processor
            success_prob = probabilities[0, success_class_idx]
            processor_probabilities[processor_name] = success_prob
            logging.debug(f"Processor: {processor_name}, Success Probability: {success_prob:.4f}")
        except Exception as e:
            logging.error(f"Error predicting for processor {processor_name}: {e}. Assigning 0.0 probability.")
            processor_probabilities[processor_name] = 0.0 # Assign a default low probability on error

    if not processor_probabilities:
        logging.warning("No processor probabilities could be calculated for any processor.")
        return None, None, {}

    # Determine the best processor based on the highest success probability
    best_processor = max(processor_probabilities, key=processor_probabilities.get)
    highest_probability = processor_probabilities[best_processor]
    logging.info(f"Best processor: {best_processor} with success probability {highest_probability:.4f}")

    return best_processor, highest_probability, processor_probabilities

if __name__ == "__main__":
    logging.info("Starting payment processor prediction script.")

    try:
        # Load all saved model artifacts, including the list of all processors
        loaded_model, loaded_feature_preprocessor, loaded_target_transformer, \
        loaded_X_train_columns, loaded_X_train_dtypes, loaded_all_processors = load_model_artifacts(MODEL_DIR)

        # --- Automatically get current time and derive features ---
        current_time = datetime.now()
        current_hour = current_time.hour
        current_day_of_week = current_time.strftime("%A") # e.g., 'Monday', 'Tuesday'

        # --- Time of day binning (MUST MATCH main.py's logic) ---
        time_bins = [5, 12, 17, 21, 24]
        time_labels = ['Morning', 'Afternoon', 'Evening', 'Late Night']
        
        current_time_of_day = None
        for i in range(len(time_bins)):
            if current_hour >= time_bins[i]:
                if i + 1 < len(time_bins) and current_hour < time_bins[i+1]:
                    current_time_of_day = time_labels[i]
                    break
                elif i + 1 == len(time_bins): # Last bin (21-24)
                    current_time_of_day = time_labels[i]
                    break
        if current_time_of_day is None: # For hours less than the first bin (e.g., 0-4)
            current_time_of_day = 'Late Night' # Assign to 'Late Night'


        # Define a new transaction for prediction
        # 'time_of_day' and 'day_of_week' are now automatically populated
        new_transaction = {
            'amount': 150.0,
            'country': 'USA',
            'currency': 'USD',
            'payment_method':'Credit Card',
            'marchant_category':'travel',
            'time_of_day': current_time_of_day, # Automatically derived
            'day_of_week': current_day_of_week  # Automatically derived
        }

        print(f"\n--- Current Time Features ---")
        print(f"Current Hour: {current_hour}")
        print(f"Current Day of Week: {current_day_of_week}")
        print(f"Current Time of Day: {current_time_of_day}")
        print(f"Input Transaction Details: {new_transaction}")


        # Predict the best processor for the new transaction
        best_processor, highest_prob, all_probs = predict_best_processor(
            new_transaction,
            loaded_model,
            loaded_feature_preprocessor,
            loaded_target_transformer,
            loaded_all_processors,
            loaded_X_train_columns,
            loaded_X_train_dtypes
        )

        print("\n--- Prediction Results ---")
        if best_processor:
            print(f"Recommended Best Processor: {best_processor}")
            print(f"Predicted Success Probability: {highest_prob:.4f}")
            print("Probabilities for all processors:")
            for proc, prob in all_probs.items():
                print(f"  {proc}: {prob:.4f}")
        else:
            print("Could not recommend a processor.")

    except Exception as e:
        logging.error(f"Prediction script failed: {e}")

    logging.info("Payment processor prediction script finished.")
