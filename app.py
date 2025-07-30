import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_DIR = "model_artifacts" 


# Cache the model loading to avoid reloading on every rerun
@st.cache_resource 
def load_model_artifacts(model_dir):
    """
    Loads the trained model, preprocessors, and all necessary metadata from disk.
    Uses st.cache_resource to load artifacts only once.
    """
    st.info(f"Loading model artifacts from {model_dir}...")
    try:
        loaded_model = joblib.load(os.path.join(model_dir, 'random_forest_regressor_model.joblib'))
        loaded_feature_preprocessor = joblib.load(os.path.join(model_dir, 'feature_preprocessor.joblib'))
        loaded_X_train_columns = joblib.load(os.path.join(model_dir, 'X_train_columns.joblib'))
        loaded_X_train_dtypes = joblib.load(os.path.join(model_dir, 'X_train_dtypes.joblib'))
        loaded_all_processors = joblib.load(os.path.join(model_dir, 'all_processors.joblib'))
        
        st.success("Model artifacts loaded successfully!")
        return loaded_model, loaded_feature_preprocessor, \
               loaded_X_train_columns, loaded_X_train_dtypes, loaded_all_processors
    except FileNotFoundError as e:
        st.error(f"Error: Model artifact not found. Please ensure all artifacts are trained and saved in '{model_dir}'. Missing file: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading model artifacts: {e}")
        st.stop()

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
                
                transaction_row.loc[0, key] = np.nan
        
    if not all_processors:
        st.error("No 'all_processors' list provided or loaded. Cannot perform predictions.")
        return None, None, {}

    for processor_name in all_processors:
        transaction_df_for_processor = transaction_row.copy()
        transaction_df_for_processor.loc[0, 'processor'] = processor_name 

        transaction_df_for_processor = transaction_df_for_processor[X_train_columns]

        try:
            transformed_transaction = feature_preprocessor.transform(transaction_df_for_processor)
            # Predict in the transformed scale
            predicted_latency_transformed = model.predict(transformed_transaction)[0]
            # Inverse transform to get original latency scale
            predicted_latency = np.expm1(predicted_latency_transformed) 
            processor_latencies[processor_name] = predicted_latency
        except Exception as e:
            
            processor_latencies[processor_name] = float('inf')

    if not processor_latencies:
        st.warning("No processor latencies could be calculated for any processor.")
        return None, None, {}

    best_processor = min(processor_latencies, key=processor_latencies.get)
    lowest_predicted_latency = processor_latencies[best_processor]

    return best_processor, lowest_predicted_latency, processor_latencies


# --- Streamlit App Layout ---

st.set_page_config(page_title="Dynamic Payment Processor Router", layout="wide") # Simpler title

st.title("Dynamic Payment Processor Router")

st.markdown("""
    Optimize payment latency by predicting the best processor.
""")

# Load artifacts
model, feature_preprocessor, X_train_columns, X_train_dtypes, all_processors = load_model_artifacts(MODEL_DIR)

# Initialize session state for prediction trigger and data storage
if 'predict_clicked' not in st.session_state:
    st.session_state['predict_clicked'] = False
if 'transaction_data' not in st.session_state:
    st.session_state['transaction_data'] = {}



trans_details_col, time_features_col, results_col = st.columns([0.8, 1, 1.2]) 

with trans_details_col:
    st.subheader("Transaction Details") 
    
    amount = st.number_input("Amount", min_value=1.0, max_value=10000.0, value=150.0, step=10.0, help="Transaction amount (e.g., USD)")
    payment_method = st.selectbox("Method", options=['Credit Card', 'Debit Card', 'Bank Transfer', 'Digital Wallet', 'Crypto'], help="Payment method used for the transaction")
    merchant_category = st.selectbox("Category", options=['Retail', 'Travel', 'Digital Goods', 'Services', 'Gaming', 'SaaS'], help="Business category of the merchant")

with time_features_col:
    st.subheader("Time Features") 
    # Made radio buttons horizontal to save vertical space
    time_input_option = st.radio(
        "Set Time:",
        ('Current Time', 'Manual'),
        index=0,
        horizontal=True, 
        help="Choose whether to use current time or specify manually"
    )

    # Time of day binning logic (must match training script's logic)
    bins = [0, 6, 12, 18, 24]
    labels = ['Late Night', 'Morning', 'Afternoon', 'Evening']

    if time_input_option == 'Current Time':
        current_time = datetime.now() 
        selected_hour = current_time.hour
        selected_day_of_week = current_time.strftime("%A") 
        selected_time_of_day = pd.cut([selected_hour], bins=bins, labels=labels, right=False, include_lowest=True)[0]

        st.write(f"Hour: **{selected_hour}:00**")
        st.write(f"Day: **{selected_day_of_week}**")
        st.write(f"Time: **{selected_time_of_day}**")
    else: # Manual Input
        selected_hour = st.slider("Hour (0-23)", 0, 23, datetime.now().hour, help="Select the hour of the transaction")
        selected_day_of_week = st.selectbox(
            "Day of Week",
            options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            index=datetime.now().weekday(), # Default to current day
            help="Select the day of the week for the transaction"
        )
        selected_time_of_day = pd.cut([selected_hour], bins=bins, labels=labels, right=False, include_lowest=True)[0]

        st.write(f"Selected Hour: **{selected_hour}:00**")
        st.write(f"Selected Day: **{selected_day_of_week}**")
        st.write(f"Calculated Time: **{selected_time_of_day}**") 
    
    st.markdown("---") 
    # The predict button is placed at the bottom of the time features column
    if st.button("Predict Processor", help="Click to get the recommended processor based on the input details.", use_container_width=True):
        st.session_state['predict_clicked'] = True
        # Store input data in session state when button is clicked
        st.session_state['transaction_data'] = {
            'amount': float(amount),
            'payment_method': payment_method, 
            'merchant_category': merchant_category,
            'time_of_day': selected_time_of_day,
            'day_of_week': selected_day_of_week
        }

with results_col:
    st.subheader("Prediction Results") 
    # This section only renders if the 'Predict Processor' button has been clicked
    if st.session_state.get('predict_clicked', False):
        if model is None:
            st.error("Model not loaded. Please ensure artifacts are available.", icon="❌")
        else:
            with st.spinner("Predicting..."):
                # Retrieve input data from session state
                new_transaction_data = st.session_state['transaction_data']

                best_processor, lowest_latency, all_latencies = predict_best_processor(
                    new_transaction_data,
                    model,
                    feature_preprocessor,
                    all_processors,
                    X_train_columns,
                    X_train_dtypes
                )

                if best_processor:
                    # Concise success message
                    st.success(f"**Recommended:** `{best_processor}`\n\n**Latency:** `{lowest_latency:.2f} ms`", icon="✨")
                    
                    # Convert dictionary to DataFrame for easy plotting
                    latencies_df = pd.DataFrame(all_latencies.items(), columns=['Processor', 'Predicted Latency (ms)'])
                    latencies_df = latencies_df.sort_values(by='Predicted Latency (ms)', ascending=True)

                    # Create the bar chart using Matplotlib/Seaborn with aggressive size reduction
                    fig, ax = plt.subplots(figsize=(4.5, 2.5))
                    sns.barplot(x='Predicted Latency (ms)', y='Processor', data=latencies_df, palette='viridis', ax=ax)
                    ax.set_title("Latency per Processor", fontsize=9) 
                    ax.set_xlabel("Latency (ms)", fontsize=8) 
                    ax.set_ylabel("") 
                    ax.tick_params(axis='y', labelsize=6)
                    ax.tick_params(axis='x', labelsize=6)
                    plt.tight_layout(pad=0.1)

                    # Highlight the best processor's bar in green
                    best_processor_index = latencies_df[latencies_df['Processor'] == best_processor].index[0]
                    for i, bar in enumerate(ax.patches):
                        if i == best_processor_index:
                            bar.set_facecolor('green') 
                        

                    st.pyplot(fig) 
                    plt.close(fig) 

                    # Detailed list in an expander for compactness
                    with st.expander("All Latencies"): 
                        sorted_latencies = sorted(all_latencies.items(), key=lambda item: item[1], reverse=False)
                        for proc, lat in sorted_latencies:
                            st.write(f"**{proc}:** `{lat:.2f} ms`")
                else:
                    st.error("Could not recommend a processor. Please check inputs and model status.", icon="❌")
    else:
        # Initial message in the results column before prediction
        st.info("Enter details on the left and click 'Predict Processor' to see results here.", icon="💡")

# Footer for the application - made very concise
st.markdown("---")
#st.caption("This is a proof-of-concept. Real-world data and continuous model monitoring are essential for production use.")