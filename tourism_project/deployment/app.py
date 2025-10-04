import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# Define the Hugging Face repo and filename for the model
HF_REPO_ID = "sauravghosh2109/tourism-package-prediction"  # Replace with your actual repo ID if different
MODEL_FILENAME = "tourism_package_prediction_model.joblib"  # Replace with your actual model filename if different

# Download and load the model
# Ensure the model filename here matches the one used in your training script's mlflow.sklearn.log_model call
try:
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
    model = joblib.load(model_path)
    st.success("Model loaded successfully from Hugging Face Hub.")
except Exception as e:
    st.error(f"Error loading model from Hugging Face Hub: {e}")
    st.info(
        f"Attempted to download model '{MODEL_FILENAME}' from repo '{HF_REPO_ID}'. Please ensure the model exists in your repo."
    )
    st.stop()  # Stop the app if the model cannot be loaded


# Streamlit UI for Tourism Package Prediction
st.title("Wellness Tourism Package Prediction")
st.write(
    """
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package based on their details and interaction data.
Please enter the customer details below to get a prediction.
"""
)

# User input fields based on the tourism.csv dataset description
# Customer Details
st.header("Customer Details")
age = st.number_input("Age", min_value=0, max_value=120, value=30)
typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
citytier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox(
    "Occupation", ["Salaried", "Freelancer", "Large Business", "Small Business"]
)  # Add other occupations as per your data
gender = st.selectbox("Gender", ["Male", "Female"])
numberofpersonvisiting = st.number_input(
    "Number of People Visiting", min_value=1, max_value=20, value=1
)
preferredpropertystar = st.selectbox(
    "Preferred Property Star", [3, 4, 5]
)  # Add other stars as per your data
maritalstatus = st.selectbox(
    "Marital Status", ["Single", "Married", "Divorced"]
)  # Add other statuses
numberoftrips = st.number_input(
    "Number of Trips Annually", min_value=0, max_value=100, value=1
)
passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
owncar = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
numberofchildrenvisiting = st.number_input(
    "Number of Children Visiting", min_value=0, max_value=10, value=0
)
designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP",
        "Director",
        "Cluster Manager",
        "Area Manager",
        "Executive",
        "Senior Executive",
        "Junior Executive",
    ],
)  # Add other designations
monthlyincome = st.number_input(
    "Monthly Income", min_value=0.0, value=50000.0, step=1000.0
)

# Customer Interaction Data
st.header("Customer Interaction Data")
pitchsatisfactionscore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
productpitched = st.selectbox(
    "Product Pitched",
    [
        "Duration",
        "Accommodation",
        "Food",
        "Transport",
        "Road trip",
        "Cultural",
        "Adventure",
        "Relaxation",
        "Museum",
        "City tour",
        "Cruise",
        "Nature",
    ],
)  # Add other products
numberoffollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=1)
durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=10.0, step=0.1)


# Assemble input into DataFrame - This needs to match the columns and order
# the model was trained on AFTER preprocessing (one-hot encoding, etc.)
# For simplicity in the app, we'll create a DataFrame with the original columns
# and assume the loaded model includes the necessary preprocessing steps within a pipeline
# or we manually apply a simplified version of preprocessing here.

# A more robust approach would be to save the ColumnTransformer from prep.py and load it here.
# However, given the complexity of reproducing the exact ColumnTransformer output manually,
# let's create a DataFrame with the original columns and assume the model handles the transformation internally
# or requires the original column names (less likely with OneHotEncoder).

# Let's create a DataFrame that mimics the structure of the *original* data used for training,
# before one-hot encoding, and then manually apply a simple one-hot encoding matching the training.

input_data = pd.DataFrame(
    [
        {
            "Age": age,
            "TypeofContact": typeofcontact,
            "CityTier": citytier,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": numberofpersonvisiting,
            "PreferredPropertyStar": preferredpropertystar,
            "MaritalStatus": maritalstatus,
            "NumberOfTrips": numberoftrips,
            "Passport": passport,
            "OwnCar": owncar,
            "NumberOfChildrenVisiting": numberofchildrenvisiting,
            "Designation": designation,
            "MonthlyIncome": monthlyincome,
            "PitchSatisfactionScore": pitchsatisfactionscore,
            "ProductPitched": productpitched,
            "NumberOfFollowups": numberoffollowups,
            "DurationOfPitch": durationofpitch,
        }
    ]
)

# --- Manual One-Hot Encoding (Simplified) ---
# This part needs to be carefully aligned with the one-hot encoding done in prep.py
# It's best to use the exact ColumnTransformer if possible, but this is a manual approximation
# based on the columns identified as nominal categorical in prep.py
nominal_categorical_cols = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched",
]
numerical_cols_including_citytier = [
    "Age",
    "CityTier",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch",
]


# Convert categorical columns to 'category' dtype before one-hot encoding
for col in nominal_categorical_cols:
    input_data[col] = input_data[col].astype("category")

# Perform one-hot encoding
# Ensure drop_first=False to match potential training behavior
input_data_processed = pd.get_dummies(
    input_data, columns=nominal_categorical_cols, drop_first=False
)

# --- Ensure columns match training data columns ---
# This is crucial. The columns and their order MUST match the Xtrain DataFrame
# that the model was trained on *after* one-hot encoding.
# The safest way is to get the exact list of columns from Xtrain.csv
# Let's assume we can load Xtrain.csv here just to get the column order and names.
# In a real deployment, you might save and load the list of column names.

try:
    xtrain_cols_path = hf_hub_download(repo_id=HF_REPO_ID, filename="Xtrain.csv")
    xtrain_cols_df = pd.read_csv(xtrain_cols_path, nrows=0)  # Load only header
    training_columns = xtrain_cols_df.columns.tolist()

    # Reindex the input data DataFrame to match the training columns
    # Missing columns will be added with a value of 0 (correct for one-hot encoding)
    # Extra columns in input_data_processed (shouldn't happen if columns are consistent) will be dropped
    input_data_aligned = input_data_processed.reindex(
        columns=training_columns, fill_value=0
    )

except Exception as e:
    st.error(f"Error aligning input data columns with training data: {e}")
    st.info(
        "Could not load training columns from Xtrain.csv. Prediction may fail due to column mismatch."
    )
    input_data_aligned = input_data_processed  # Proceed with unaligned data, likely leading to error


if st.button("Predict Purchase"):
    if "model" in locals() and model is not None:
        try:
            # Make prediction
            prediction = model.predict(input_data_aligned)[0]

            # Display result
            result = "Yes, likely to purchase" if prediction == 1 else "No, not likely to purchase"
            st.subheader("Prediction Result:")
            if prediction == 1:
                st.success(f"The model predicts: **{result}**")
            else:
                st.info(f"The model predicts: **{result}**")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please ensure the input data format matches the model's expected input.")

    else:
        st.warning("Model not loaded. Please check the model loading steps.")
