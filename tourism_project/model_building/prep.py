# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder # Keep LabelEncoder for the target variable if needed later, or remove if not used.
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token="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")
DATASET_PATH = "hf://datasets/sauravghosh2109/tourism-package-prediction/tourism.csv" # Corrected path to the uploaded file
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=['CustomerID'], inplace=True) # Corrected unique identifier

# Identify categorical and numerical columns
# Assuming 'CityTier' is already numerical based on description (Tier 1, 2, 3)
# Other numerical columns based on description: Age, NumberOfPersonVisiting, PreferredPropertyStar, NumberOfTrips, NumberOfChildrenVisiting, MonthlyIncome, PitchSatisfactionScore, NumberOfFollowups, DurationOfPitch, Passport, OwnCar
numerical_cols = ['Age', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'NumberOfTrips', 'NumberOfChildrenVisiting', 'MonthlyIncome', 'PitchSatisfactionScore', 'NumberOfFollowups', 'DurationOfPitch', 'Passport', 'OwnCar', 'CityTier']
# Nominal categorical columns to be one-hot encoded
nominal_categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'Designation', 'ProductPitched']


# Separate target variable
target_col = 'ProdTaken' # Corrected target column
X = df.drop(columns=[target_col])
y = df[target_col]

# Create a column transformer for one-hot encoding nominal categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), nominal_categorical_cols)
    ],
    remainder='passthrough' # Keep numerical columns as they are
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# Convert the processed data back to a DataFrame to maintain column names (optional but helpful)
# Get feature names after one-hot encoding
onehot_feature_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(nominal_categorical_cols)
all_feature_names = list(onehot_feature_names) + [col for col in numerical_cols if col in X.columns] # Ensure numerical columns are also in original X

X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)


# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X_processed_df, y, test_size=0.2, random_state=42
)

# Save the processed data splits to CSV files
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="sauravghosh2109/tourism-package-prediction",
        repo_type="dataset",
    )
