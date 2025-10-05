import pandas as pd
import sklearn
import numpy as np # Import numpy
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer # Import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
import os
import mlflow
# for handling class imbalance
from sklearn.utils.class_weight import compute_class_weight
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Initialize HfApi for uploading the model file
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define constants for the dataset and output paths
DATASET_PATH = "hf://datasets/sauravghosh2109/tourism-package-predictor/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=['CustomerID'], inplace=True)

# Identify categorical and numerical columns
numerical_cols = ['Age', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'NumberOfTrips', 'NumberOfChildrenVisiting', 'MonthlyIncome', 'PitchSatisfactionScore', 'NumberOfFollowups', 'DurationOfPitch', 'Passport', 'OwnCar', 'CityTier']
nominal_categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'Designation', 'ProductPitched']

# Separate target variable
target_col = 'ProdTaken'
X = df.drop(columns=[target_col])
y = df[target_col]

# Create a column transformer for one-hot encoding nominal categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_categorical_cols)
    ],
    remainder='passthrough'
)

# Apply preprocessing to the whole dataset first to fit the preprocessor
X_processed = preprocessor.fit_transform(X)

# Save the fitted preprocessor
preprocessor_filename = "preprocessor.joblib"
preprocessor_save_path = os.path.join("tourism_project/model_building", preprocessor_filename)
joblib.dump(preprocessor, preprocessor_save_path)
print(f"Preprocessor saved locally to {preprocessor_save_path}")


# Convert the processed data back to a DataFrame to maintain column names
all_feature_names = preprocessor.get_feature_names_out()
X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)

print(f"Shape of X_processed_df: {X_processed_df.shape}")
print(f"Number of feature names: {len(all_feature_names)}")
print("Feature names:", all_feature_names)


# Perform train-test split on the processed data
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X_processed_df, y, test_size=0.2, random_state=42
)

# Save the processed data splits to CSV files
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


# Upload the processed data splits to Hugging Face Hub
files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]
repo_id = "sauravghosh2109/tourism-package-predictor" # Dataset repo

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id=repo_id,
        repo_type="dataset",
    )
print("Processed data splits uploaded to Hugging Face Hub.")


# Set the class weight to handle class imbalance
classes = np.array([0, 1])
class_weights = compute_class_weight('balanced', classes=classes, y=ytrain)
class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
scale_pos_weight_value = class_weight_dict[1] if 1 in class_weight_dict else 1

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight_value, random_state=42, use_label_encoder=False, eval_metric='logloss')

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'gamma': [0],
    'lambda': [1],
    'alpha': [0]
}

# Model training with GridSearchCV
grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='recall', n_jobs=-1)

with mlflow.start_run():
    grid_search.fit(Xtrain, ytrain)

    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_cross_val_recall", mean_score)
            mlflow.log_metric("std_cross_val_recall", std_score)

    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision_1": train_report['1']['precision'] if '1' in train_report else 0,
        "train_recall_1": train_report['1']['recall'] if '1' in train_report else 0,
        "train_f1-score_1": train_report['1']['f1-score'] if '1' in train_report else 0,
        "test_accuracy": test_report['accuracy'],
        "test_precision_1": test_report['1']['precision'] if '1' in test_report else 0,
        "test_recall_1": test_report['1']['recall'] if '1' in test_report else 0,
        "test_f1-score_1": test_report['1']['f1-score'] if '1' in test_report else 0
    })

    mlflow.sklearn.log_model(best_model, "tourism_package_prediction_model")

    print("Model training and evaluation complete. Check MLflow UI for results.")

# Save the trained model to a joblib file
model_filename = "tourism_package_prediction_model.joblib"
model_save_path = os.path.join("tourism_project/model_building", model_filename)
joblib.dump(best_model, model_save_path)
print(f"Model saved locally to {model_save_path}")

# Upload the saved model file and preprocessor file to Hugging Face Hub
model_repo_id = "sauravghosh2109/tourism-package-predictor-model"
model_repo_type = "model"

try:
    api.repo_info(repo_id=model_repo_id, repo_type=model_repo_type)
    print(f"Model space '{model_repo_id}' already exists. Uploading files.")
except RepositoryNotFoundError:
    print(f"Model space '{model_repo_id}' not found. Creating new space...")
    create_repo(repo_id=model_repo_id, repo_type=model_repo_type, private=False)
    print(f"Model space '{model_repo_id}' created.")
except Exception as e:
    print(f"Error checking for model space: {e}")

try:
    api.upload_file(
        path_or_fileobj=model_save_path,
        path_in_repo=model_filename,
        repo_id=model_repo_id,
        repo_type=model_repo_type,
    )
    print(f"Model file '{model_filename}' uploaded to Hugging Face Hub.")

    api.upload_file(
        path_or_fileobj=preprocessor_save_path,
        path_in_repo=preprocessor_filename,
        repo_id=model_repo_id, # Upload preprocessor to the same model repo
        repo_type=model_repo_type,
    )
    print(f"Preprocessor file '{preprocessor_filename}' uploaded to Hugging Face Hub.")

except Exception as e:
    print(f"Error uploading files to Hugging Face Hub: {e}")
