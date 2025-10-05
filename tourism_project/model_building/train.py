import pandas as pd
import sklearn
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
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

Xtrain_path = "hf://datasets/sauravghosh2109/tourism-package-predictor/Xtrain.csv"
Xtest_path = "hf://datasets/sauravghosh2109/tourism-package-predictor/Xtest.csv"
ytrain_path = "hf://datasets/sauravghosh2109/tourism-package-predictor/ytrain.csv"
ytest_path = "hf://datasets/sauravghosh2109/tourism-package-predictor/ytest.csv"

# Load the processed data splits
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze() # Use squeeze to convert DataFrame to Series
ytest = pd.read_csv(ytest_path).squeeze() # Use squeeze to convert DataFrame to Series

print("Processed data loaded successfully.")
print(f"Xtrain shape: {Xtrain.shape}")
print(f"Xtest shape: {Xtest.shape}")
print(f"ytrain shape: {ytrain.shape}")
print(f"ytest shape: {ytest.shape}")


# Re-identify numerical features based on the processed data (excluding one-hot encoded columns)
# Assuming columns that are not one-hot encoded are the original numerical columns and CityTier
# We can get these by finding columns in Xtrain that don't have '_TypeofContact', etc. suffixes
# Or, more simply, reuse the numerical_cols list from the prep step if it's consistent.
# Let's assume the numerical_cols list from prep.py is correct for the raw data,
# and the one-hot encoding only applied to nominal_categorical_cols.
# After one-hot encoding and dropping the original nominal categorical columns, the remaining
# columns should be the original numerical columns plus the new one-hot encoded columns.

# We can identify numerical columns in Xtrain_processed by checking their dtype
numerical_features = Xtrain.select_dtypes(include=['float64', 'int64']).columns.tolist()

# The target column is 'ProdTaken'

# Set the class weight to handle class imbalance in the target variable 'ProdTaken'
# Since ytrain is a pandas Series, compute_class_weight can be applied directly
classes = [0, 1] # Assuming the target classes are 0 and 1
class_weights = compute_class_weight('balanced', classes=classes, y=ytrain)
# Convert class weights to a dictionary for XGBoost
class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
# XGBoost's scale_pos_weight expects the weight for the positive class (1) relative to the negative class (0)
scale_pos_weight_value = class_weight_dict[1] if 1 in class_weight_dict else 1 # Default to 1 if only one class exists


# Define base XGBoost model
# Use scale_pos_weight for binary classification imbalance
xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight_value, random_state=42, use_label_encoder=False, eval_metric='logloss')


# Define hyperparameter grid - adjusted for a potentially smaller search space initially
# and focusing on parameters relevant for this problem.
param_grid = {
    'n_estimators': [100, 200], # Number of boosting rounds
    'max_depth': [3, 5, 7],     # Maximum depth of a tree
    'learning_rate': [0.01, 0.05, 0.1], # Step size shrinkage
    'subsample': [0.7, 0.8, 0.9], # Subsample ratio of the training instances
    'colsample_bytree': [0.7, 0.8, 0.9], # Subsample ratio of columns when constructing each tree
    'gamma': [0, 0.1, 0.2], # Minimum loss reduction required to make a further partition on a leaf node
    'lambda': [1, 1.5], # L2 regularization term on weights
    'alpha': [0, 0.5] # L1 regularization term on weights
}

# Model training with GridSearchCV - directly on XGBoost model as preprocessing is already done
# We are now training the xgb_model directly on the preprocessed data
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='recall', n_jobs=-1) # Using recall as a primary metric due to potential imbalance

with mlflow.start_run():
    # Hyperparameter tuning
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i] # This is the mean recall score
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_cross_val_recall", mean_score) # Log recall
            mlflow.log_metric("std_cross_val_recall", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    # Evaluate on training and testing sets
    # Use the best model to make predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Evaluate using classification report
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log key metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision_1": train_report['1']['precision'] if '1' in train_report else 0, # Handle case where class 1 has no predictions
        "train_recall_1": train_report['1']['recall'] if '1' in train_report else 0,
        "train_f1-score_1": train_report['1']['f1-score'] if '1' in train_report else 0,
        "test_accuracy": test_report['accuracy'],
        "test_precision_1": test_report['1']['precision'] if '1' in test_report else 0, # Handle case where class 1 has no predictions
        "test_recall_1": test_report['1']['recall'] if '1' in test_report else 0,
        "test_f1-score_1": test_report['1']['f1-score'] if '1' in test_report else 0
    })

    # Log the model
    mlflow.sklearn.log_model(best_model, "tourism_package_prediction_model")

print("Model training and evaluation complete. Check MLflow UI for results.")
