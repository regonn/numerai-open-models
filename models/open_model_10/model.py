import gc
import json
import os
import random

import cloudpickle
import lightgbm as lgb
import numerapi
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

load_dotenv()

DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
NUMERAI_PUBLIC_ID = os.environ["NUMERAI_PUBLIC_ID"]
NUMERAI_SECRET_KEY = os.environ["NUMERAI_SECRET_KEY"]
NUMERAI_MODEL_ID = "open_model_10"
napi = numerapi.NumerAPI(NUMERAI_PUBLIC_ID, NUMERAI_SECRET_KEY)

# Download data
napi.download_dataset("v5.0/train.parquet")
napi.download_dataset("v5.0/validation.parquet")
napi.download_dataset("v5.0/live.parquet")
napi.download_dataset("v5.0/features.json")
napi.download_dataset("v5.0/live_example_preds.parquet")

model_file_name = f"models/{NUMERAI_MODEL_ID}/predict_{NUMERAI_MODEL_ID}.pkl"
feature_metadata = json.load(open("v5.0/features.json"))
features = feature_metadata["feature_sets"]["small"]

# Download feature importance from Numerai to identify potential risk factors
try:
    feature_importance = pd.Series(json.load(open("v5.0/features_importance.json")))
    # Sort features by importance
    top_features = list(feature_importance.sort_values(ascending=False).head(50).index)
    # Use features with high importance for neutralization
    neutralization_features = [f for f in top_features if f in features][:20]
except:
    # If feature importance file is not available, use some features with "group" in name as risk factors
    neutralization_features = [f for f in features if "group" in f][:20]
    if len(neutralization_features) < 5:
        # If no group features, use first 20 features for neutralization
        neutralization_features = features[:20]

target = "target"

# Train data
train = pd.read_parquet("v5.0/train.parquet", columns=["era"] + features + [target])
train = train.dropna(subset=target, axis=0)
train.to_csv("train.csv", index=False)

# Validation data
validation = pd.read_parquet(
    "v5.0/validation.parquet",
    columns=["era", "data_type"] + features + [target],
)
validation = validation[validation["data_type"] == "validation"]
validation = validation.dropna(subset=[target], axis=0)

# Get example predictions for checking format
example_preds = pd.read_parquet("v5.0/live_example_preds.parquet")


# Feature engineering - add feature interactions and statistical features
def engineer_features(df):
    # Create a copy to avoid modifying the original dataframe
    df_engineered = df.copy()

    # Only perform era-based feature engineering if 'era' column exists
    if "era" in df.columns:
        try:
            # Group features by era to capture temporal dynamics
            era_stats = df.groupby("era")[features].mean()
            era_stats.columns = [f"era_mean_{col}" for col in era_stats.columns]
            df_engineered = df_engineered.join(era_stats, on="era")
        except Exception as e:
            print(f"Warning: Could not create era-based features: {e}")
            # Continue without era-based features

    # Create interaction features between high importance features
    # Use only top 5 features to avoid explosion of dimensions
    top_5_features = neutralization_features[:5]
    for i in range(len(top_5_features)):
        for j in range(i + 1, len(top_5_features)):
            feat_i, feat_j = top_5_features[i], top_5_features[j]
            # Multiplication interaction
            df_engineered[f"{feat_i}_mul_{feat_j}"] = df_engineered[feat_i] * df_engineered[feat_j]
            # Division interaction (with safety for zeros)
            df_engineered[f"{feat_i}_div_{feat_j}"] = df_engineered[feat_i] / (df_engineered[feat_j].replace(0, 1e-8))

    # Add polynomial features for top features
    for feat in top_5_features:
        df_engineered[f"{feat}_squared"] = df_engineered[feat] ** 2

    return df_engineered


# Apply feature engineering
print("Applying feature engineering...")
train_engineered = engineer_features(train)
validation_engineered = engineer_features(validation)

# Get the new feature list after engineering
engineered_features = [
    col
    for col in train_engineered.columns
    if col not in ["era", "data_type", target] and not pd.isna(train_engineered[col]).any()
]

gc.collect()


# Feature neutralization function
def neutralize(df, columns, by, proportion=1.0):
    """Neutralize a dataframe of predictions against a set of features

    Parameters
    ----------
    df: DataFrame with the predictions
    columns: list of str, columns to neutralize
    by: list of str, columns to neutralize against
    proportion: float, amount of neutralization to apply (1.0 = full neutralization)

    Returns
    -------
    DataFrame with neutralized predictions

    """
    if proportion == 0.0:
        return df.copy()

    # Convert to numpy for performance
    scores = df[columns].values
    factors = df[by].values

    # Standardize the factors
    factors = StandardScaler().fit_transform(factors)

    # Perform neutralization (regression residuals)
    scores_neutralized = np.zeros_like(scores)
    for i in range(scores.shape[1]):
        # Perform linear regression
        model = LinearRegression(fit_intercept=False).fit(factors, scores[:, i])
        # Calculate the residuals
        scores_neutralized[:, i] = scores[:, i] - proportion * model.predict(factors)

    # Convert back to DataFrame
    df_neutralized = pd.DataFrame(scores_neutralized, index=df.index, columns=columns)

    return df_neutralized


# Improved model parameters that will help avoid early stopping at iteration 1
model_params = {
    "n_estimators": 2000,
    "learning_rate": 0.001,  # Much lower learning rate to prevent early stopping at 1
    "max_depth": 3,  # Even more controlled tree depth to prevent overfitting
    "num_leaves": 15,  # Fewer leaves to prevent overfitting
    "colsample_bytree": 0.5,  # Increase column sampling for better generalization
    "subsample": 0.5,  # More aggressive subsampling to prevent overfitting
    "reg_alpha": 1.0,  # Much stronger L1 regularization
    "reg_lambda": 5.0,  # Much stronger L2 regularization
    "min_child_samples": 50,  # Require more samples per leaf for stability
    "min_child_weight": 1.0,  # Higher weight requirement for leaf nodes
    "bagging_freq": 1,  # Perform bagging at every iteration
    "feature_fraction": 0.6,  # Use fewer features per tree for diversity
    "random_state": 42,
    "n_jobs": -1,  # Use all CPU cores for training
    "verbose": -1,  # Silent mode
    "boosting_type": "gbdt",  # Use traditional gradient boosting
    "min_gain_to_split": 0.01,  # Require more gain for splits to occur
    "min_data_in_bin": 5,  # More data in each bin for feature discretization
    "max_bin": 100,  # Reduce number of bins for feature discretization
}

print("Training model with cross-validation...")

# Extract unique eras for time-based validation
unique_eras = sorted(train_engineered["era"].unique())
n_eras = len(unique_eras)
n_folds = 5  # Number of folds for cross-validation

# Initialize list to store models from each fold
models = []

# Track validation scores across folds
val_scores = []

# Time-series cross-validation - use recent eras for validation
for fold in range(n_folds):
    print(f"Training fold {fold + 1}/{n_folds}")

    # Determine cutoff points for this fold
    # We'll use time-based splits to avoid data leakage
    validation_eras = unique_eras[-(fold + 1) * 3 :] if fold < n_folds - 1 else unique_eras[-3:]
    train_eras = [era for era in unique_eras if era not in validation_eras]

    # Split data
    train_fold = train_engineered[train_engineered["era"].isin(train_eras)]
    val_fold = train_engineered[train_engineered["era"].isin(validation_eras)]

    # Initialize and train model for this fold
    model = lgb.LGBMRegressor(**model_params)

    # Train with a more patient early stopping using validation set
    model.fit(
        train_fold[engineered_features],
        train_fold[target],
        eval_set=[(val_fold[engineered_features], val_fold[target])],
        eval_metric=["mae", "rmse"],  # Use multiple metrics for better evaluation
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=True),  # Much more patient early stopping
            lgb.log_evaluation(period=50),  # Log more frequently
        ],
        init_model=None,  # Start fresh each time
    )

    # Save best iteration for this fold
    print(f"Fold {fold + 1} - Best iteration: {model.best_iteration_}")
    val_scores.append(model.best_score_["valid_0"]["l1"])

    # Add model to our collection
    models.append(model)

# Calculate and print average validation score
avg_val_score = sum(val_scores) / len(val_scores)
print(f"Average validation MAE across folds: {avg_val_score:.6f}")

# Train a final model with more iterations and without early stopping
print("Training final model on all data...")
# For the final model, we'll use a slightly different configuration
final_model_params = model_params.copy()
final_model_params["n_estimators"] = 500  # Set a fixed number of iterations for final model
final_model = lgb.LGBMRegressor(**final_model_params)

# Train without early stopping on all data
final_model.fit(
    train_engineered[engineered_features],
    train_engineered[target],
    eval_set=[(validation_engineered[engineered_features], validation_engineered[target])],
    eval_metric=["mae", "rmse"],  # Use multiple metrics
    callbacks=[
        lgb.log_evaluation(period=50),  # Log frequently
    ],
    # No early stopping for final model - train for all iterations
)

# For prediction, we'll use both the cross-validated models and the final model
# This ensemble approach should improve robustness

# Predict
live_features = pd.read_parquet("v5.0/live.parquet", columns=features)
# Apply feature engineering to live data
live_features_engineered = engineer_features(live_features)


def predict(live_features_pd: pd.DataFrame) -> pd.DataFrame:
    # Apply feature engineering to the input data
    live_features_eng = engineer_features(live_features_pd)

    # Make sure all required engineered features exist
    # Add any missing engineered features with zeros
    for feat in engineered_features:
        if feat not in live_features_eng.columns:
            print(f"Warning: Feature {feat} missing in live data. Adding with zeros.")
            live_features_eng[feat] = 0.0

    # Ensure features are in the same order as they were during training
    live_features_for_prediction = live_features_eng[engineered_features]

    # Generate predictions from each of our cross-validation models
    cv_predictions = []
    for fold_model in models:
        # Get predictions from this fold's model
        fold_preds = fold_model.predict(live_features_for_prediction)
        cv_predictions.append(fold_preds)

    # Get predictions from our final model
    final_preds = final_model.predict(live_features_for_prediction)

    # Create an ensemble by averaging all model predictions (including final model)
    all_predictions = cv_predictions + [final_preds]
    ensemble_predictions = np.mean(all_predictions, axis=0)

    # Convert to DataFrame with correct format
    predictions_df = pd.DataFrame(index=live_features_pd.index)
    predictions_df["prediction"] = ensemble_predictions

    # Apply feature neutralization with increased proportion
    # This helps reduce exposure to common risk factors
    neutralized_predictions = neutralize(
        df=pd.concat([live_features_pd[neutralization_features], predictions_df], axis=1),
        columns=["prediction"],
        by=neutralization_features,
        proportion=0.6,  # Increased neutralization (60%)
    )

    # Scale predictions to match example predictions distribution
    preds_mean = neutralized_predictions["prediction"].mean()
    preds_std = neutralized_predictions["prediction"].std()
    example_mean = example_preds["prediction"].mean()
    example_std = example_preds["prediction"].std()

    # Standardize then rescale to match example distribution
    neutralized_predictions["prediction"] = (
        neutralized_predictions["prediction"] - preds_mean
    ) / preds_std * example_std + example_mean

    # Apply rank-based transformation to ensure robustness
    # This method is more stable than sigmoid for financial data
    def rank_transform(x, min_val=0.001, max_val=0.999):
        # Convert to ranks (0 to 1)
        ranks = pd.Series(x).rank(pct=True)
        # Scale to desired range
        scaled = ranks * (max_val - min_val) + min_val
        return scaled.values

    # Apply rank transformation
    rank_preds = rank_transform(neutralized_predictions["prediction"])

    # Final predictions
    final_predictions = pd.DataFrame(index=live_features_pd.index)
    final_predictions["prediction"] = rank_preds

    # Ensure predictions are strictly between 0 and 1
    # Clip any potential extreme values just to be safe
    final_predictions["prediction"] = final_predictions["prediction"].clip(0.001, 0.999)

    # Ensure the predictions match the expected format
    return final_predictions


p = cloudpickle.dumps(predict)
with open(model_file_name, "wb") as f:
    f.write(p)

print(f"Model saved to {model_file_name}")
