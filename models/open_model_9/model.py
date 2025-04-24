import gc
import json
import os

import cloudpickle
import lightgbm as lgb
import numerapi
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

load_dotenv()

DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
NUMERAI_PUBLIC_ID = os.environ["NUMERAI_PUBLIC_ID"]
NUMERAI_SECRET_KEY = os.environ["NUMERAI_SECRET_KEY"]
NUMERAI_MODEL_ID = "open_model_9"
napi = numerapi.NumerAPI(NUMERAI_PUBLIC_ID, NUMERAI_SECRET_KEY)

# Download data
napi.download_dataset("v5.0/train.parquet")
napi.download_dataset("v5.0/validation.parquet")
napi.download_dataset("v5.0/live.parquet")
napi.download_dataset("v5.0/features.json")
napi.download_dataset("v5.0/live_example_preds.parquet")

model_file_name = f"models/{NUMERAI_MODEL_ID}/predict_{NUMERAI_MODEL_ID}.pkl"
feature_metadata = json.load(open("v5.0/features.json"))
features = feature_metadata["feature_sets"]["medium"]

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

# Validation data
validation = pd.read_parquet(
    "v5.0/validation.parquet",
    columns=["era", "data_type"] + features + [target],
)
validation = validation[validation["data_type"] == "validation"]
validation = validation.dropna(subset=[target], axis=0)

# Get example predictions for checking format
example_preds = pd.read_parquet("v5.0/live_example_preds.parquet")

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


# Train model with improved parameters
model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=6,  # Slightly deeper trees
    num_leaves=2**6 - 1,  # More leaves for more complexity
    colsample_bytree=0.1,
    subsample=0.8,  # Add subsampling for better generalization
    reg_alpha=0.1,  # L1 regularization to reduce overfitting
    reg_lambda=1.0,  # L2 regularization to reduce overfitting
    random_state=42,  # For reproducibility
)

# This will take a few minutes 🍵
print("Training model...")
model.fit(
    train[features],
    train[target],
    eval_set=[(validation[features], validation[target])],
    eval_metric="mae",
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
    ],
)

print(f"Best iteration: {model.best_iteration_}")

# Predict
live_features = pd.read_parquet("v5.0/live.parquet", columns=features)


def predict(live_features_pd: pd.DataFrame) -> pd.DataFrame:
    # Generate raw predictions
    live_predictions = model.predict(live_features_pd[features])

    # Convert to DataFrame with correct format
    predictions_df = pd.DataFrame(index=live_features_pd.index)
    predictions_df["prediction"] = live_predictions

    # Apply feature neutralization to reduce exposure to common risk factors
    neutralized_predictions = neutralize(
        df=pd.concat([live_features_pd[neutralization_features], predictions_df], axis=1),
        columns=["prediction"],
        by=neutralization_features,
        proportion=0.5,  # Partial neutralization (50%)
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

    # Apply custom sigmoid transformation to ensure predictions are between 0 and 1
    # Define sigmoid function directly instead of importing from scipy
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # Normalize to a good range for sigmoid
    preds = neutralized_predictions["prediction"].values
    preds_standardized = (preds - np.mean(preds)) / np.std(preds)

    # Apply sigmoid to constrain values between 0 and 1
    preds_sigmoid = sigmoid(preds_standardized)

    # Final predictions
    final_predictions = pd.DataFrame(index=live_features_pd.index)
    final_predictions["prediction"] = preds_sigmoid

    # Ensure predictions are strictly between 0 and 1
    # Clip any potential extreme values just to be safe
    final_predictions["prediction"] = final_predictions["prediction"].clip(0.001, 0.999)

    # Ensure the predictions match the expected format
    return final_predictions


p = cloudpickle.dumps(predict)
with open(model_file_name, "wb") as f:
    f.write(p)

print(f"Model saved to {model_file_name}")
