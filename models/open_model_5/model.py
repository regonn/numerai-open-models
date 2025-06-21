import gc
import json
import os
import sys
import traceback
from logging import getLogger

import cloudpickle
import numerapi
import pandas as pd
import psutil
import requests
import xgboost as xgb
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2.service_account import Credentials

logger = getLogger(__name__)

load_dotenv()

DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
NUMERAI_PUBLIC_ID = os.environ["NUMERAI_PUBLIC_ID"]
NUMERAI_SECRET_KEY = os.environ["NUMERAI_SECRET_KEY"]
NUMERAI_MODEL_ID = "open_model_5"


def post_discord(message):
    """Send message to Discord webhook"""
    try:
        requests.post(DISCORD_WEBHOOK_URL, {"content": message})
    except Exception as e:
        print(f"Failed to send Discord notification: {e}")


def get_error_details():
    """Get detailed error information including type, message, and location"""
    exc_type, exc_value, exc_traceback = sys.exc_info()

    if exc_type is None:
        return "No error information available"

    # Get error type and message
    error_type = exc_type.__name__
    error_message = str(exc_value)

    # Get the most recent traceback frame (where the error occurred)
    if exc_traceback:
        tb_list = traceback.extract_tb(exc_traceback)
        if tb_list:
            # Get the last frame (most recent)
            last_frame = tb_list[-1]
            error_location = f"{last_frame.filename}:{last_frame.lineno} in {last_frame.name}"
        else:
            error_location = "Unknown location"
    else:
        error_location = "Unknown location"

    # Get full traceback for debugging
    full_traceback = traceback.format_exc()

    return {"type": error_type, "message": error_message, "location": error_location, "full_traceback": full_traceback}


def memory_log_message():
    used = f"{psutil.virtual_memory().used / (1024**2):.2f} MB"
    total = f"{psutil.virtual_memory().total / (1024**2):.2f} MB"
    return f"Memory used: {used} / {total}"


# class MemoryLoggingCallback(xgb.callback.TrainingCallback):
#     def memory_log_message(self):
#         used = f"{psutil.virtual_memory().used / (1024 ** 2):.2f} MB"
#         total = f"{psutil.virtual_memory().total / (1024 ** 2):.2f} MB"
#         return f"Memory used: {used} / {total}"

#     def after_iteration(self, model, epoch, evals_log):
#         if epoch % 10 == 0:
#             logger.info(f"Iteration: {epoch}, {self.memory_log_message()}")


# Main execution with error handling
try:
    # Send start notification
    start_message = f"🚀 **{NUMERAI_MODEL_ID}** training process started"
    post_discord(start_message)
    print(start_message)

    napi = numerapi.NumerAPI(NUMERAI_PUBLIC_ID, NUMERAI_SECRET_KEY)

    # GCP credentials setup
    post_discord(f"🔧 Setting up GCP credentials for {NUMERAI_MODEL_ID}...")
    cred = Credentials.from_service_account_info(
        {
            "type": "service_account",
            "project_id": os.environ["GCP_PROJECT_ID"],
            "private_key_id": os.environ["GCS_PRIVATE_KEY_ID"],
            "private_key": os.environ["GCS_PRIVATE_KEY"].replace("\\n", "\n"),
            "client_email": os.environ["GCS_CLIENT_MAIL"],
            "client_id": os.environ["GCS_CLIENT_ID"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.environ["GCS_CLIENT_X509_CERT_URL"],
        }
    )
    client = storage.Client(credentials=cred)
    bucket = client.get_bucket(os.environ["GCS_BUCKET_NAME"])

    # Download data
    post_discord(f"📥 Downloading datasets for {NUMERAI_MODEL_ID}...")
    napi.download_dataset("v5.0/train.parquet")
    napi.download_dataset("v5.0/validation.parquet")
    napi.download_dataset("v5.0/live.parquet")
    napi.download_dataset("v5.0/features.json")
    napi.download_dataset("v5.0/live_example_preds.parquet")
    predict_csv_file_name = f"tournament_predictions_{NUMERAI_MODEL_ID}.csv"
    model_file_name = f"./predict_{NUMERAI_MODEL_ID}.pkl"

    # Load data
    feature_metadata = json.load(open("v5.0/features.json"))
    features = feature_metadata["feature_sets"]["small"]
    targets = ["target_bravo_20", "target_bravo_60"]
    logger.info(f"Using {len(features)} features")
    logger.info(memory_log_message())
    post_discord(f"📊 Using {len(features)} features for {NUMERAI_MODEL_ID}")

    # Train data
    post_discord(f"📊 Loading training data for {NUMERAI_MODEL_ID}...")
    train = pd.read_parquet("v5.0/train.parquet", columns=["era"] + features + targets)
    train = train.dropna(subset=targets, axis=0)
    logger.info(f"Loaded {len(train)} rows of training data")
    logger.info(memory_log_message())

    dtrain = xgb.DMatrix(train[features], train[targets])
    del train
    gc.collect()
    logger.info("Create Xgboost Dataset")
    logger.info(memory_log_message())

    # Validation data
    post_discord(f"📊 Loading validation data for {NUMERAI_MODEL_ID}...")
    validation = pd.read_parquet(
        "v5.0/validation.parquet",
        columns=["era", "data_type"] + features + targets,
    )
    validation = validation[validation["data_type"] == "validation"]
    logger.info(f"Loaded {len(validation)} rows of validation data")
    logger.info(memory_log_message())

    validation = validation.dropna(subset=targets, axis=0)

    dvalid = xgb.DMatrix(validation[features], validation[targets])
    del validation
    gc.collect()
    logger.info("Create Xgboost Dataset")
    logger.info(memory_log_message())

    # Train model
    post_discord(f"🤖 Starting model training for {NUMERAI_MODEL_ID}...")
    logger.info("Start training")
    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "tree_method": "hist",
        "multi_strategy": "multi_output_tree",
        "eval_metric": "rmse",
        "seed": 46,  # sexy random seed
    }

    # early_stop = xgb.callback.EarlyStopping(
    #     rounds=100,
    #     save_best=True,
    #     maximize=False,
    #     metric_name="rmse",
    #     data_name="eval",
    # )

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=[(dvalid, "eval")],
        verbose_eval=False,
        # callbacks=[early_stop, MemoryLoggingCallback()],
    )
    logger.info("end training")
    logger.info(memory_log_message())
    post_discord(f"✅ Model training completed for {NUMERAI_MODEL_ID}")

    # Get example predictions for checking format
    example_preds = pd.read_parquet("v5.0/live_example_preds.parquet")

    # Create predict function
    def create_predict_function():
        """Create a self-contained predict function with all necessary state"""
        # Capture the current state in the closure
        captured_model = model
        captured_features = features.copy()
        captured_targets = targets.copy()
        captured_example_preds = example_preds.copy()

        def self_contained_predict(live_features_pd: pd.DataFrame) -> pd.DataFrame:
            """Self-contained predict function that works on any server"""
            # Use captured state instead of global variables
            model_features = captured_features

            # Ensure we have the required features
            live_features_for_prediction = live_features_pd[model_features]

            # Create XGBoost DMatrix
            dlive = xgb.DMatrix(live_features_for_prediction)

            # Make predictions (multi-target)
            predictions = captured_model.predict(dlive)

            # For multi-target models, we use the first target (target_bravo_20)
            # This matches the original logic where we used predictions[:, 0]
            single_target_predictions = predictions[:, 0]

            # Convert to DataFrame with correct format
            predictions_df = pd.DataFrame(index=live_features_pd.index)
            predictions_df["prediction"] = single_target_predictions

            # Apply rank-based transformation to ensure robustness
            def rank_transform(x, min_val=0.001, max_val=0.999):
                # Convert to ranks (0 to 1)
                ranks = pd.Series(x).rank(pct=True)
                # Scale to desired range
                scaled = ranks * (max_val - min_val) + min_val
                return scaled.values

            # Apply rank transformation
            rank_preds = rank_transform(predictions_df["prediction"])

            # Final predictions
            final_predictions = pd.DataFrame(index=live_features_pd.index)
            final_predictions["prediction"] = rank_preds

            # Ensure predictions are strictly between 0 and 1
            final_predictions["prediction"] = final_predictions["prediction"].clip(0.001, 0.999)

            return final_predictions

        return self_contained_predict

    # Create the predict function
    predict = create_predict_function()

    # Save model
    post_discord(f"💾 Saving model for {NUMERAI_MODEL_ID}...")
    p = cloudpickle.dumps(predict)
    with open(model_file_name, "wb") as f:
        f.write(p)

    print(f"Model saved to {model_file_name}")

    # Upload to Cloud Storage
    post_discord(f"☁️ Uploading model to GCS for {NUMERAI_MODEL_ID}...")
    gcs_file_name = f"predict_{NUMERAI_MODEL_ID}.pkl"
    blob = bucket.blob(gcs_file_name)
    blob.upload_from_filename(model_file_name)
    print(f"Model uploaded to GCS: {gcs_file_name}")

    # Send success notification
    success_message = f"✅ **{NUMERAI_MODEL_ID}** training completed successfully!\n💾 Model saved and uploaded to GCS"
    post_discord(success_message)
    print(success_message)

except Exception:
    # Get detailed error information
    error_details = get_error_details()

    # Create detailed error message
    error_message = f"""❌ **{NUMERAI_MODEL_ID}** training failed!

**Error Type:** {error_details["type"]}
**Error Message:** {error_details["message"]}
**Error Location:** {error_details["location"]}

**Full Traceback:**
```
{error_details["full_traceback"]}
```"""

    # Send error notification to Discord
    post_discord(error_message)

    # Also print to console
    print(f"Error occurred: {error_details['type']}: {error_details['message']}")
    print(f"Location: {error_details['location']}")
    print(f"Full traceback:\n{error_details['full_traceback']}")

    # Re-raise the exception to maintain the original error handling
    raise
