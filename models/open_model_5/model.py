import requests
import os
import numerapi
from dotenv import load_dotenv
import pandas as pd
import json
import gc
from logging import getLogger
import psutil
import xgboost as xgb

logger = getLogger(__name__)

load_dotenv()

DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
NUMERAI_PUBLIC_ID = os.environ["NUMERAI_PUBLIC_ID"]
NUMERAI_SECRET_KEY = os.environ["NUMERAI_SECRET_KEY"]
NUMERAI_MODEL_ID = os.environ["NUMERAI_MODEL_ID"]


def post_discord(message):
    requests.post(DISCORD_WEBHOOK_URL, {"content": message})


def memory_log_message():
    used = f"{psutil.virtual_memory().used / (1024 ** 2):.2f} MB"
    total = f"{psutil.virtual_memory().total / (1024 ** 2):.2f} MB"
    return f"Memory used: {used} / {total}"


# class MemoryLoggingCallback(xgb.callback.TrainingCallback):
#     def memory_log_message(self):
#         used = f"{psutil.virtual_memory().used / (1024 ** 2):.2f} MB"
#         total = f"{psutil.virtual_memory().total / (1024 ** 2):.2f} MB"
#         return f"Memory used: {used} / {total}"

#     def after_iteration(self, model, epoch, evals_log):
#         if epoch % 10 == 0:
#             logger.info(f"Iteration: {epoch}, {self.memory_log_message()}")


post_discord(f"Start: {NUMERAI_MODEL_ID}")
try:
    napi = numerapi.NumerAPI(NUMERAI_PUBLIC_ID, NUMERAI_SECRET_KEY)

    # Download data
    napi.download_dataset("v4.3/train_int8.parquet")
    napi.download_dataset("v4.3/validation_int8.parquet")
    napi.download_dataset("v4.3/live_int8.parquet")
    napi.download_dataset("v4.3/features.json")
    predict_csv_file_name = f"tournament_predictions_{NUMERAI_MODEL_ID}.csv"

    # Load data
    feature_metadata = json.load(open("v4.3/features.json"))
    features = feature_metadata["feature_sets"]["small"]
    targets = ["target_cyrus_v4_20", "target_cyrus_v4_60"]
    logger.info(f"Using {len(features)} features")
    logger.info(memory_log_message())

    # Train data
    train = pd.read_parquet(
        "v4.3/train_int8.parquet", columns=["era"] + features + targets
    )
    train = train.dropna(subset=targets, axis=0)
    logger.info(f"Loaded {len(train)} rows of training data")
    logger.info(memory_log_message())

    dtrain = xgb.DMatrix(train[features], train[targets])
    del train
    gc.collect()
    logger.info("Create Xgboost Dataset")
    logger.info(memory_log_message())

    # Validation data
    validation = pd.read_parquet(
        "v4.3/validation_int8.parquet",
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

    # Predict
    live_features = pd.read_parquet("v4.3/live_int8.parquet", columns=features)
    dlive = xgb.DMatrix(live_features[features])
    # live_predictions = model.predict(
    #     dlive, iteration_range=(0, model.best_iteration + 1)
    # )
    live_predictions = model.predict(dlive)

    # Submit
    submission = pd.Series(live_predictions[:, 0], index=live_features.index).rank(
        pct=True
    )
    submission.to_frame("prediction").to_csv(predict_csv_file_name, index=True)
    model_id = napi.get_models()[NUMERAI_MODEL_ID]
    napi.upload_predictions(predict_csv_file_name, model_id=model_id)

    post_discord(f"Submit Success: {NUMERAI_MODEL_ID}")
except Exception as e:
    post_discord(f"Numerai Submit Failure. Model: {NUMERAI_MODEL_ID}, Error: {str(e)}")
