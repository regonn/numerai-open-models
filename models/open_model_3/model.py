import gc
import json
import os
from logging import getLogger

import numerapi
import pandas as pd
import psutil
import requests
import xgboost as xgb
from dotenv import load_dotenv

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


class MemoryLoggingCallback(xgb.callback.TrainingCallback):
    def memory_log_message(self):
        used = f"{psutil.virtual_memory().used / (1024 ** 2):.2f} MB"
        total = f"{psutil.virtual_memory().total / (1024 ** 2):.2f} MB"
        return f"Memory used: {used} / {total}"

    def after_iteration(self, model, epoch, evals_log):
        if epoch % 100 == 0:
            logger.info(f"Iteration: {epoch}, {self.memory_log_message()}")


post_discord(f"Start: {NUMERAI_MODEL_ID}")
try:
    napi = numerapi.NumerAPI(NUMERAI_PUBLIC_ID, NUMERAI_SECRET_KEY)

    # Download data
    napi.download_dataset("v5.0/train.parquet")
    napi.download_dataset("v5.0/validation.parquet")
    napi.download_dataset("v5.0/live.parquet")
    napi.download_dataset("v5.0/features.json")
    predict_csv_file_name = f"tournament_predictions_{NUMERAI_MODEL_ID}.csv"

    # Load data
    feature_metadata = json.load(open("v5.0/features.json"))
    features = feature_metadata["feature_sets"]["medium"]
    logger.info(f"Using {len(features)} features")
    logger.info(memory_log_message())

    # Train data
    train = pd.read_parquet("v5.0/train.parquet", columns=["era"] + features + ["target"])
    logger.info(f"Loaded {len(train)} rows of training data")
    logger.info(memory_log_message())

    dtrain = xgb.DMatrix(train[features], label=train["target"])
    del train
    gc.collect()
    logger.info("Create Xgboost Dataset")
    logger.info(memory_log_message())

    # Validation data
    validation = pd.read_parquet(
        "v5.0/validation.parquet",
        columns=["era", "data_type"] + features + ["target"],
    )
    validation = validation[validation["data_type"] == "validation"]
    logger.info(f"Loaded {len(validation)} rows of validation data")
    logger.info(memory_log_message())

    dvalid = xgb.DMatrix(validation[features], label=validation["target"])
    del validation
    gc.collect()
    logger.info("Create Xgboost Dataset")
    logger.info(memory_log_message())

    # Train model
    logger.info("Start training")
    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.02,
        "max_depth": 5,
        "subsample": 1.0 / 16,
        "colsample_bytree": (2**5) / len(features),  # features needs to be defined
        "eval_metric": "rmse",
        "seed": 46,  # sexy random seed
    }

    early_stop = xgb.callback.EarlyStopping(
        rounds=100,
        save_best=True,
        maximize=False,
        metric_name="rmse",
        data_name="eval",
    )

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dvalid, "eval")],
        verbose_eval=False,
        callbacks=[early_stop, MemoryLoggingCallback()],
    )
    logger.info("end training")
    logger.info(memory_log_message())

    # Predict
    live_features = pd.read_parquet("v5.0/live.parquet", columns=features)
    dlive = xgb.DMatrix(live_features[features])
    live_predictions = model.predict(dlive, iteration_range=(0, model.best_iteration + 1))

    # Submit
    submission = pd.Series(live_predictions, index=live_features.index).rank(pct=True)
    submission.to_frame("prediction").to_csv(predict_csv_file_name, index=True)
    model_id = napi.get_models()[NUMERAI_MODEL_ID]
    napi.upload_predictions(predict_csv_file_name, model_id=model_id)

    post_discord(f"Submit Success: {NUMERAI_MODEL_ID}")
except Exception as e:
    post_discord(f"Numerai Submit Failure. Model: {NUMERAI_MODEL_ID}, Error: {str(e)}")
