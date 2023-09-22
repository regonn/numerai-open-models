import requests
import os
import numerapi
from dotenv import load_dotenv
import pandas as pd
import json
import lightgbm as lgb
import gc
from logging import getLogger
import psutil

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


def memory_callback(env):
    if env.iteration % 100 == 0 or env.iteration == env.end_iteration - 1:
        logger.info(f"Iteration: {env.iteration}, {memory_log_message()}")


post_discord(f"Start: {NUMERAI_MODEL_ID}")
try:
    napi = numerapi.NumerAPI(NUMERAI_PUBLIC_ID, NUMERAI_SECRET_KEY)

    # Download data
    napi.download_dataset("v4.2/train_int8.parquet")
    napi.download_dataset("v4.2/validation_int8.parquet")
    napi.download_dataset("v4.2/live_int8.parquet")
    napi.download_dataset("v4.2/features.json")
    predict_csv_file_name = f"tournament_predictions_{NUMERAI_MODEL_ID}.csv"

    # Load data
    feature_metadata = json.load(open("v4.2/features.json"))
    features = feature_metadata["feature_sets"]["all"]
    logger.info(f"Using {len(features)} features")
    logger.info(memory_log_message())

    # Train data
    train = pd.read_parquet(
        "v4.2/train_int8.parquet", columns=["era"] + features + ["target"]
    )
    logger.info(f"Loaded {len(train)} rows of training data")
    logger.info(memory_log_message())

    dtrain = lgb.Dataset(train[features], label=train["target"])
    del train
    gc.collect()
    logger.info("Create LGBM Dataset")
    logger.info(memory_log_message())

    # Validation data
    validation = pd.read_parquet(
        "v4.2/validation_int8.parquet",
        columns=["era", "data_type"] + features + ["target"],
    )
    validation = validation[validation["data_type"] == "validation"]
    logger.info(f"Loaded {len(validation)} rows of validation data")
    logger.info(memory_log_message())

    dvalid = lgb.Dataset(validation[features], label=validation["target"])
    del validation
    gc.collect()
    logger.info("Create LGBM Dataset")
    logger.info(memory_log_message())

    # Train model
    logger.info("Start training")
    params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "metric": "l2",
        "learning_rate": 0.004,
        "max_depth": 6,
        "num_leaves": 2**6 - 1,
        "colsample_bytree": 0.1,
        "random_state": 46,  # sexy random seed
        "force_col_wise": True,
    }
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dvalid],
        num_boost_round=3000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            memory_callback,
        ],
    )
    logger.info("end training")
    logger.info(memory_log_message())

    # Predict
    live_features = pd.read_parquet("v4.2/live_int8.parquet", columns=features)
    live_predictions = model.predict(live_features[features])

    # Submit
    submission = pd.Series(live_predictions, index=live_features.index)
    submission.to_frame("prediction").to_csv(predict_csv_file_name, index=True)
    model_id = napi.get_models()[NUMERAI_MODEL_ID]
    napi.upload_predictions(predict_csv_file_name, model_id=model_id)

    post_discord(f"Submit Success: {NUMERAI_MODEL_ID}")
except Exception as e:
    post_discord(f"Numerai Submit Failure. Model: {NUMERAI_MODEL_ID}, Error: {str(e)}")
