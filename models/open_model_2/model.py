import gc
import json
import os
from logging import getLogger

import lightgbm as lgb
import numerapi
import numpy as np
import pandas as pd
import psutil
import requests
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


def memory_callback(env):
    if env.iteration % 100 == 0 or env.iteration == env.end_iteration - 1:
        logger.info(f"Iteration: {env.iteration}, {memory_log_message()}")


def neutralize(df, target="prediction", by=None, proportion=1.0):
    if by is None:
        by = [x for x in df.columns if x.startswith("feature")]

    scores = df[target]
    exposures = df[by].values

    # constant column to make sure the series is completely neutral to exposures
    exposures = np.hstack((exposures, np.array([np.mean(scores)] * len(exposures)).reshape(-1, 1)))

    scores -= proportion * (exposures @ (np.linalg.pinv(exposures) @ scores.values))
    return scores / scores.std()


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

    dtrain = lgb.Dataset(train[features], label=train["target"])
    del train
    gc.collect()
    logger.info("Create LGBM Dataset")
    logger.info(memory_log_message())

    # Validation data
    validation = pd.read_parquet(
        "v5.0/validation.parquet",
        columns=["era", "data_type"] + features + ["target"],
    )
    validation = validation[validation["data_type"] == "validation"]
    logger.info(f"Loaded {len(validation)} rows of validation data")
    logger.info(memory_log_message())

    dvalid = lgb.Dataset(validation[features], label=validation["target"])
    del validation
    gc.collect()
    logger.info(memory_log_message())

    # Train model
    logger.info("Start training")
    params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "metric": "l2",
        "learning_rate": 0.001,
        "max_depth": 10,
        "num_leaves": 2**10,
        "colsample_bytree": 0.1,
        "min_date_in_leaf": 10000,
        "random_state": 46,  # sexy random seed
        "force_col_wise": True,
    }
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dvalid],
        num_boost_round=30_000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=500, verbose=True),
            memory_callback,
        ],
    )
    logger.info("end training")
    logger.info(memory_log_message())

    # Predict
    live_features = pd.read_parquet("v5.0/live.parquet", columns=features)
    live_predictions = model.predict(live_features[features], num_iteration=model.best_iteration)
    live_features["prediction"] = live_predictions

    submission = neutralize(live_features, target="prediction", by=None, proportion=1.0).rank(pct=True)

    # Submit
    submission.to_frame("prediction").to_csv(predict_csv_file_name, index=True)
    model_id = napi.get_models()[NUMERAI_MODEL_ID]
    napi.upload_predictions(predict_csv_file_name, model_id=model_id)

    post_discord(f"Submit Success: {NUMERAI_MODEL_ID}")
except Exception as e:
    post_discord(f"Numerai Submit Failure. Model: {NUMERAI_MODEL_ID}, Error: {str(e)}")
