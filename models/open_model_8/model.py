import gc
import json
import os
from logging import getLogger

import catboost as cb
import lightgbm as lgb
import numerapi
import numpy as np
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


class MemoryLoggingCallback(xgb.callback.TrainingCallback):
    def memory_log_message(self):
        used = f"{psutil.virtual_memory().used / (1024 ** 2):.2f} MB"
        total = f"{psutil.virtual_memory().total / (1024 ** 2):.2f} MB"
        return f"Memory used: {used} / {total}"

    def after_iteration(self, model, epoch, evals_log):
        if epoch % 100 == 0:
            logger.info(f"Iteration: {epoch}, {self.memory_log_message()}")


def reduce_memory_usage(df):
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == np.int64:
            df[col] = df[col].astype(np.int32)
    return df


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
    features = feature_metadata["feature_sets"]["small"]
    logger.info(f"Using {len(features)} features")
    logger.info(memory_log_message())

    # Train data
    train = pd.read_parquet("v5.0/train.parquet", columns=["era"] + features + ["target"])
    train = reduce_memory_usage(train)
    logger.info(f"Loaded {len(train)} rows of training data")
    logger.info(memory_log_message())

    # Create datasets for LightGBM, XGBoost, and CatBoost
    dtrain_lgb = lgb.Dataset(train[features], label=train["target"])
    dtrain_xgb = xgb.DMatrix(train[features], label=train["target"])
    dtrain_cb = cb.Pool(train[features], label=train["target"])

    del train
    gc.collect()
    logger.info("Created datasets for LightGBM, XGBoost, and CatBoost")
    logger.info(memory_log_message())

    # Validation data
    validation = pd.read_parquet(
        "v5.0/validation.parquet",
        columns=["era", "data_type"] + features + ["target"],
    )
    validation = validation[validation["data_type"] == "validation"]
    validation = reduce_memory_usage(validation)
    logger.info(f"Loaded {len(validation)} rows of validation data")
    logger.info(memory_log_message())

    # Create datasets for LightGBM, XGBoost, and CatBoost
    dvalid_lgb = lgb.Dataset(validation[features], label=validation["target"])
    dvalid_xgb = xgb.DMatrix(validation[features], label=validation["target"])
    dvalid_cb = cb.Pool(validation[features], label=validation["target"])

    del validation
    gc.collect()
    logger.info(memory_log_message())

    # Define parameters
    params_lgb = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "metric": "l2",
        "learning_rate": 0.005,
        "max_depth": 6,
        "num_leaves": 2**6 - 1,
        "colsample_bytree": 0.1,
        "random_state": 46,  # sexy random seed
        "force_col_wise": True,
    }

    params_xgb = {
        "objective": "reg:squarederror",
        "learning_rate": 0.005,
        "max_depth": 6,
        "colsample_bytree": 0.1,
        "seed": 46,
    }

    params_cb = {
        "loss_function": "RMSE",
        "learning_rate": 0.005,
        "depth": 6,
        "random_seed": 46,
        "verbose": False,
    }

    # Train models
    logger.info("Start training LightGBM")
    model_lgb = lgb.train(
        params_lgb,
        dtrain_lgb,
        valid_sets=[dvalid_lgb],
        num_boost_round=3000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            memory_callback,
        ],
    )
    logger.info("Finished training LightGBM")
    logger.info(memory_log_message())

    logger.info("Start training XGBoost")
    model_xgb = xgb.train(
        params_xgb,
        dtrain_xgb,
        num_boost_round=3000,
        evals=[(dvalid_xgb, "validation")],
        callbacks=[
            xgb.callback.EarlyStopping(
                rounds=100,
                save_best=True,
                maximize=False,
                metric_name="rmse",
                data_name="validation",
            ),
            MemoryLoggingCallback(),
        ],
    )
    logger.info("Finished training XGBoost")
    logger.info(memory_log_message())

    logger.info("Start training CatBoost")
    model_cb = cb.CatBoostRegressor(
        iterations=3000,
        **params_cb,
    )
    model_cb.fit(
        dtrain_cb,
        eval_set=dvalid_cb,
        early_stopping_rounds=100,
        use_best_model=True,
        verbose=100,
    )
    logger.info("Finished training CatBoost")
    logger.info(memory_log_message())

    # Predict
    live_features = pd.read_parquet("v5.0/live.parquet", columns=features)
    live_features = reduce_memory_usage(live_features)

    dlive_xgb = xgb.DMatrix(live_features[features])
    dlive_cb = cb.Pool(live_features[features])

    preds_lgb = model_lgb.predict(live_features[features], num_iteration=model_lgb.best_iteration)
    preds_xgb = model_xgb.predict(dlive_xgb, iteration_range=(0, model_xgb.best_iteration + 1))
    preds_cb = model_cb.predict(dlive_cb)

    # Ensemble predictions
    live_features["prediction"] = (preds_lgb + preds_xgb + preds_cb) / 3
    submission = neutralize(live_features, target="prediction", by=None, proportion=1.0).rank(pct=True)

    # Submit
    submission.to_frame("prediction").to_csv(predict_csv_file_name, index=True)
    model_id = napi.get_models()[NUMERAI_MODEL_ID]
    napi.upload_predictions(predict_csv_file_name, model_id=model_id)

    post_discord(f"Submit Success: {NUMERAI_MODEL_ID}")
except Exception as e:
    post_discord(f"Numerai Submit Failure. Model: {NUMERAI_MODEL_ID}, Error: {str(e)}")
