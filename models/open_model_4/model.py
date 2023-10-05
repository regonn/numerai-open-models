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
import numpy as np
import scipy.stats
import optuna
from google.oauth2.service_account import Credentials
from google.cloud import storage
import pickle


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


def numerai_corr(preds, target):
    ranked_preds = (preds.rank(method="average").values - 0.5) / preds.count()
    gauss_ranked_preds = scipy.stats.norm.ppf(ranked_preds)
    centered_target = target - target.mean()
    preds_p15 = np.sign(gauss_ranked_preds) * np.abs(gauss_ranked_preds) ** 1.5
    target_p15 = np.sign(centered_target) * np.abs(centered_target) ** 1.5
    return np.corrcoef(preds_p15, target_p15)[0, 1]


def objective(trial):
    param = {
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "feature_pre_filter": False,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    gbm = lgb.train(
        param,
        dtrain,
        valid_sets=[dvalid],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
        ],
    )

    logger.info(f"best_iteration: {gbm.best_iteration}")
    preds = gbm.predict(validation[features], num_iteration=gbm.best_iteration)
    target = validation["target"]

    score = numerai_corr(pd.Series(preds, index=validation.index), target)
    return score


def optuna_callback(study, trial):
    logger.info(f"Trial {trial.number}, {memory_log_message()}, Value: {trial.value}")
    if study.best_trial.number == trial.number:
        logger.info(f"Update Best Params: {trial.params}")


post_discord(f"Start: {NUMERAI_MODEL_ID}")
try:
    napi = numerapi.NumerAPI(NUMERAI_PUBLIC_ID, NUMERAI_SECRET_KEY)

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
    blob = bucket.blob("open_model_4_study.pkl")

    # Download data
    napi.download_dataset("v4.2/train_int8.parquet")
    napi.download_dataset("v4.2/validation_int8.parquet")
    napi.download_dataset("v4.2/live_int8.parquet")
    napi.download_dataset("v4.2/features.json")
    predict_csv_file_name = f"tournament_predictions_{NUMERAI_MODEL_ID}.csv"

    # Load data
    feature_metadata = json.load(open("v4.2/features.json"))
    features = feature_metadata["feature_sets"]["medium"]
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

    gc.collect()
    logger.info("Create LGBM Dataset")
    logger.info(memory_log_message())

    # Train model
    logger.info("Start training")
    try:
        blob_data = blob.download_as_bytes()
        study = pickle.loads(blob_data)
        logger.info("Study loaded from GCS.")
    except Exception as e:
        logger.info(f"Could not load study from GCS: {e}")
        study = optuna.create_study(direction="maximize")
        logger.info("New study created.")

    study.optimize(objective, n_trials=5, callbacks=[optuna_callback])

    study_data = pickle.dumps(study)
    blob.upload_from_string(study_data)

    best_params = study.best_params
    best_params["objective"] = "regression"
    best_params["metric"] = "l2"
    best_params["verbosity"] = -1
    best_params["boosting_type"] = "gbdt"
    best_params["feature_pre_filter"] = False
    model = lgb.train(
        best_params,
        dtrain,
        valid_sets=[dvalid],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
        ],
    )
    logger.info("end training")
    logger.info(memory_log_message())

    # Predict
    live_features = pd.read_parquet("v4.2/live_int8.parquet", columns=features)
    live_predictions = model.predict(live_features[features])

    # Submit
    submission = pd.Series(live_predictions, index=live_features.index).rank(pct=True)
    submission.to_frame("prediction").to_csv(predict_csv_file_name, index=True)
    model_id = napi.get_models()[NUMERAI_MODEL_ID]
    napi.upload_predictions(predict_csv_file_name, model_id=model_id)

    post_discord(f"Submit Success: {NUMERAI_MODEL_ID}")
except Exception as e:
    post_discord(f"Numerai Submit Failure. Model: {NUMERAI_MODEL_ID}, Error: {str(e)}")
