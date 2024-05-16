import requests
import os
import numerapi
from dotenv import load_dotenv
import pandas as pd
import json
import gc
from logging import getLogger
import psutil
from pytorch_tabular import TabularModel
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.models import (
    TabNetModelConfig,
)
from pytorch_tabular.models.common.heads import LinearHeadConfig

logger = getLogger(__name__)

load_dotenv()

DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
NUMERAI_PUBLIC_ID = os.environ["NUMERAI_PUBLIC_ID"]
NUMERAI_SECRET_KEY = os.environ["NUMERAI_SECRET_KEY"]
NUMERAI_MODEL_ID = os.environ["NUMERAI_MODEL_ID"]


def memory_log_message():
    used = f"{psutil.virtual_memory().used / (1024 ** 2):.2f} MB"
    total = f"{psutil.virtual_memory().total / (1024 ** 2):.2f} MB"
    return f"Memory used: {used} / {total}"


def post_discord(message):
    requests.post(DISCORD_WEBHOOK_URL, {"content": message})


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
    targets = ["target"]
    logger.info(f"Using {len(features)} features")
    logger.info(memory_log_message())

    # Model Data
    head_config = LinearHeadConfig(
        layers="",  # No additional layer in head, just a mapping layer to output_dim
        dropout=0.1,
        initialization="kaiming",
    ).__dict__

    model_config = TabNetModelConfig(
        task="regression",
        learning_rate=1e-3,
        head="LinearHead",  # Linear Head
        head_config=head_config,  # Linear Head Config
    )

    data_config = DataConfig(
        target=targets,  # target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
        continuous_cols=features,
        categorical_cols=[],
    )

    trainer_config = TrainerConfig(
        auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
        batch_size=2048,
        max_epochs=10,
        accelerator="cpu",
    )
    optimizer_config = OptimizerConfig()

    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )

    # Train data
    train = pd.read_parquet(
        "v4.3/train_int8.parquet", columns=["era"] + features + targets
    )
    train = train.dropna(subset=targets, axis=0)
    logger.info(f"Loaded {len(train)} rows of training data")
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

    model.fit(train=train, validation=validation)
    logger.info("end training")
    logger.info(memory_log_message())

    # Predict
    live_features = pd.read_parquet("v4.3/live_int8.parquet", columns=features)
    prediction = model.predict(live_features, include_input_features=False).rank(
        pct=True
    )

    # Submit
    prediction.rename(columns={"target_prediction": "prediction"}).to_csv(
        predict_csv_file_name, index=True
    )
    model_id = napi.get_models()[NUMERAI_MODEL_ID]
    napi.upload_predictions(predict_csv_file_name, model_id=model_id)

    post_discord(f"Submit Success: {NUMERAI_MODEL_ID}")
except Exception as e:
    post_discord(f"Numerai Submit Failure. Model: {NUMERAI_MODEL_ID}, Error: {str(e)}")
