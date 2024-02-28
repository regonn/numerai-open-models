import os
import numerapi
from dotenv import load_dotenv
import pandas as pd
import json
import gc
import cloudpickle
import lightgbm as lgb

load_dotenv()

DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
NUMERAI_PUBLIC_ID = os.environ["NUMERAI_PUBLIC_ID"]
NUMERAI_SECRET_KEY = os.environ["NUMERAI_SECRET_KEY"]
NUMERAI_MODEL_ID = "open_model_6"
napi = numerapi.NumerAPI(NUMERAI_PUBLIC_ID, NUMERAI_SECRET_KEY)

# Download data
# napi.download_dataset("v4.3/train_int8.parquet")
# napi.download_dataset("v4.3/validation_int8.parquet")
# napi.download_dataset("v4.3/live_int8.parquet")
# napi.download_dataset("v4.3/features.json")
model_file_name = f"models/{NUMERAI_MODEL_ID}/predict_{NUMERAI_MODEL_ID}.pkl"

# Load data
feature_metadata = json.load(open("v4.3/features.json"))
features = feature_metadata["feature_sets"]["small"]
target = "target"

# Train data
train = pd.read_parquet(
    "v4.3/train_int8.parquet", columns=["era"] + features + [target]
)
train = train.dropna(subset=target, axis=0)

# Validation data
# validation = pd.read_parquet(
#     "v4.2/validation_int8.parquet",
#     columns=["era", "data_type"] + features + targets,
# )
# validation = validation[validation["data_type"] == "validation"]
# validation = validation.dropna(subset=targets, axis=0)

gc.collect()


# Train model
model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=5,
    num_leaves=2**5 - 1,
    colsample_bytree=0.1,
)

# This will take a few minutes 🍵
model.fit(train[features], train["target"])
# Predict
live_features = pd.read_parquet("v4.3/live_int8.parquet", columns=features)


def predict(live_features_pd: pd.DataFrame) -> pd.DataFrame:
    live_predictions = model.predict(live_features_pd[features])
    submission = pd.Series(list(live_predictions), index=live_features_pd.index)
    return submission.to_frame("prediction")


p = cloudpickle.dumps(predict)
with open(model_file_name, "wb") as f:
    f.write(p)
