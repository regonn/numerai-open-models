import requests
import os
import numerapi
import polars as pl
from dotenv import load_dotenv

load_dotenv()

DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
NUMERAI_PUBLIC_ID = os.environ["NUMERAI_PUBLIC_ID"]
NUMERAI_SECRET_KEY = os.environ["NUMERAI_SECRET_KEY"]
NUMERAI_MODEL_ID = os.environ["NUMERAI_MODEL_ID"]


def post_discord(message):
    requests.post(DISCORD_WEBHOOK_URL, {"content": message})


try:
    napi = numerapi.NumerAPI(NUMERAI_PUBLIC_ID, NUMERAI_SECRET_KEY)
    napi.download_dataset(
        "v4.1/live_example_preds.parquet", "live_example_preds.parquet"
    )
    pl.read_parquet("./live_example_preds.parquet").select(
        ["id", "prediction"]
    ).write_csv(f"tournament_predictions_{NUMERAI_MODEL_ID}.csv")
    model_id = napi.get_models()[NUMERAI_MODEL_ID]
    napi.upload_predictions(
        f"tournament_predictions_{NUMERAI_MODEL_ID}.csv", model_id=model_id
    )

    post_discord(f"Submit Success: {NUMERAI_MODEL_ID}")
except Exception as e:
    post_discord(f"Numerai Submit Failure. Error: {str(e)}")
