# %%
import gc
import os
from datetime import datetime, timedelta
from logging import getLogger

import numpy as np
import pandas as pd
import psutil
import requests
import talib
import xgboost as xgb
import yfinance as yf
from dotenv import load_dotenv
from numerapi import CryptoAPI
from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt
# import seaborn as sns

logger = getLogger(__name__)

load_dotenv()

DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
NUMERAI_PUBLIC_ID = os.environ["NUMERAI_PUBLIC_ID"]
NUMERAI_SECRET_KEY = os.environ["NUMERAI_SECRET_KEY"]
NUMERAI_MODEL_ID = "crypto_open_model_1"
TARGET_NAME = "target"
SUBMISSION_COLUMNS = ["symbol"]
PREDICTION_NAME = "signal"


def post_discord(message):
    requests.post(DISCORD_WEBHOOK_URL, {"content": message})


def memory_log_message():
    used = f"{psutil.virtual_memory().used / (1024 ** 2):.2f} MB"
    total = f"{psutil.virtual_memory().total / (1024 ** 2):.2f} MB"
    return f"Memory used: {used} / {total}"


def memory_callback(env):
    if env.iteration % 100 == 0 or env.iteration == env.end_iteration - 1:
        logger.info(f"Iteration: {env.iteration}, {memory_log_message()}")


def volatility(prices, window=5):
    return prices.pct_change().rolling(window).std()


def sr(prices, window):
    ret = prices.pct_change().rolling(window).mean()
    # 安全資産リターンを引くようにする
    return (ret - 0) / volatility(prices, window)


class MemoryLoggingCallback(xgb.callback.TrainingCallback):
    def memory_log_message(self):
        used = f"{psutil.virtual_memory().used / (1024 ** 2):.2f} MB"
        total = f"{psutil.virtual_memory().total / (1024 ** 2):.2f} MB"
        return f"Memory used: {used} / {total}"

    def after_iteration(self, model, epoch, evals_log):
        if epoch % 100 == 0:
            logger.info(f"Iteration: {epoch}, {self.memory_log_message()}")


def fetch_yfinance(symbols, start="2020-06-01"):
    date_format = "%Y-%m-%d"
    # for MACD calculation
    start_date = datetime.strptime(start, date_format) - timedelta(days=80)

    symbols_with_usd = [symbol.upper() + "-USD" for symbol in symbols]
    print(" ".join(symbols_with_usd))
    raw_data = yf.download(" ".join(symbols_with_usd), start=start_date.strftime(date_format), threads=True)
    cols = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
    full_data = raw_data[cols].stack().reset_index()
    full_data.columns = [
        "date",
        "ticker",
        "close",
        "raw_close",
        "high",
        "low",
        "open",
        "volume",
    ]
    full_data["date"] = pd.to_datetime(full_data["date"], format="%Y-%m-%d").dt.strftime("%Y%m%d").astype(int)
    full_data["symbol"] = full_data.ticker.map(dict(zip(symbols_with_usd, symbols)))
    return full_data


# # %%
post_discord(f"Start: {NUMERAI_MODEL_ID}")
try:
    # %%
    print("Downloading data...")

    # %%
    capi = CryptoAPI(secret_key=NUMERAI_SECRET_KEY, public_id=NUMERAI_PUBLIC_ID)
    capi.download_dataset("crypto/v1.0/live_universe.parquet", "live_universe.parquet")
    capi.download_dataset("crypto/v1.0/train_targets.parquet", "train_targets.parquet")
    model_id = capi.get_models()[NUMERAI_MODEL_ID]
    train = pd.read_parquet("train_targets.parquet")
    live = pd.read_parquet("live_universe.parquet")
    symbols = train["symbol"].unique()
    data = fetch_yfinance(symbols)
    train["date"] = pd.to_datetime(train["date"], format="%Y-%m-%d").dt.strftime("%Y%m%d").astype(int)
    # %%
    full_data = pd.merge(data, train, on=["date", "symbol"], how="left")
    full_data.set_index("date", inplace=True)
    # %%
    print("Calculating indicators...")

    # %%
    ticker_groups = full_data.groupby("ticker")
    # %%
    for period in [20, 40, 60]:
        print(f"Calculating indicators for period: {period}")
        print("RSI")
        full_data[f"RSI_{period}"] = ticker_groups["close"].transform(lambda x: talib.RSI(x, period))
        print("SMA")
        full_data[f"SMA_{period}"] = ticker_groups["close"].transform(lambda x: talib.SMA(x, period))
        print("VOLATILITY")
        full_data[f"VOLATILITY_{period}"] = ticker_groups["close"].transform(lambda x: volatility(x, period))
        print("SR")
        full_data[f"SR_{period}"] = ticker_groups["close"].transform(lambda x: sr(x, period))
        print("MOM")
        full_data[f"MOM_{period}"] = ticker_groups["close"].transform(lambda x: talib.MOM(x, period))
        print("EMA")
        full_data[f"EMA_{period}"] = ticker_groups["close"].transform(lambda x: talib.EMA(x, period))
        print("DEMA")
        full_data[f"DEMA_{period}"] = ticker_groups["close"].transform(lambda x: talib.DEMA(x, period))
        print("TEMA")
        full_data[f"TEMA_{period}"] = ticker_groups["close"].transform(lambda x: talib.TEMA(x, period))
        print("TRIMA")
        full_data[f"TRIMA_{period}"] = ticker_groups["close"].transform(lambda x: talib.TRIMA(x, period))
        print("WMA")
        full_data[f"WMA_{period}"] = ticker_groups["close"].transform(lambda x: talib.WMA(x, period))

        # print("ATR")
        # full_data[f"ATR_{period}"] = ticker_groups[["low", "high", "close"]].apply(
        #     lambda x: pd.DataFrame({"ATR": talib.ATR(x["high"], x["low"], x["close"], timeperiod=period)})
        # )["ATR"]
        # print("NATR")
        # full_data[f"NATR_{period}"] = ticker_groups[["low", "high", "close"]].apply(
        #     lambda x: pd.DataFrame({"NATR": talib.NATR(x["high"], x["low"], x["close"], timeperiod=period)})
        # )["NATR"]

    full_data = full_data.drop(columns=["raw_close", "high", "low", "open", "volume"])
    gc.collect()
    # %%

    for f_period, s_period in zip([20, 40, 60], [40, 60, 80]):
        print(f"Calculating indicators for fast period: {f_period}, slow period: {s_period}")
        print("APO")
        full_data[f"APO_{f_period}_{s_period}"] = ticker_groups["close"].transform(
            lambda x: talib.APO(x, f_period, s_period)
        )
        print("MACD")
        full_data[f"MACD_{f_period}_{s_period}"] = ticker_groups["close"].transform(
            lambda x: talib.MACD(x, f_period, s_period)[0]
        )

    del ticker_groups
    gc.collect()

    indicators = list(
        set(full_data.columns.values.tolist())
        - {
            "ticker",
            "symbol",
            "close",
            "raw_close",
            "high",
            "low",
            "open",
            "volume",
            "target",
        }
    )

    full_data = full_data.dropna(subset=indicators)
    date_groups = full_data.groupby(full_data.index)

    print("Calculating quintiles...")
    for indicator in indicators:
        print(indicator)
        full_data[f"{indicator}_quintile"] = (
            date_groups[indicator]
            .transform(lambda group: pd.qcut(group, 5, labels=False, duplicates="drop"))
            .astype(np.float32)
        )

    feature_names = [f for f in full_data.columns for y in ["quintile"] if y in f]
    # %%

    features = feature_names
    # %%
    train_data, validation_data = train_test_split(full_data, test_size=0.3, shuffle=False, random_state=46)
    # %%
    gc.collect()
    train_data.dropna(subset=(features + [TARGET_NAME] + SUBMISSION_COLUMNS), inplace=True)
    # %%
    validation_data.dropna(subset=(features + [TARGET_NAME] + SUBMISSION_COLUMNS), inplace=True)

    latest_date = full_data.index.max()
    live_data = full_data.loc[full_data.index == latest_date]

    # %%
    del full_data
    gc.collect()
    # %%
    live_data[features + [TARGET_NAME] + SUBMISSION_COLUMNS].to_csv("./live_data.csv", index=True)
    # %%
    train_data[features + [TARGET_NAME] + SUBMISSION_COLUMNS].to_csv("./train_data.csv", index=True)
    validation_data[features + [TARGET_NAME] + SUBMISSION_COLUMNS].to_csv("./validation_data.csv", index=True)
    # %%
    train_data = pd.read_csv("./train_data.csv", index_col=0)
    validation_data = pd.read_csv("./validation_data.csv", index_col=0)

    feature_names = [f for f in train_data.columns for y in ["quintile"] if y in f]
    # %%
    feature_cols = [col for col in train_data.columns if col.startswith("feature_")]
    features = feature_names + feature_cols

    dtrain = xgb.DMatrix(train_data[features], label=train_data[TARGET_NAME])
    dvalid = xgb.DMatrix(validation_data[features], label=validation_data[TARGET_NAME])
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
        num_boost_round=4000,
        evals=[(dvalid, "eval")],
        verbose_eval=False,
        callbacks=[early_stop, MemoryLoggingCallback()],
    )

    # feature importance
    # raw_importances = model.feature_importance(importance_type="gain")
    # feature_name = model.boosters[0].feature_name()
    # importance_df = pd.DataFrame(data=raw_importances, columns=feature_name)
    # 平均値でソートする
    # sorted_indices = importance_df.mean(axis=0).sort_values(ascending=False).index
    # sorted_importance_df = importance_df.loc[:, sorted_indices]
    # 上位をプロットする
    # %%
    # PLOT_TOP_N = 20
    # plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
    # _, ax = plt.subplots(figsize=(8, 8))
    # ax.grid()
    # ax.set_xscale("log")
    # ax.set_ylabel("Feature")
    # ax.set_xlabel("Importance")
    # sns.boxplot(data=sorted_importance_df[plot_cols], orient="h", ax=ax)
    # plt.show()
    # %%

    live_data = pd.read_csv("./live_data.csv", index_col=0)
    dlive = xgb.DMatrix(live_data[features])

    train_data[PREDICTION_NAME] = model.predict(dtrain, iteration_range=(0, model.best_iteration + 1))
    validation_data[PREDICTION_NAME] = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))
    live_data[PREDICTION_NAME] = model.predict(dlive, iteration_range=(0, model.best_iteration + 1))

    predict_csv_file_name = f"crypto_predictions_{NUMERAI_MODEL_ID}.csv"
    live_data[["symbol", "signal"]].reset_index(drop=True).to_csv(predict_csv_file_name, index=False)
    capi.upload_predictions(predict_csv_file_name, model_id=model_id)

    # validation_predict_csv_file_name = f"crypto_validation_predictions_{NUMERAI_MODEL_ID}.csv"
    # validation_data.reset_index(drop=False)[["symbol", "date", "signal"]].to_csv(
    #     validation_predict_csv_file_name, index=False
    # )
    # napi.upload_diagnostics(validation_predict_csv_file_name, model_id=model_id)

except Exception as e:
    post_discord(f"Numerai Submit Failure. Model: {NUMERAI_MODEL_ID}, Error: {str(e)}")

# %%
