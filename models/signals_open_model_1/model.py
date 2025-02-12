# %%
import gc
import os
from datetime import timedelta
from logging import getLogger

import kaggle
import numpy as np
import pandas as pd
import psutil
import requests
import talib
import xgboost as xgb
from dotenv import load_dotenv
from numerapi import NumerAPI, SignalsAPI
from sklearn import preprocessing

# import matplotlib.pyplot as plt
# import seaborn as sns

logger = getLogger(__name__)

load_dotenv()

DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
NUMERAI_PUBLIC_ID = os.environ["NUMERAI_PUBLIC_ID"]
NUMERAI_SECRET_KEY = os.environ["NUMERAI_SECRET_KEY"]
NUMERAI_MODEL_ID = "signals_open_model_1"
CATEGORY_COLUMNS = ["country", "sector", "industry"]
SUBMISSION_COLUMNS = ["numerai_ticker"]
TARGET_NAME = "target"
PREDICTION_NAME = "signal"
TICKER_CTRY_MAP = {
    "AU": "AU",
    "AV": "AT",
    "BB": "BE",
    "BZ": "BR",
    "CA": "CA",
    "CB": "CO",
    "CH": "CN",
    "CI": "CL",
    "CN": "CA",
    "CP": "CZ",
    "DC": "DK",
    "EY": "EG",
    "FH": "FI",
    "FP": "FR",
    "GA": "GR",
    "GR": "DE",
    "GY": "DE",
    "HB": "HU",
    "HK": "HK",
    "ID": "IE",
    "IJ": "ID",
    "IM": "IT",
    "IN": "IN",
    "IT": "IL",
    "JP": "JP",
    "KS": "KR",
    "LN": "GB",
    "MF": "MX",
    "MK": "MY",
    "NA": "NL",
    "NO": "NO",
    "NZ": "NZ",
    "PE": "PE",
    "PL": "PT",
    "PM": "PH",
    "PW": "PL",
    "QD": "QA",
    "RM": "RU",
    "SJ": "ZA",
    "SM": "ES",
    "SP": "SG",
    "SS": "SE",
    "SW": "CH",
    "TB": "TH",
    "TI": "TR",
    "TT": "TW",
    "UH": "AE",
    "US": "US",
    "UQ": "US",
}


def map_country_code(row):
    if pd.isna(row["bloomberg_ticker"]):
        return None

    split_ticker = row["bloomberg_ticker"].split()
    if len(split_ticker) < 2:
        return None

    ticker = split_ticker[0]
    country_code = split_ticker[-1]
    iso_country_code = TICKER_CTRY_MAP.get(country_code)

    if iso_country_code is None:
        return None
    else:
        return f"{ticker} {iso_country_code}"


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


# %%
post_discord(f"Start: {NUMERAI_MODEL_ID}")
try:
    # %%
    print("Downloading data...")
    kaggle.api.dataset_download_files("code1110/yfinance-stock-price-data-for-numerai-signals", path="./", unzip=True)
    napi = NumerAPI()
    napi.download_dataset("signals/v2.0/train.parquet", "train.parquet")
    napi.download_dataset("signals/v2.0/validation.parquet", "validation.parquet")
    napi.download_dataset("signals/v2.0/live.parquet", "live.parquet")
    napi.download_dataset("signals/v2.0/live_example_preds.parquet", "live_example_preds.parquet")
    napi.download_dataset(
        "signals/v2.0/validation_example_preds.parquet",
        "validation_example_preds.parquet",
    )
    # %%
    api = SignalsAPI(secret_key=NUMERAI_SECRET_KEY, public_id=NUMERAI_PUBLIC_ID)
    model_id = api.get_models()[NUMERAI_MODEL_ID]

    ticker_map = pd.read_csv("signals_ticker_map_w_bbg.csv")
    ticker_map["numerai_ticker"] = ticker_map.apply(map_country_code, axis=1)

    # %%
    full_data = pd.read_parquet("full_data.parquet")

    full_data["date"] = pd.to_datetime(full_data["date"], format="%Y%m%d")

    full_data.set_index("date", inplace=True)

    full_data = full_data[full_data.close > 0]
    gc.collect()
    # %%
    print("Calculating indicators...")
    full_data["bloomberg_ticker"] = full_data.ticker.map(
        dict(zip(ticker_map["ticker"], ticker_map["bloomberg_ticker"]))
    )
    full_data["numerai_ticker"] = full_data.ticker.map(dict(zip(ticker_map["ticker"], ticker_map["numerai_ticker"])))
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

    full_data.dropna(inplace=True)
    gc.collect()

    full_data["country"] = full_data["bloomberg_ticker"].str.split(" ", expand=True)[1]
    full_data["country"] = full_data["country"]

    for column in ["country"]:
        print(column)
        le = preprocessing.LabelEncoder()
        le.fit(full_data[column])
        full_data[column] = le.transform(full_data[column])
    # %%
    yfinance_sector_industry = pd.read_csv("./yfinance_sector_industry.csv")
    yfinance_sector_industry.dropna(inplace=True)

    yfinance_sector_industry["bloomberg_ticker"] = yfinance_sector_industry["ticker"].map(
        dict(zip(ticker_map["yahoo"], ticker_map["bloomberg_ticker"]))
    )
    yfinance_sector_industry.dropna(inplace=True)

    for column in ["sector", "industry"]:
        print(column)
        le = preprocessing.LabelEncoder()
        le.fit(yfinance_sector_industry[column])
        yfinance_sector_industry[column] = le.transform(yfinance_sector_industry[column])

    gc.collect()

    full_data = pd.merge(
        full_data.reset_index(),
        yfinance_sector_industry[["sector", "industry", "bloomberg_ticker"]],
        how="left",
        on="bloomberg_ticker",
        copy=False,
    ).set_index("date")

    del yfinance_sector_industry
    gc.collect()

    full_data["sector"] = full_data["sector"].fillna(full_data["sector"].max() + 1)
    full_data["industry"] = full_data["industry"].fillna(full_data["industry"].max() + 1)

    full_data["sector"] = full_data["sector"].astype(np.uint64)
    full_data["industry"] = full_data["industry"].astype(np.uint64)
    gc.collect()

    full_data["ret"] = full_data.groupby("ticker")["close"].pct_change()
    full_data["industry_ret"] = full_data.groupby(["date", "industry"])["ret"].transform(lambda x: x.mean())
    full_data["country_ret"] = full_data.groupby(["date", "country"])["ret"].transform(lambda x: x.mean())
    full_data["sector_ret"] = full_data.groupby(["date", "sector"])["ret"].transform(lambda x: x.mean())

    indicators = list(
        set(full_data.columns.values.tolist())
        - {
            "ticker",
            "bloomberg_ticker",
            "numerai_ticker",
            "country",
            "sector",
            "close",
            "industry",
            "ret",
            "raw_close",
            "high",
            "low",
            "open",
            "volume",
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

    full_data.dropna(inplace=True)
    gc.collect()

    feature_names = [f for f in full_data.columns for y in ["quintile"] if y in f] + CATEGORY_COLUMNS
    # %%
    train_df = pd.read_parquet("train.parquet")
    train_df["date"] = pd.to_datetime(train_df["date"], format="%Y-%m-%d")
    feature_cols = [col for col in train_df.columns if col.startswith("feature_")]
    feature_cols = [s for s in feature_cols if s not in ("feature_country", "feature_exchange_code")]
    features = feature_names + feature_cols
    # %%

    train_data = pd.merge(full_data.reset_index(), train_df, on=["date", "numerai_ticker"], copy=False).set_index(
        "date"
    )
    # %%
    gc.collect()
    train_data.dropna(subset=(features + [TARGET_NAME] + SUBMISSION_COLUMNS), inplace=True)
    # %%
    validation_df = pd.read_parquet("validation.parquet")
    validation_df["date"] = pd.to_datetime(validation_df["date"], format="%Y-%m-%d")
    validation_data = pd.merge(
        full_data.reset_index(),
        validation_df,
        on=["date", "numerai_ticker"],
        copy=False,
    ).set_index("date")
    validation_data.dropna(subset=(features + [TARGET_NAME] + SUBMISSION_COLUMNS), inplace=True)
    live_df = pd.read_parquet("live.parquet")
    live_df["date"] = pd.to_datetime(live_df["date"], format="%Y-%m-%d")
    live_date = live_df["date"].iloc[0]

    # %%
    live_data = pd.merge(full_data.reset_index(), live_df, on=["date", "numerai_ticker"], copy=False).set_index("date")
    # %%
    for day_before in range(1, 10):
        missing_tickers = set(live_df["numerai_ticker"]) - set(live_data["numerai_ticker"])
        missing_data = live_df[live_df["numerai_ticker"].isin(missing_tickers)]
        the_day = live_date - timedelta(days=day_before)
        missing_data.loc[:, "date"] = missing_data["date"] - timedelta(days=day_before)
        the_day_data = pd.merge(
            full_data.reset_index(),
            missing_data,
            on=["date", "numerai_ticker"],
            copy=False,
        ).set_index("date")
        live_data = pd.concat([live_data, the_day_data])
        print(f"Number of live tickers to submit: {len(the_day_data)}(@{day_before} day before)")
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
    gc.collect()

    feature_names = [f for f in train_data.columns for y in ["quintile"] if y in f] + CATEGORY_COLUMNS
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

    predict_csv_file_name = f"signals_predictions_{NUMERAI_MODEL_ID}.csv"
    live_data[["numerai_ticker", "signal"]].reset_index(drop=True).to_csv(predict_csv_file_name, index=False)
    api.upload_predictions(predict_csv_file_name, model_id=model_id)

    validation_predict_csv_file_name = f"signals_validation_predictions_{NUMERAI_MODEL_ID}.csv"
    validation_data.reset_index(drop=False)[["numerai_ticker", "date", "signal"]].to_csv(
        validation_predict_csv_file_name, index=False
    )
    api.upload_diagnostics(validation_predict_csv_file_name, model_id=model_id)

except Exception as e:
    post_discord(f"Numerai Submit Failure. Model: {NUMERAI_MODEL_ID}, Error: {str(e)}")
