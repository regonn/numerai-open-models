# %%
import gc
import os
from datetime import datetime, timedelta
from logging import getLogger

import catboost as cb
import kand
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import requests
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
    cols = ["Close", "High", "Low", "Open", "Volume"]
    full_data = raw_data[cols].stack().reset_index()
    full_data.columns = [
        "date",
        "ticker",
        "close",
        "high",
        "low",
        "open",
        "volume",
    ]
    full_data["date"] = pd.to_datetime(full_data["date"], format="%Y-%m-%d").dt.strftime("%Y%m%d").astype(int)
    full_data["symbol"] = full_data.ticker.map(dict(zip(symbols_with_usd, symbols)))
    return full_data


# # %%
# post_discord(f"Start: {NUMERAI_MODEL_ID}")
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

    # 160行未満のグループを除外
    group_sizes = ticker_groups.size()
    valid_tickers = group_sizes[group_sizes >= 160].index
    full_data = full_data[full_data["ticker"].isin(valid_tickers)]

    # グループを再作成
    ticker_groups = full_data.groupby("ticker")

    for period in [20, 40, 60]:
        print(f"Calculating indicators for period: {period}")
        print("RSI")
        full_data[f"RSI_{period}"] = ticker_groups["close"].transform(
            lambda x: pd.Series(kand.rsi(np.array(x, dtype=np.float64), period)[0], index=x.index)
        )
        print("SMA")
        full_data[f"SMA_{period}"] = ticker_groups["close"].transform(
            lambda x: pd.Series(kand.sma(np.array(x, dtype=np.float64), period), index=x.index)
        )
        print("VOLATILITY")
        full_data[f"VOLATILITY_{period}"] = ticker_groups["close"].transform(lambda x: volatility(x, period))
        print("SR")
        full_data[f"SR_{period}"] = ticker_groups["close"].transform(lambda x: sr(x, period))
        print("MOM")
        full_data[f"MOM_{period}"] = ticker_groups["close"].transform(
            lambda x: pd.Series(kand.mom(np.array(x, dtype=np.float64), period), index=x.index)
        )
        print("EMA")
        full_data[f"EMA_{period}"] = ticker_groups["close"].transform(
            lambda x: pd.Series(kand.ema(np.array(x, dtype=np.float64), period), index=x.index)
        )
        print("DEMA")
        full_data[f"DEMA_{period}"] = ticker_groups["close"].transform(
            lambda x: pd.Series(kand.dema(np.array(x, dtype=np.float64), period)[0], index=x.index)
        )
        # print("TEMA")
        # full_data[f"TEMA_{period}"] = ticker_groups["close"].transform(
        #     lambda x: pd.Series(kand.tema(np.array(x, dtype=np.float64), period)[0], index=x.index)
        # )
        print("TRIMA")
        full_data[f"TRIMA_{period}"] = ticker_groups["close"].transform(
            lambda x: pd.Series(kand.trima(np.array(x, dtype=np.float64), period)[0], index=x.index)
        )
        print("WMA")
        full_data[f"WMA_{period}"] = ticker_groups["close"].transform(
            lambda x: pd.Series(kand.wma(np.array(x, dtype=np.float64), period), index=x.index)
        )

        # print("ATR")
        # full_data[f"ATR_{period}"] = ticker_groups[["low", "high", "close"]].apply(
        #     lambda x: pd.DataFrame({"ATR": talib.ATR(x["high"], x["low"], x["close"], timeperiod=period)})
        # )["ATR"]
        # print("NATR")
        # full_data[f"NATR_{period}"] = ticker_groups[["low", "high", "close"]].apply(
        #     lambda x: pd.DataFrame({"NATR": talib.NATR(x["high"], x["low"], x["close"], timeperiod=period)})
        # )["NATR"]

    full_data = full_data.drop(columns=["close", "high", "low", "open", "volume"])
    gc.collect()
    # %%

    for f_period, s_period in zip([20, 40, 60], [40, 60, 80]):
        print(f"Calculating indicators for fast period: {f_period}, slow period: {s_period}")
        print("APO")
        # full_data[f"APO_{f_period}_{s_period}"] = ticker_groups["close"].transform(
        #     lambda x: kand.apo(np.array(x, dtype=np.float64), f_period, s_period)
        # )
        print("MACD")
        full_data[f"MACD_{f_period}_{s_period}"] = ticker_groups["close"].transform(
            lambda x: pd.Series(
                kand.macd(np.array(x, dtype=np.float64), f_period, s_period, signal_period=9)[0], index=x.index
            )
        )

    del ticker_groups
    gc.collect()

    indicators = list(
        set(full_data.columns.values.tolist())
        - {
            "ticker",
            "symbol",
            "close",
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

    dtrain_lgb = lgb.Dataset(train_data[features], label=train_data[TARGET_NAME])
    dtrain_xgb = xgb.DMatrix(train_data[features], label=train_data[TARGET_NAME])
    dtrain_cb = cb.Pool(train_data[features], label=train_data[TARGET_NAME])

    gc.collect()

    dvalid_lgb = lgb.Dataset(validation_data[features], label=validation_data[TARGET_NAME])
    dvalid_xgb = xgb.DMatrix(validation_data[features], label=validation_data[TARGET_NAME])
    dvalid_cb = cb.Pool(validation_data[features], label=validation_data[TARGET_NAME])

    gc.collect()

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

    dlive_xgb = xgb.DMatrix(live_data[features])
    dlive_cb = cb.Pool(live_data[features])

    preds_lgb = model_lgb.predict(live_data[features], num_iteration=model_lgb.best_iteration)
    preds_xgb = model_xgb.predict(dlive_xgb, iteration_range=(0, model_xgb.best_iteration + 1))
    preds_cb = model_cb.predict(dlive_cb)

    live_data[PREDICTION_NAME] = (preds_lgb + preds_xgb + preds_cb) / 3

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
