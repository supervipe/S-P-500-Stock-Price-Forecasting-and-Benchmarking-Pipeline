import time
import numpy as np

import pandas as pd
import polars as pl

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


CSV_PATH = "data/raw/all_stocks_5yr.csv"

SMA_WINDOW = 20
RSI_WINDOW = 14

RF_PARAMS = dict(n_estimators=200, random_state=42, n_jobs=-1)

FEATURES = ["open", "high", "low", "close", "volume", "sma20", "rsi14"]


def add_indicators_pandas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["name", "date"]).copy()

    df["sma20"] = df.groupby("name")["close"].transform(
        lambda s: s.rolling(SMA_WINDOW).mean()
    )

    def rsi14(close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.rolling(RSI_WINDOW).mean()
        avg_loss = loss.rolling(RSI_WINDOW).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df["rsi14"] = df.groupby("name")["close"].transform(rsi14)

    return df


def add_indicators_polars(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["name", "date"])

    delta = (pl.col("close") - pl.col("close").shift(1)).over("name")
    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)

    avg_gain = gain.rolling_mean(RSI_WINDOW).over("name")
    avg_loss = loss.rolling_mean(RSI_WINDOW).over("name")

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df = df.with_columns(
        [
            pl.col("close").rolling_mean(SMA_WINDOW).over("name").alias("sma20"),
            rsi.alias("rsi14"),
        ]
    )

    return df


def train_two_models_per_ticker(df_ticker: pd.DataFrame):
    """
    df_ticker must already contain: date, Name, open/high/low/close/volume, sma20, rsi14
    """
    df_ticker = df_ticker.sort_values("date").copy()

    df_ticker["target_next_close"] = df_ticker["close"].shift(-1)

    df_ticker = df_ticker.dropna(subset=FEATURES + ["target_next_close"])
    if len(df_ticker) < 50:
        return None

    X = df_ticker[FEATURES].to_numpy()
    y = df_ticker["target_next_close"].to_numpy()

    split_idx = int(len(df_ticker) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    def metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = root_mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return mae, rmse

    lr_mae, lr_rmse = metrics(y_test, pred_lr)
    rf_mae, rf_rmse = metrics(y_test, pred_rf)

    return {
        "rows_used": len(df_ticker),
        "lr_mae": lr_mae,
        "lr_rmse": lr_rmse,
        "rf_mae": rf_mae,
        "rf_rmse": rf_rmse,
    }


def run_part2_pandas(csv_path: str = CSV_PATH) -> pd.DataFrame:
    t0 = time.perf_counter()
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])

    t_load = time.perf_counter()

    df = add_indicators_pandas(df)
    t_ind = time.perf_counter()

    results = []
    for ticker, g in df.groupby("name"):
        out = train_two_models_per_ticker(g)
        if out is None:
            continue
        out["name"] = ticker
        results.append(out)

    t_models = time.perf_counter()

    res_df = pd.DataFrame(results)
    res_df["engine"] = "pandas"

    print("\n[PANDAS] Timing")
    print(f"Load CSV:     {t_load - t0:.3f}s")
    print(f"Indicators:   {t_ind - t_load:.3f}s")
    print(f"Modeling:     {t_models - t_ind:.3f}s")
    print(f"Total:        {t_models - t0:.3f}s")

    return res_df


def run_part2_polars(csv_path: str = CSV_PATH) -> pd.DataFrame:
    t0 = time.perf_counter()

    df = pl.read_csv(csv_path)
    df = df.with_columns(
        [pl.col("date").str.strptime(pl.Date, strict=False).alias("date")]
    )

    t_load = time.perf_counter()

    df = add_indicators_polars(df)
    t_ind = time.perf_counter()

    pdf = df.to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])

    results = []
    for ticker, g in pdf.groupby("name"):
        out = train_two_models_per_ticker(g)
        if out is None:
            continue
        out["name"] = ticker
        results.append(out)

    t_models = time.perf_counter()

    res_df = pd.DataFrame(results)
    res_df["engine"] = "polars"

    print("\n[POLARS] Timing")
    print(f"Load CSV:     {t_load - t0:.3f}s")
    print(f"Indicators:   {t_ind - t_load:.3f}s")
    print(f"Modeling:     {t_models - t_ind:.3f}s")
    print(f"Total:        {t_models - t0:.3f}s")

    return res_df


def run_compare_libraries():
    pandas_res = run_part2_pandas(CSV_PATH)
    polars_res = run_part2_polars(CSV_PATH)

    combined = pd.concat([pandas_res, polars_res], ignore_index=True)

    summary = (
        combined.groupby("engine")[["lr_mae", "lr_rmse", "rf_mae", "rf_rmse"]]
        .mean()
        .reset_index()
    )

    print("\n===== Overall Average Metrics (lower is better) =====")
    print(summary)

    combined.to_csv("data/processed/part2_per_ticker_results.csv", index=False)
    summary.to_csv("data/processed/part2_summary_results.csv", index=False)

    print("\nSaved:")
    print(" - part2_per_ticker_results.csv")
    print(" - part2_summary_results.csv")


if __name__ == "__main__":
    run_compare_libraries()
