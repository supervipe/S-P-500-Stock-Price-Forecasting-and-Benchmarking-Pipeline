import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATASET_PATH = "data/raw/all_stocks_5yr.csv"

PART1_TABLE_PATH = "data/processed/part1_benchmark_results.csv"
PART2_SUMMARY_PATH = "data/processed/part2_summary_results.csv"
PART2_TICKER_METRICS_PATH = "data/processed/part2_per_ticker_results.csv"

SMA_WINDOW = 20
RSI_WINDOW = 14

FEATURES = ["open", "high", "low", "close", "volume", "sma20", "rsi14"]

RF_PARAMS = dict(
    n_estimators=50,
    random_state=42,
    n_jobs=-1,
)


@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "Name" in df.columns and "name" not in df.columns:
        df = df.rename(columns={"Name": "name"})

    df["date"] = pd.to_datetime(df["date"])
    return df


def add_indicators_manual(df_one: pd.DataFrame) -> pd.DataFrame:
    """
    Adds SMA(20) and RSI(14) to a single-ticker dataframe.
    Expects columns: date, close (and other OHLCV)
    """
    df_one = df_one.sort_values("date").copy()

    df_one["sma20"] = df_one["close"].rolling(SMA_WINDOW).mean()

    delta = df_one["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(RSI_WINDOW).mean()
    avg_loss = loss.rolling(RSI_WINDOW).mean()

    rs = avg_gain / avg_loss
    df_one["rsi14"] = 100 - (100 / (1 + rs))

    return df_one


def train_models_for_one_ticker(df_one: pd.DataFrame):
    df_one = df_one.sort_values("date").copy()
    df_one["target_next_close"] = df_one["close"].shift(-1)

    last_row = df_one.dropna(subset=FEATURES).tail(1)
    if last_row.empty:
        return None

    X_last = last_row[FEATURES].to_numpy()
    last_date = last_row["date"].iloc[0]

    df_train = df_one.dropna(subset=FEATURES + ["target_next_close"])
    if len(df_train) < 60:
        return None

    X = df_train[FEATURES].to_numpy()
    y = df_train["target_next_close"].to_numpy()
    dates = df_train["date"].to_numpy()

    split = int(len(df_train) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    d_test = dates[split:]

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    next_lr = float(lr.predict(X_last)[0])
    next_rf = float(rf.predict(X_last)[0])

    def metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return mae, rmse

    lr_mae, lr_rmse = metrics(y_test, pred_lr)
    rf_mae, rf_rmse = metrics(y_test, pred_rf)

    preds_df = pd.DataFrame(
        {
            "date": pd.to_datetime(d_test),
            "actual_next_close": y_test,
            "pred_lr": pred_lr,
            "pred_rf": pred_rf,
        }
    )

    metrics_df = pd.DataFrame(
        {
            "model": ["LinearRegression", "RandomForest"],
            "MAE": [lr_mae, rf_mae],
            "RMSE": [lr_rmse, rf_rmse],
            "Next-day prediction": [next_lr, next_rf],
        }
    )

    return preds_df, metrics_df, last_date


def plot_predictions(preds: pd.DataFrame, model_col: str, title: str):
    fig = plt.figure()
    plt.plot(preds["date"], preds["actual_next_close"], label="Actual (next-day close)")
    plt.plot(preds["date"], preds[model_col], label=f"Predicted ({model_col})")
    plt.xlabel("Date (test period)")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    st.pyplot(fig)


def run_dashboard():
    st.set_page_config(page_title="CSIS 4260 - Assignment 1 Dashboard", layout="wide")
    st.title("CSIS 4260 – Assignment 1 (Part 3 Dashboard)")

    st.sidebar.header("Controls")

    if not os.path.exists(DATASET_PATH):
        st.error(
            f"Dataset not found: {DATASET_PATH}. Put it in the same folder as this app."
        )
        st.stop()

    df = load_dataset(DATASET_PATH)

    tickers = sorted(df["name"].unique().tolist())
    ticker = st.sidebar.selectbox("Select a company ticker", tickers)

    model_choice = st.sidebar.radio(
        "Choose model to plot", ["LinearRegression", "RandomForest"]
    )
    model_col = "pred_lr" if model_choice == "LinearRegression" else "pred_rf"

    df_one = df[df["name"] == ticker].copy()
    df_one = add_indicators_manual(df_one)

    trained = train_models_for_one_ticker(df_one)

    if trained is None:
        st.warning("Not enough data after indicators to train/test for this ticker.")
    else:
        preds_df, metrics_df, last_date = trained

        st.subheader("Next Trading Day Forecast (based on latest available row)")
        st.write(f"Latest date in dataset: **{pd.to_datetime(last_date).date()}**")

        lr_next = metrics_df.loc[
            metrics_df["model"] == "LinearRegression", "Next-day prediction"
        ].iloc[0]
        rf_next = metrics_df.loc[
            metrics_df["model"] == "RandomForest", "Next-day prediction"
        ].iloc[0]

        c1, c2 = st.columns(2)
        c1.metric("LinearRegression — Predicted next close", f"{lr_next:.2f}")
        c2.metric("RandomForest — Predicted next close", f"{rf_next:.2f}")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Predictions for {ticker}")
        if trained is None:
            st.warning(
                "Not enough data after indicators to train/test for this ticker."
            )
        else:
            preds_df, metrics_df, last_date = trained
            plot_predictions(
                preds_df, model_col, f"{ticker} — Actual vs Predicted (Test Split)"
            )
            st.caption(
                "Target = next-day close. 80/20 split is time-based (backtesting style)."
            )

    with col2:
        st.subheader("Model Metrics (Selected Ticker)")
        if trained is None:
            st.write("—")
        else:
            preds_df, metrics_df, last_date = trained
            st.dataframe(metrics_df, use_container_width=True)

    st.divider()

    st.header("Benchmark & Results Tables")

    t1, t2 = st.columns(2)

    with t1:
        st.subheader("Part 1 — CSV vs Parquet Benchmark")
        if os.path.exists(PART1_TABLE_PATH):
            part1_df = pd.read_csv(PART1_TABLE_PATH)
            st.dataframe(part1_df, use_container_width=True)
        else:
            st.info(
                f"Missing {PART1_TABLE_PATH}. If you used my Part 1 exporter, run:\n"
                f"`python part1_benchmark.py`\n"
                f"to generate it."
            )

    with t2:
        st.subheader("Part 2 — Overall Summary (Pandas vs Polars)")
        if os.path.exists(PART2_SUMMARY_PATH):
            part2_sum = pd.read_csv(PART2_SUMMARY_PATH)
            st.dataframe(part2_sum, use_container_width=True)
        else:
            st.info(
                f"Missing {PART2_SUMMARY_PATH}. Run your Part 2 script to generate:\n"
                f"`part2_summary_results.csv` and `part2_per_ticker_results.csv`."
            )


if __name__ == "__main__":
    run_dashboard()
