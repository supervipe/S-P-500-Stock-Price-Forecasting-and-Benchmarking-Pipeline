import pandas as pd
import time
import os

CSV_PATH = "data/raw/all_stocks_5yr.csv"
PARQUET_PATH = "data/processed/all_stocks_5yr.parquet"

SCALES = {"1x": 1, "10x": 10, "100x": 100}


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def benchmark_read_csv(path):
    start = time.perf_counter()
    df = pd.read_csv(path)
    return df, time.perf_counter() - start


def benchmark_write_parquet(df, path):
    start = time.perf_counter()
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    return time.perf_counter() - start


def benchmark_read_parquet(path):
    start = time.perf_counter()
    df = pd.read_parquet(path, engine="pyarrow")
    return df, time.perf_counter() - start


def run_storage_benchmark(
    csv_path: str = CSV_PATH,
    parquet_path: str = PARQUET_PATH,
    out_csv: str = "data/processed/part1_benchmark_results.csv",
):
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    processed_dir = os.path.dirname(parquet_path)

    df_1x, csv_read_1x = benchmark_read_csv(csv_path)
    parquet_write_1x = benchmark_write_parquet(df_1x, parquet_path)
    _, parquet_read_1x = benchmark_read_parquet(parquet_path)

    results = [
        {
            "Scale": "1x",
            "Rows": len(df_1x),
            "CSV Size (MB)": file_size_mb(csv_path),
            "Parquet Size (MB)": file_size_mb(parquet_path),
            "CSV Read (s)": csv_read_1x,
            "Parquet Read (s)": parquet_read_1x,
            "CSV Write (s)": 0.0,
            "Parquet Write (s)": parquet_write_1x,
        }
    ]

    for label, mult in SCALES.items():
        if label == "1x":
            continue

        df_scaled = pd.concat([df_1x] * mult, ignore_index=True)

        scaled_csv = os.path.join(processed_dir, f"scaled_{label}.csv")
        scaled_parquet = os.path.join(processed_dir, f"scaled_{label}.parquet")

        start = time.perf_counter()
        df_scaled.to_csv(scaled_csv, index=False)
        csv_write = time.perf_counter() - start

        _, csv_read = benchmark_read_csv(scaled_csv)
        parquet_write = benchmark_write_parquet(df_scaled, scaled_parquet)
        _, parquet_read = benchmark_read_parquet(scaled_parquet)

        results.append(
            {
                "Scale": label,
                "Rows": len(df_scaled),
                "CSV Size (MB)": file_size_mb(scaled_csv),
                "Parquet Size (MB)": file_size_mb(scaled_parquet),
                "CSV Read (s)": csv_read,
                "Parquet Read (s)": parquet_read,
                "CSV Write (s)": csv_write,
                "Parquet Write (s)": parquet_write,
            }
        )

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)

    print("Saved Part 1 results to:", os.path.abspath(out_csv))
    return out_df


if __name__ == "__main__":
    run_storage_benchmark()
