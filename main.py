from part1 import run_storage_benchmark
from part2 import run_compare_libraries
import subprocess, sys


def run_part3():
    subprocess.run([sys.executable, "-m", "streamlit", "run", "part3.py"], check=True)


if __name__ == "__main__":
    run_storage_benchmark()
    run_compare_libraries()
    run_part3()
