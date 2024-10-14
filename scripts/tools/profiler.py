import json
import matplotlib.pyplot as plt
from glob import glob
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()

    files = glob(os.path.join(args.data_dir, "tuned_"+"*", "stats", "*.json"))

    num_gs_values = []
    elapsed_time_values = []

    # 各ファイルをループしてデータを抽出
    for filename in files:
        with open(filename, 'r') as f:
            data = json.load(f)
            if data["elapsed_time"] < 10:
                num_gs_values.append(data['num_GS'])
                elapsed_time_values.append(data['elapsed_time'])

    # データのプロット
    plt.scatter(num_gs_values, elapsed_time_values)
    plt.xlabel('num_GS')
    plt.ylabel('elapsed_time')
    plt.savefig(os.path.join("figures", "plot.png"))
    plt.show()