'''
This module is for 1st term experiment data
'''

# 必要なモジュールのインポート
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ファイル選択のための関数
def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    return file_path

# 視線速度計算関数
def calculate_velocity(data, sampling_rate=20):
    delta_t = 1 / sampling_rate  # サンプリング間隔（秒）
    dx = np.diff(data['x'])
    dy = np.diff(data['y'])
    velocity = np.sqrt(dx**2 + dy**2) / delta_t
    return np.concatenate(([0], velocity))  # 最初の速度は0で埋める

# 動的閾値法による注視/サッケード分類
def dynamic_threshold(velocity, window_size=50, k=1.5):
    # NumPy配列に変換
    velocity = velocity.values  # これでインデックスの問題を回避
    smoothed_velocity = gaussian_filter1d(velocity, sigma=5)
    thresholds = []
    classifications = []

    for i in range(len(velocity)):
        start_idx = max(0, i - window_size)
        end_idx = min(len(velocity), i + window_size)
        local_mean = np.mean(smoothed_velocity[start_idx:end_idx])
        local_std = np.std(smoothed_velocity[start_idx:end_idx])
        threshold = local_mean + k * local_std
        thresholds.append(threshold)
        classifications.append("saccade" if velocity[i] > threshold else "fixation")

    return thresholds, classifications

# メイン処理
def main():
    file_path = select_file()
    if not file_path:
        print("ファイルが選択されませんでした。")
        return

    # CSV読み込み
    data = pd.read_csv(file_path)
    data.columns = ['timestamp', 'x', 'y']

    # 外れ値 (0, 0) を除外
    data = data[(data['x'] != 0) & (data['y'] != 0)]

    # 視線速度の計算
    data['velocity'] = calculate_velocity(data)

    # 動的閾値を計算
    thresholds, classifications = dynamic_threshold(data['velocity'])
    data['threshold'] = thresholds
    data['classification'] = classifications

    # 結果を保存
    output_file = file_path.replace(".csv", "-Classified.csv")
    data.to_csv(output_file, index=False)
    print(f"処理結果を保存しました: {output_file}")

    # 可視化
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(data['velocity'], label="Velocity")
    plt.plot(data['threshold'], label="Dynamic Threshold", linestyle="--")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.show()
    '''

if __name__ == "__main__":
    main()
