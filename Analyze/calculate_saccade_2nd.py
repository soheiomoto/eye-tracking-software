'''
This module is for experiment data after 2nd term

  Dependencies:
  - numpy                     2.1.2
  - pandas                    2.2.3
  - matplotlib                3.9.2
  - scipy                     1.14.1
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

# タイムスタンプを秒単位に変換する関数
def convert_timestamp_to_seconds(timestamp):
    try:
        # タイムスタンプを文字列として扱う
        timestamp = str(timestamp).zfill(12)  # 桁数不足をゼロで埋める
        hours = int(timestamp[:2])
        minutes = int(timestamp[2:4])
        seconds = int(timestamp[4:6])
        microseconds = int(timestamp[6:])
        total_seconds = hours * 3600 + minutes * 60 + seconds + microseconds / 1_000_000
        return total_seconds
    except (ValueError, TypeError):
        return np.nan  # 変換に失敗した場合は NaN を返す

# 視線速度計算関数
def calculate_velocity(data, sampling_rate=20):
    delta_t = 1 / sampling_rate  # サンプリング間隔（秒）
    dx = np.diff(data['X Coordinate'])
    dy = np.diff(data['Y Coordinate'])
    velocity = np.sqrt(dx**2 + dy**2) / delta_t
    return np.concatenate(([0], velocity))  # 最初の速度は0で埋める

# 動的閾値法による注視/サッケード分類
def dynamic_threshold(velocity, window_size=50, k=1.5):
    # velocity を NumPy 配列に変換
    velocity = velocity.to_numpy() if isinstance(velocity, pd.Series) else velocity

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
    data.columns = ['Record Time', 'X Coordinate', 'Y Coordinate']

    # タイムスタンプを秒に変換
    data['timestamp'] = data['Record Time'].apply(convert_timestamp_to_seconds)

    # 欠損値を除外
    data = data.dropna(subset=['timestamp', 'X Coordinate', 'Y Coordinate'])

    # 視線速度の計算
    data['velocity'] = calculate_velocity(data)

    # 動的閾値を計算
    thresholds, classifications = dynamic_threshold(data['velocity'])
    data['threshold'] = thresholds
    data['classification'] = classifications

    # 結果を保存
    output_file = file_path.replace("-Reduced.csv", "-Classified.csv")
    data.to_csv(output_file, index=False)
    print(f"処理結果を保存しました: {output_file}")

if __name__ == "__main__":
    main()