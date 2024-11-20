# 必要なモジュールのインポート
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy.spatial.distance import euclidean
from math import radians

# CSVファイル選択ダイアログ
def load_csv():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    return file_path

# hhmmss形式のタイムスタンプを秒に変換
def convert_to_seconds(timestamp):
    # タイムスタンプが整数型の場合は文字列に変換
    timestamp = str(timestamp)
    h = int(timestamp[:2])
    m = int(timestamp[2:4])
    s = int(timestamp[4:])
    return h * 3600 + m * 60 + s

# 時間範囲をhhmmss形式で指定し、データをフィルタリング
def filter_data(df, start_time_str, end_time_str):
    # 時間範囲を秒に変換
    start_time = convert_to_seconds(start_time_str)
    end_time = convert_to_seconds(end_time_str)
    
    # timestampを秒に変換
    df['time_seconds'] = df['timestamp'].apply(convert_to_seconds)
    
    return df[(df['time_seconds'] >= start_time) & (df['time_seconds'] <= end_time)]

# 注視時間（Fixation Duration）の計算
def calculate_fixation_duration(df):
    fixation_df = df[df['classification'] == 'fixation']
    fixation_duration = fixation_df.groupby('classification').size()  # 注視時間（ms）
    return fixation_duration.sum()

# 平均サッケード振幅（Average Saccade Amplitude）の計算
def calculate_avg_saccade_amplitude(df):
    saccade_df = df[df['classification'] == 'saccade']
    amplitudes = [euclidean((x1, y1), (x2, y2)) for x1, y1, x2, y2 in zip(saccade_df['x'][:-1], saccade_df['y'][:-1], saccade_df['x'][1:], saccade_df['y'][1:])]
    return np.mean(amplitudes)

# 最大サッケード振幅（Max Saccade Amplitude）の計算
def calculate_max_saccade_amplitude(df):
    saccade_df = df[df['classification'] == 'saccade']
    amplitudes = [euclidean((x1, y1), (x2, y2)) for x1, y1, x2, y2 in zip(saccade_df['x'][:-1], saccade_df['y'][:-1], saccade_df['x'][1:], saccade_df['y'][1:])]
    return np.max(amplitudes)

# サッケードのカウント（Saccade Count）の計算
def calculate_saccade_count(df):
    saccade_df = df[df['classification'] == 'saccade']
    return saccade_df.shape[0]

# 平均速度（Average Velocity）の計算
def calculate_avg_velocity(df):
    velocities = df['velocity'].values
    return np.mean(velocities)

# 速度の標準偏差（Velocity Std Dev）の計算
def calculate_velocity_std_dev(df):
    velocities = df['velocity'].values
    return np.std(velocities)

# 視覚的探索距離（Search Distance）の計算
def calculate_search_distance(df):
    fixation_df = df[df['classification'] == 'fixation']
    distance = 0
    for i in range(1, len(fixation_df)):
        distance += euclidean((fixation_df['x'].iloc[i], fixation_df['y'].iloc[i]), (fixation_df['x'].iloc[i-1], fixation_df['y'].iloc[i-1]))
    return distance

# 視覚的探索の均一性（Search Uniformity）の計算
def calculate_search_uniformity(df):
    fixation_df = df[df['classification'] == 'fixation']
    x_values = fixation_df['x']
    y_values = fixation_df['y']
    area = np.ptp(x_values) * np.ptp(y_values)  # 視線が移動した範囲
    return area / len(fixation_df)

# 平均視線回転（Average Gaze Rotation）の計算
def calculate_avg_gaze_rotation(df):
    fixation_df = df[df['classification'] == 'fixation']
    angles = []
    for i in range(1, len(fixation_df)):
        dx = fixation_df['x'].iloc[i] - fixation_df['x'].iloc[i-1]
        dy = fixation_df['y'].iloc[i] - fixation_df['y'].iloc[i-1]
        angle = np.arctan2(dy, dx)
        angles.append(angle)
    return np.mean(angles)

# メイン関数
def main():
    # CSVファイルの選択
    file_path = load_csv()
    df = pd.read_csv(file_path)

    # timestamp列を文字列に変換
    df['timestamp'] = df['timestamp'].astype(str)

    # 座標(0,0)は外れ値として扱う
    df = df[(df['x'] != 0) & (df['y'] != 0)]

    # ユーザーにhhmmss形式で時間範囲を指定
    start_time_str = input("開始時間を入力してください (hhmmss形式): ")
    end_time_str = input("終了時間を入力してください (hhmmss形式): ")

    # 時間範囲でフィルタリング
    df_filtered = filter_data(df, start_time_str, end_time_str)

    # 特徴量の計算
    fixation_duration = calculate_fixation_duration(df_filtered)
    avg_saccade_amplitude = calculate_avg_saccade_amplitude(df_filtered)
    max_saccade_amplitude = calculate_max_saccade_amplitude(df_filtered)
    saccade_count = calculate_saccade_count(df_filtered)
    avg_velocity = calculate_avg_velocity(df_filtered)
    velocity_std_dev = calculate_velocity_std_dev(df_filtered)
    search_distance = calculate_search_distance(df_filtered)
    search_uniformity = calculate_search_uniformity(df_filtered)
    avg_gaze_rotation = calculate_avg_gaze_rotation(df_filtered)

    # 結果を表示
    print("Fixation Duration (ms):", fixation_duration)
    print("Average Saccade Amplitude (px):", avg_saccade_amplitude)
    print("Max Saccade Amplitude (px):", max_saccade_amplitude)
    print("Saccade Count:", saccade_count)
    print("Average Velocity:", avg_velocity)
    print("Velocity Std Dev:", velocity_std_dev)
    print("Search Distance (px):", search_distance)
    print("Search Uniformity:", search_uniformity)
    print("Average Gaze Rotation (rad):", avg_gaze_rotation)

# 実行
if __name__ == "__main__":
    main()
