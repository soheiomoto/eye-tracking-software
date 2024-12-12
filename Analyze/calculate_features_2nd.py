import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy.spatial.distance import euclidean

# CSVファイル選択ダイアログ
def load_csv():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    return file_path

# データの時間範囲（Time Range）の計算
def calculate_time_range(df):
    timestamps = pd.to_datetime(df['Record Time'].str[:6], format='%H%M%S', errors='coerce')
    return (timestamps.max() - timestamps.min()).total_seconds()

# 平均速度（Average Velocity）の計算
def calculate_avg_velocity(df):
    velocities = df['velocity'].values
    return np.mean(velocities) if len(velocities) > 0 else 0

# 切り替え回数（Transition Count）の計算
def calculate_transition_count(df):
    transitions = (df['classification'] != df['classification'].shift()).sum() - 1
    return transitions if transitions > 0 else 0

# サッケードカウントの割合（Saccade Count Ratio）の計算
def calculate_saccade_count_ratio(df):
    saccade_df = df[df['classification'] == 'saccade']
    total_count = len(df)
    saccade_count = len(saccade_df)
    return saccade_count / total_count * 100 if total_count > 0 else 0

# 平均サッケード振幅（Average Saccade Amplitude）の計算
def calculate_avg_saccade_amplitude(df):
    saccade_df = df[df['classification'] == 'saccade']
    amplitudes = [euclidean((x1, y1), (x2, y2)) for x1, y1, x2, y2 in zip(saccade_df['X Coordinate'][:-1], saccade_df['Y Coordinate'][:-1], saccade_df['X Coordinate'][1:], saccade_df['Y Coordinate'][1:])]
    return np.mean(amplitudes) if amplitudes else 0

# 最大サッケード振幅（Max Saccade Amplitude）の計算
def calculate_max_saccade_amplitude(df):
    saccade_df = df[df['classification'] == 'saccade']
    amplitudes = [euclidean((x1, y1), (x2, y2)) for x1, y1, x2, y2 in zip(saccade_df['X Coordinate'][:-1], saccade_df['Y Coordinate'][:-1], saccade_df['X Coordinate'][1:], saccade_df['Y Coordinate'][1:])]
    return np.max(amplitudes) if amplitudes else 0

# 視線の分布（Gaze Dispersion）の計算
def calculate_gaze_dispersion(df):
    fixation_df = df[df['classification'] == 'fixation']
    if len(fixation_df) > 0:
        x_std = np.std(fixation_df['X Coordinate'])
        y_std = np.std(fixation_df['Y Coordinate'])
        return np.sqrt(x_std**2 + y_std**2)
    return 0

# 視線パスのフラクタル次元（Fractal Dimension）の計算
def calculate_fractal_dimension(df):
    if len(df) < 2:
        return 0
    x = df['X Coordinate'].values
    y = df['Y Coordinate'].values
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    total_distance = np.sum(distances)
    bounding_box = (np.ptp(x), np.ptp(y))
    bounding_box_diagonal = np.sqrt(bounding_box[0]**2 + bounding_box[1]**2)
    return total_distance / bounding_box_diagonal if bounding_box_diagonal > 0 else 0

# 平均視線パス直線性（Average Path Linearity）の計算
def calculate_avg_path_linearity(df):
    saccade_segments = []
    start_index = None

    for i in range(1, len(df)):
        if df['classification'].iloc[i-1] == 'fixation' and df['classification'].iloc[i] == 'saccade':
            start_index = i
        elif df['classification'].iloc[i-1] == 'saccade' and df['classification'].iloc[i] == 'fixation' and start_index is not None:
            saccade_segments.append((start_index, i))
            start_index = None

    linearities = []
    for start, end in saccade_segments:
        segment = df.iloc[start:end+1]
        if len(segment) < 2:
            continue
        start_point = (segment['X Coordinate'].iloc[0], segment['Y Coordinate'].iloc[0])
        end_point = (segment['X Coordinate'].iloc[-1], segment['Y Coordinate'].iloc[-1])
        direct_distance = euclidean(start_point, end_point)
        path_distance = np.sum(np.sqrt(np.diff(segment['X Coordinate'])**2 + np.diff(segment['Y Coordinate'])**2))
        linearities.append(direct_distance / path_distance if path_distance > 0 else 0)

    return np.mean(linearities) if linearities else 0

# メイン関数
def main():
    # CSVファイルの選択
    file_path = load_csv()
    df = pd.read_csv(file_path)

    # Record Timeを文字列に変換
    df['Record Time'] = df['Record Time'].astype(str)

    # 座標(0,0)は外れ値として扱う
    df = df[(df['X Coordinate'] != 0) & (df['Y Coordinate'] != 0)]

    # 特徴量の計算
    time_range = calculate_time_range(df)
    avg_velocity = calculate_avg_velocity(df)
    transition_count = calculate_transition_count(df)
    saccade_count_ratio = calculate_saccade_count_ratio(df)
    avg_saccade_amplitude = calculate_avg_saccade_amplitude(df)
    max_saccade_amplitude = calculate_max_saccade_amplitude(df)
    gaze_dispersion = calculate_gaze_dispersion(df)
    fractal_dimension = calculate_fractal_dimension(df)
    avg_path_linearity = calculate_avg_path_linearity(df)

    # 結果を表示
    print(time_range, avg_velocity, transition_count, saccade_count_ratio, avg_saccade_amplitude, max_saccade_amplitude, gaze_dispersion, fractal_dimension, avg_path_linearity, sep=',')

# 実行
if __name__ == "__main__":
    main()
