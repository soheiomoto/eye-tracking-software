import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from datetime import datetime

# 画面サイズ
WIDTH, HEIGHT = 1920, 1080

# 関数: CSVファイルを選択
def select_csv_file():
    root = tk.Tk()
    root.withdraw()  # GUIを表示しない
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    return file_path

# 関数: 時間の範囲でデータをフィルタリング
def filter_data_by_timestamp(df, start_time, end_time):
    # timestamp列をdatetimeに変換
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%H%M%S", errors='coerce')
    # start_timeとend_timeをdatetimeに変換
    start_time = datetime.strptime(start_time, "%H%M%S")
    end_time = datetime.strptime(end_time, "%H%M%S")
    # 時間範囲でフィルタリング
    return df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

# 関数: 視線軌跡を矢印で表示（大雑把に）
def plot_gaze_trajectory(df, step=1.0):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)  # Y軸反転

    # 座標(0,0)を外れ値として除外し、x, yを数値型に変換
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    
    df_filtered = df[(df['x'] != 0) & (df['y'] != 0)]  # 座標(0,0)を除外

    # 無効なデータ（NaN）がある場合は除外
    df_filtered = df_filtered.dropna(subset=['x', 'y'])

    # データを一定の間隔（例えば10行）でサンプリング
    sampled_df = df_filtered.iloc[::step]

    # 視線の軌跡を矢印で表示（大雑把に）
    for i in range(len(sampled_df) - 1):
        x1, y1 = sampled_df.iloc[i]['x'], sampled_df.iloc[i]['y']
        x2, y2 = sampled_df.iloc[i + 1]['x'], sampled_df.iloc[i + 1]['y']
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle="->", lw=1))

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

# メイン関数
def main():
    # CSVファイルを選択
    file_path = select_csv_file()

    if file_path:
        # CSVファイルを読み込む（ヘッダー行を無視して手動でカラム名を指定）
        df = pd.read_csv(file_path, names=["timestamp", "x", "y"], header=None)

        # 時間範囲を設定
        start_time = "163427"  # 例: 09:00:00
        end_time = "163434"    # 例: 09:15:00

        # データを時間範囲でフィルタリング
        df_filtered = filter_data_by_timestamp(df, start_time, end_time)

        # 視線軌跡をプロット（間引き表示）
        plot_gaze_trajectory(df_filtered, step=1.0)
    else:
        print("ファイルが選択されていません")

if __name__ == "__main__":
    main()
