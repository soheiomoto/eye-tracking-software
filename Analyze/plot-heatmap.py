#proceedでも入力可能

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Tk, filedialog

# ファイル選択ダイアログを表示
root = Tk()
root.withdraw()  # Tkinterのメインウィンドウを非表示にする
file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV Files", "*.csv")])

# CSVファイルの読み込み
if file_path:
    df = pd.read_csv(file_path)

    # データの確認
    print("Original data:")
    print(df.head())

    # 複数のタイムスタンプ範囲を指定
    timestamp_ranges = [(163407, 163426), (163427, 163434)]  # 例: 複数の範囲

    # フィルタリングしたデータを格納するためのデータフレーム
    df_filtered = pd.DataFrame()

    # 各タイムスタンプ範囲でデータをフィルタリング
    for start, end in timestamp_ranges:
        df_range = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
        df_filtered = pd.concat([df_filtered, df_range], ignore_index=True)

    # 座標が(0, 0)のデータを外れ値として除外
    df_filtered = df_filtered[(df_filtered['x'] != 0) | (df_filtered['y'] != 0)]

    # ヒートマップ作成のためのデータ準備
    x_values = df_filtered['x']
    y_values = df_filtered['y']

    # ヒートマップを描画
    plt.figure(figsize=(10, 6))
    sns.kdeplot(x=x_values, y=y_values, cmap="viridis", fill=True, bw_adjust=0.5, cbar=True)

    # 画面サイズ（1920x1080）に合わせてプロット範囲を調整
    plt.xlim(0, 1920)
    plt.ylim(0, 1080)

    # Y軸の反転
    plt.gca().invert_yaxis()

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Gaze Heatmap (Multiple Timestamp Ranges)')
    plt.show()
else:
    print("No file selected.")
