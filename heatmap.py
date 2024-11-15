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

    # timestampの範囲を指定
    timestamp_start = 163427
    timestamp_end = 163434

    # timestamp範囲でデータをフィルタリング
    df_filtered = df[(df['timestamp'] >= timestamp_start) & (df['timestamp'] <= timestamp_end)]

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

    # Y軸の反転（ここで反転を行う）
    plt.gca().invert_yaxis()

    #plt.title(f'Gaze Heatmap (Y-Axis Flipped in Output) for Timestamps {timestamp_start} to {timestamp_end}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()
else:
    print("No file selected.")
