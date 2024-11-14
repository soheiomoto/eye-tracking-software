import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CSVファイルの読み込み（例: "gaze_data.csv"）
df = pd.read_csv('C://Users//PC_User//iCloudDrive//Documents//01_University//02_Research//Working-Directory//eye-tracking-software//Experiment-Data-1st//1st-English//1st-English-J1//1st-English-J1-01.csv')

# データの確認
print("Original data:")
print(df.head())

# timestampの範囲を指定（例: 1000〜2000）
#timestamp_start = 163407
#timestamp_end = 163526

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

plt.title(f'Gaze Heatmap (Y-Axis Flipped in Output) for Timestamps {timestamp_start} to {timestamp_end}')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()
