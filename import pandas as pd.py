import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CSVファイルの読み込み（例: "gaze_data.csv"）
df = pd.read_csv('C://Users//www//Desktop//analyze//1st-Japanese-J1-01.csv')  # 'timestamp', 'x', 'y' カラムがあると仮定

# データの確認
print("Original data:")
print(df.head())

# Y座標を反転
df['y'] = -df['y']

# 座標が(0, 0)のデータを外れ値として除外
df = df[(df['x'] != 0) | (df['y'] != 0)]

# ヒートマップ作成のためのデータ準備
x_values = df['x']
y_values = df['y']

# ヒートマップを描画
plt.figure(figsize=(10, 6))
sns.kdeplot(x=x_values, y=y_values, cmap="viridis", fill=True, bw_adjust=0.5, cbar=True)

# 画面サイズ（1920x1080）に合わせてプロット範囲を調整
plt.xlim(0, 1920)
plt.ylim(-1080, 0)

plt.title('Gaze Heatmap (Y-Axis Flipped)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()
