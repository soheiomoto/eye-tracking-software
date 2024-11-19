# 必要なモジュールのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# CSVファイル選択
Tk().withdraw()
file_path = askopenfilename(filetypes=[("CSV Files", "*.csv")])  # ファイルエクスプローラーを表示

# データの読み込み
data = pd.read_csv(file_path)
data = data[['timestamp', 'x', 'y']]

# timestampの範囲指定（hhmmss形式で指定）
start_time = input("開始時間を入力してください (hhmmss形式): ")
end_time = input("終了時間を入力してください (hhmmss形式): ")

# timestampをdatetime形式に変換
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%H%M%S').dt.strftime('%H%M%S')

# 指定した時間範囲でデータをフィルタリング
filtered_data = data[(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)]

# 座標(0,0)の除外（外れ値として扱う）
filtered_data = filtered_data[(filtered_data['x'] != 0) | (filtered_data['y'] != 0)]

# x, y座標のデータのみ取得
coordinates = filtered_data[['x', 'y']]

# データのスケーリング
scaler = StandardScaler()
data_scaled = scaler.fit_transform(coordinates)

# DBSCANモデルの設定とクラスタリング実行
dbscan = DBSCAN(eps=0.05, min_samples=10)  # epsは距離閾値、min_samplesは最小サンプル数
labels = dbscan.fit_predict(data_scaled)

# クラスタ数（-1はノイズとして扱われる）
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f'クラスタ数: {n_clusters}')

# クラスタリング結果の可視化
plt.figure(figsize=(10, 8))
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # ノイズは黒で表示
        col = 'k'
    
    class_member_mask = (labels == k)
    xy = coordinates[class_member_mask]
    
    plt.plot(xy['x'], xy['y'], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

# X軸とY軸の範囲設定
plt.xlim(0, 1920)
plt.ylim(0, 1080)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.gca().invert_yaxis()  # y軸反転（画面座標系に合わせる）
plt.show()