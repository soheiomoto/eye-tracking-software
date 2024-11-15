"""
  Dependencies:
    fsspec                    2024.10.0
    matplotlib                3.9.2
    numpy                     2.1.2
    pandas                    2.2.3
    pip                       24.3.1
    scikit-learn              1.5.2
    scipy                     1.14.1
    seaborn                   0.13.2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 1. データの読み込み
data = pd.read_csv('C://Users//www//Desktop//analyze//1st-Japanese-J1-01.csv')  # 視線データCSVを読み込み
data = data[['x', 'y']]  # x, y座標のみ抽出

# 2. データのスケーリング
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 3. DBSCANモデルの設定とクラスタリング実行
dbscan = DBSCAN(eps=0.05, min_samples=10)  # epsは距離閾値、min_samplesは最小サンプル数
labels = dbscan.fit_predict(data_scaled)

# クラスタ数（-1はノイズとして扱われる）
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f'クラスタ数: {n_clusters}')

# 4. クラスタリング結果の可視化
plt.figure(figsize=(10, 8))
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # ノイズは黒で表示
        col = 'k'
    
    class_member_mask = (labels == k)
    xy = data[class_member_mask]
    
    plt.plot(xy['x'], xy['y'], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

plt.title('Gaze Clustering (DBSCAN)')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.gca().invert_yaxis()
plt.show()
