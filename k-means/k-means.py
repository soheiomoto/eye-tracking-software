import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# サンプルの成績データ（26人分）
scores = np.array([75, 85, 92, 60, 70, 95, 65, 80, 88, 55, 91, 78, 85, 70, 67, 72, 95, 60, 85, 90, 83, 87, 65, 74, 93, 76]).reshape(-1, 1)

# データのスケーリング（平均0, 標準偏差1に調整）
scaler = StandardScaler()
scores_scaled = scaler.fit_transform(scores)

# K-means++によるクラスタリング
k = 3  # クラスタ数（例：3つのグループに分ける）
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
kmeans.fit(scores_scaled)

# クラスタリング結果
labels = kmeans.labels_  # 各データポイントのクラスタのラベル
centroids = kmeans.cluster_centers_  # 各クラスタの中心

# 結果をプロット
plt.figure(figsize=(8, 6))
plt.scatter(scores, np.zeros_like(scores), c=labels, cmap='viridis', s=100, edgecolors='k', marker='o', label='Data points')
plt.scatter(centroids, np.zeros_like(centroids), c='red', s=300, marker='X', label='Centroids')

# グラフの設定
plt.title('K-means++ クラスタリング結果')
plt.xlabel('テストの成績')
plt.ylabel('クラスタ')
plt.yticks([])  # Y軸の目盛りを消す（1次元なので意味がない）
plt.legend()
plt.show()

# 各クラスタに所属する成績の表示
for i in range(k):
    print(f"クラスタ {i+1} の成績: {scores[labels == i].flatten()}")