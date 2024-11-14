from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# 成績データ
scores = np.array([20, 20, 20, 30, 25, 20, 25, 25, 20, 10, 5, 15, 10, 10]).reshape(-1, 1)

# シルエットスコアを計算
silhouette_scores = []
k_range = range(2, 15)  # K=2からK=10まで試す（最低でも2クラスタ必要）

for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(scores)
    score = silhouette_score(scores, kmeans.labels_)
    silhouette_scores.append(score)

# 最適なKを見つける
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"最適なクラスタ数は: {optimal_k}")

# プロット
plt.plot(k_range, silhouette_scores, marker='o')
plt.xlabel('クラスタ数 (K)')
plt.ylabel('シルエットスコア')
plt.title('シルエット法')
plt.show()
