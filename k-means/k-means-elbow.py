import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# 成績データ
scores = np.array([20, 20, 20, 30, 25, 20, 25, 25, 20, 10, 5, 15, 10, 10]).reshape(-1, 1)

# SSEを計算
sse = []
k_range = range(1, 15)  # 1から10までのKを試す

for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(scores)
    sse.append(kmeans.inertia_)  # SSE（誤差の平方和）

# プロット
plt.plot(k_range, sse, marker='o')
plt.xlabel('クラスタ数 (K)')
plt.ylabel('SSE (誤差の平方和)')
plt.title('エルボー法')
plt.show()
