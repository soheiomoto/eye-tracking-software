import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

# 成績データ
scores = np.array([50, 35]).reshape(-1, 1)

# BICを計算して最適なクラスタ数を選ぶ
bic_scores = []
k_range = range(1, 3)  # 1から10までのクラスタ数を試す

for k in k_range:
    gmm = GaussianMixture(n_components=k, random_state=0)
    gmm.fit(scores)
    bic_scores.append(gmm.bic(scores))  # BICを計算

# BICのプロット
plt.plot(k_range, bic_scores, marker='o')
plt.xlabel('クラスタ数 (K)')
plt.ylabel('BIC')
plt.title('BICを用いたクラスタ数の選択')
plt.show()

# 最適なK（BICが最小となるK）を選択
optimal_k = k_range[np.argmin(bic_scores)]
print(f"最適なクラスタ数は: {optimal_k}")
