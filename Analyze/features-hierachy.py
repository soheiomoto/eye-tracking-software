import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ファイル選択ダイアログを表示
def select_file():
    Tk().withdraw()  # Tkinterのデフォルトウィンドウを非表示
    file_path = askopenfilename(
        title="CSVファイルを選択してください",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    return file_path

# ファイル選択
file_path = select_file()
if not file_path:
    print("ファイルが選択されませんでした。プログラムを終了します。")
    exit()

# 1. CSVファイルの読み込みとデータ構造の確認
data = pd.read_csv(file_path)

# データ構造の確認
print("データ構造:")
print(data.info())
print("\n欠損値:")
print(data.isnull().sum())

# 被験者IDを分離して、数値データを取得
subject_ids = data.iloc[:, 0]  # 最初の列が被験者IDと仮定
numerical_data = data.iloc[:, 1:]  # 残りの列が数値データ

# 数値列のみを抽出
numerical_data = numerical_data.apply(pd.to_numeric, errors='coerce')  # 数値に変換できないものはNaNにする

# 2. スケールの標準化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# 3. 距離行列の作成とクラスタリング (ウォード法)
linkage_matrix = linkage(scaled_data, method='ward')

# 4. デンドログラムの作成
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, labels=subject_ids.values, leaf_rotation=90, leaf_font_size=10)
plt.xlabel("Participant")
plt.ylabel("Distance")
plt.show()

# 5. クラスタ数4でクラスタ割り当て
num_clusters = 4
clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
data['Cluster'] = clusters

# 6. 数値列のみに対してクラスタごとの特徴抽出
# クラスタごとに平均値と分散を計算
cluster_summary = data.groupby('Cluster').agg({col: ['mean', 'var'] for col in numerical_data.columns})

print("\n各クラスタの平均値と分散:")
print(cluster_summary)

# クラスタの分布
cluster_distribution = data['Cluster'].value_counts()
print("\nクラスタ分布:")
print(cluster_distribution)

# クラスタを2Dプロット (PCAを使用して次元削減)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(8, 6))
for cluster in range(1, num_clusters + 1):
    cluster_points = reduced_data[clusters == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")

plt.title("Cluster Visualization (PCA Reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()
