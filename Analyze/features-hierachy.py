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

# NaNを含む行を削除
valid_data = numerical_data.dropna()

# 2. スケールの標準化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(valid_data)

# 3. 距離行列の作成とクラスタリング (ウォード法)
linkage_matrix = linkage(scaled_data, method='ward')

# 4. デンドログラムの作成
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, labels=subject_ids.values, leaf_rotation=90, leaf_font_size=10)
plt.xlabel("Participants", fontsize=14)
plt.ylabel("Distance", fontsize=14)
plt.show()

# 5. クラスタ数4でクラスタ割り当て
num_clusters = 4
clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
data['Cluster'] = clusters

# 6. 数値列のみに対してクラスタごとの特徴抽出
# クラスタごとに平均値と分散を計算
cluster_summary = data.groupby('Cluster').agg({col: ['mean', 'var'] for col in valid_data.columns})

print("\n各クラスタの平均値と分散:")
print(cluster_summary)

# クラスタの分布
cluster_distribution = data['Cluster'].value_counts()
print("\nクラスタ分布:")
print(cluster_distribution)

# PCAで次元削減 (手動で重みを指定)
# ユーザーが各成分の重みを指定する
custom_weights = np.array([
    [0.408796, 0.348461],
    [-0.402395, 0.312792],
    [-0.194278, 0.383311],
    [0.453299, 0.224352],
    [-0.429541, 0.257102],
    [0.097201, 0.577437],
    [-0.447025, -0.183308],
    [0.184270, -0.387735],
])
if custom_weights.shape[0] != scaled_data.shape[1]:
    raise ValueError("重みの行数がデータの次元数と一致していません。")

# 重みを適用して次元削減
reduced_data_custom = scaled_data @ custom_weights

# クラスタ別の色を指定
cluster_colors = {
    1: '#1f77b4',  # 青
    2: '#ff7f0e',  # オレンジ
    3: '#2ca02c',  # 緑
    4: '#d62728',  # 赤
    5: '#bd81dc'   # 紫
}

# 重心の色を指定
centroid_colors = {
    1: '#1f77b4',  # 濃い青
    2: '#e56c00',  # 濃いオレンジ
    3: '#006400',  # 濃い緑
    4: '#b20304',  # 濃い赤
    5: '#a157c8'   # 濃い紫
}

# プロット
plt.figure(figsize=(10, 8))
for cluster in range(1, num_clusters + 1):
    cluster_points = reduced_data_custom[clusters == cluster]
    plt.scatter(
        cluster_points[:, 0], cluster_points[:, 1],
        label=f"Cluster {cluster}",
        color=cluster_colors[cluster],  # クラスタデータの色
        alpha=0.6, s=50
    )

# 重心を計算してプロット
for cluster in range(1, num_clusters + 1):
    cluster_data = reduced_data_custom[clusters == cluster]
    centroid = cluster_data.mean(axis=0)
    plt.scatter(
        centroid[0], centroid[1],
        marker='v', color=centroid_colors[cluster],  # 重心の色
        s=200, label=f"Centroid {cluster}"
    )

# グラフの装飾
plt.xlabel("Efficient Concentrated Behaviour", fontsize=14)
plt.ylabel("Visual Exploratory Behaviour", fontsize=14)
plt.legend(fontsize=10)
plt.show()