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

# 3. PCAで次元削減 (自動で重みを設定)
pca = PCA(n_components=2)  # 2次元に削減
pca_result = pca.fit_transform(scaled_data)  # データをPCAで変換

# 4. 距離行列の作成とクラスタリング (ウォード法)
linkage_matrix = linkage(scaled_data, method='ward')

# 5. クラスタリング結果に基づいてラベルを調整
valid_subject_ids = subject_ids[valid_data.index]  # 欠損値を削除した後のsubject_ids
if len(linkage_matrix) != len(valid_subject_ids) - 1:
    print("警告: クラスタ数とラベルの数が一致しません。")

# 6. デンドログラムの作成
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, labels=valid_subject_ids.values, leaf_rotation=90, leaf_font_size=10)
#plt.axhline(y=5, color='r', linestyle='--', label='Cutoff Distance: 5')  # カットオフラインを追加
#plt.legend(fontsize=12)  # 凡例を追加
plt.xlabel("Participants", fontsize=15)
plt.ylabel("Distance", fontsize=15)
plt.show()

# 6. クラスタ数4でクラスタ割り当て
num_clusters = int(input("クラスタ数を入力してください（デフォルト: 4）: ") or 4)
# クラスタリング結果を取得
clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

# 元のデータに統合
# 有効なデータに基づくインデックスにのみクラスタ情報を割り当て
data['Cluster'] = np.nan  # 初期値としてNaNを設定
data.loc[valid_data.index, 'Cluster'] = clusters  # クラスタを有効データのインデックスに割り当て

# 7. 数値列のみに対してクラスタごとの特徴抽出
# クラスタごとに平均値と分散を計算
cluster_summary = data.groupby('Cluster').agg({col: ['mean', 'var'] for col in valid_data.columns})

print("\n各クラスタの平均値と分散:")
print(cluster_summary)

# クラスタの分布
cluster_distribution = data['Cluster'].value_counts()
print("\nクラスタ分布:")
print(cluster_distribution)

# PCA成分の確認
pca_components = pd.DataFrame(
    pca.components_,
    columns=numerical_data.columns,
    index=[f"PCA Component {i+1}" for i in range(pca.n_components_)]
)
print("\nPCA成分（主成分負荷量）:")
print(pca_components)

# クラスタ別の色を指定
cluster_colors = {
    1: '#1f77b4',  # 青
    2: '#ff7f0e',  # オレンジ
    3: '#2ca02c',  # 緑
    4: '#d62728',  # 赤
    5: '#bd81dc',  # 紫
    6: '#deb887',  # 茶色
    7: '#f0e68c'   # 黄色
}

# 重心の色を指定
centroid_colors = {
    1: '#1f77b4',  # 濃い青
    2: '#e56c00',  # 濃いオレンジ
    3: '#006400',  # 濃い緑
    4: '#b20304',  # 濃い赤
    5: '#a157c8',  # 濃い紫
    6: '#8b4513',  # 濃い茶色
    7: '#ffd700'   # 濃い黄色
}

# プロット
plt.figure(figsize=(10, 8))
for cluster in range(1, num_clusters + 1):
    cluster_points = pca_result[clusters == cluster]
    plt.scatter(
        cluster_points[:, 0], cluster_points[:, 1],
        label=f"Cluster {cluster}",
        color=cluster_colors[cluster],  # クラスタデータの色
        alpha=0.6, s=50
    )

# 重心を計算してプロット
for cluster in range(1, num_clusters + 1):
    cluster_data = pca_result[clusters == cluster]
    centroid = cluster_data.mean(axis=0)
    plt.scatter(
        centroid[0], centroid[1],
        marker='v', color=centroid_colors[cluster],  # 重心の色
        s=200, label=f"Centroid {cluster}"
    )

# グラフの装飾
plt.xlabel("PCA Component 1", fontsize=15)
plt.ylabel("PCA Component 2", fontsize=15)
plt.legend(fontsize=10)
plt.show()