from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import webbrowser

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

# CSVファイルの読み込みとデータ構造の確認
data = pd.read_csv(file_path)

# 被験者IDを分離して、数値データを取得
subject_ids = data.iloc[:, 0]  # 最初の列が被験者IDと仮定
numerical_data = data.iloc[:, 1:]  # 残りの列が数値データ

# 数値列のみを抽出
numerical_data = numerical_data.apply(pd.to_numeric, errors='coerce')  # 数値に変換できないものはNaNにする

# NaNを含む行を削除
valid_data = numerical_data.dropna()

# スケールの標準化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(valid_data)

# 3次元PCA
pca_3d = PCA(n_components=3)
pca_result_3d = pca_3d.fit_transform(scaled_data)
explained_variance_ratio_3d = pca_3d.explained_variance_ratio_

# クラスタリング
linkage_matrix = linkage(scaled_data, method='ward')
num_clusters = int(input("クラスタ数を入力してください（デフォルト: 4）: ") or 4)
clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

# クラスタ結果をDataFrameに統合
result_df = pd.DataFrame(pca_result_3d, columns=['PC1', 'PC2', 'PC3'])
result_df['Cluster'] = clusters
result_df['SubjectID'] = subject_ids[valid_data.index].values

# 3次元プロット（Plotly）
fig_3d = px.scatter_3d(
    result_df, x='PC1', y='PC2', z='PC3', color='Cluster', symbol='Cluster',
    hover_name='SubjectID', title='3D PCA Cluster Visualization'
)
fig_3d.update_traces(marker=dict(size=6))
fig_3d.update_layout(
    scene=dict(
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        zaxis_title='PCA Component 3'
    )
)

# グラフを一時ファイルに保存してブラウザで表示
def display_plotly_figure(fig):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
        fig.write_html(temp_file.name)
        webbrowser.open(f"file://{temp_file.name}")

display_plotly_figure(fig_3d)

# 既存の3次元プロットを生成
def save_plotly_figure(fig, filename):
    fig.write_html(filename)

# 一時ファイルではなく、GitHubにアップロード用のファイル名で保存
file_path_to_save = "pca_3d_plot.html"
save_plotly_figure(fig_3d, file_path_to_save)