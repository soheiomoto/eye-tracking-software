from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
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

# クラスタ別の色設定
cluster_colors = {
    1: 'blue',
    2: 'orange',
    3: 'green',
    4: 'red'
}

# 重心の色設定
centroid_colors = {
    1: 'darkblue',
    2: 'darkorange',
    3: 'darkgreen',
    4: 'darkred'
}

# 各クラスタの重心を計算
centroids = result_df.groupby('Cluster')[['PC1', 'PC2', 'PC3']].mean()

# 記号の設定（クラスタごとに異なる記号を設定）
cluster_symbols = {
    1: 'circle',
    2: 'circle',
    3: 'circle',
    4: 'circle'
}

# 3次元プロット（Plotly）
fig_3d = go.Figure()

# 各クラスタのデータポイントをプロット
for cluster in range(1, num_clusters + 1):
    cluster_data = result_df[result_df['Cluster'] == cluster]
    fig_3d.add_trace(go.Scatter3d(
        x=cluster_data['PC1'],
        y=cluster_data['PC2'],
        z=cluster_data['PC3'],
        mode='markers',
        marker=dict(size=6, color=cluster_colors[cluster], symbol=cluster_symbols[cluster]),
        name=f'Cluster {cluster}',
        hovertext=cluster_data['SubjectID']
    ))

# 各クラスタの重心をプロット
for cluster in range(1, num_clusters + 1):
    centroid = centroids.loc[cluster]
    fig_3d.add_trace(go.Scatter3d(
        x=[centroid['PC1']],
        y=[centroid['PC2']],
        z=[centroid['PC3']],
        mode='markers',
        marker=dict(size=7, color=centroid_colors[cluster], symbol='x', line=dict(width=1, color='black')),
        name=f'Centroid {cluster}'
    ))

# グラフのレイアウト
fig_3d.update_layout(
    title='3D PCA Cluster Visualization with Centroids',
    scene=dict(
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        zaxis_title='PCA Component 3'
    ),
    legend=dict(title="Clusters")
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
file_path_to_save = "PCA_3D_Graph.html"
save_plotly_figure(fig_3d, file_path_to_save)
