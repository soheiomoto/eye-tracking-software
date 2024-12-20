import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
from pingouin import welch_anova, pairwise_gameshowell
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

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

# データの読み込み
df = pd.read_csv(file_path)

# 特徴量の列名
features = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5',
            'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10']

# グループ情報
groups = df['Group'].unique()

# 結果を格納するリスト
welch_results = {}
games_howell_results = {}
kruskal_results = {}
dunn_results = {}

# 各特徴量に対してウェルチANOVAとKruskal-Wallis検定を実行
for feature in features:
    # ウェルチANOVA
    welch_result = welch_anova(dv=feature, between="Group", data=df)
    welch_results[feature] = welch_result

    # Games-Howell法による事後検定
    games_howell_result = pairwise_gameshowell(dv=feature, between="Group", data=df)
    games_howell_results[feature] = games_howell_result

    # Kruskal-Wallis検定（非パラメトリック検定）
    groups_data = [df[df['Group'] == group][feature].dropna() for group in groups]
    kruskal_result = kruskal(*groups_data)
    kruskal_results[feature] = kruskal_result
        # Dunn検定の修正
    dunn_results = {}
    for feature in features:
        dunn_result = posthoc_dunn(df, val_col=feature, group_col='Group', p_adjust='bonferroni')
        dunn_results[feature] = dunn_result

# 結果の表示
print("ウェルチANOVAの結果:")
for feature, result in welch_results.items():
    print(f'\n{feature} のウェルチANOVA結果:')
    print(result)

print("\nGames-Howell法の事後検定結果:")
for feature, result in games_howell_results.items():
    print(f'\n{feature} のGames-Howell法結果:')
    print(result)

print("\n非パラメトリック検定（Kruskal-Wallis）の結果:")
for feature, result in kruskal_results.items():
    print(f'\n{feature} のKruskal-Wallis検定結果:')
    print(f"統計量: {result.statistic}, p値: {result.pvalue}")

print("\nDunn検定（多重比較補正付き）の結果:")
for feature, result in dunn_results.items():
    print(f'\n{feature} のDunn検定結果:')
    print(result)
