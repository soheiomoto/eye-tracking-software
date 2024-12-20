import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
from pingouin import welch_anova, pairwise_gameshowell
from scipy.stats import kruskal, f_oneway
from scikit_posthocs import posthoc_dunn
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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
anova_results = {}
tukey_results = {}

# 各特徴量に対してウェルチANOVA、通常のANOVA、Kruskal-Wallis検定を実行
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

    # Dunn検定の修正（多重比較補正を追加）
    dunn_result = posthoc_dunn(df, val_col=feature, group_col='Group', p_adjust='bonferroni')  # 'bonferroni'を変更可能
    dunn_results[feature] = dunn_result

    # 通常のANOVA
    anova_groups_data = [df[df['Group'] == group][feature].dropna() for group in groups]
    anova_result = f_oneway(*anova_groups_data)
    anova_results[feature] = anova_result

    # Tukeyの事後検定
    model = ols(f'{feature} ~ C(Group)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    tukey_result = pairwise_tukeyhsd(df[feature], df['Group'])
    tukey_results[feature] = tukey_result

# 結果の表示
print("\n通常のANOVAの結果:")
for feature, result in anova_results.items():
    print(f'\n{feature} の通常のANOVA結果:')
    print(f"F値: {result.statistic}, p値: {result.pvalue}")

print("\nTukeyの事後検定結果:")
for feature, result in tukey_results.items():
    print(f'\n{feature} のTukey事後検定結果:')
    print(result)

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