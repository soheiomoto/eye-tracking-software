import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import kruskal

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
features = ['特徴量1', '特徴量2', '特徴量3', '特徴量4', '特徴量5', '特徴量6', '特徴量7', '特徴量8', '特徴量9']

# グループ情報
groups = df['グループ'].unique()

# 結果を格納するリスト
anova_results = {}
kruskal_results = {}

# 各特徴量に対してウェルチANOVAとKruskal-Wallis検定を実行
for feature in features:
    # ウェルチANOVA
    model = ols(f'{feature} ~ C(グループ)', data=df).fit()
    anova_result = anova_lm(model, typ=2)
    anova_results[feature] = anova_result

    # Kruskal-Wallis検定（非パラメトリック検定）
    groups_data = [df[df['グループ'] == group][feature].dropna() for group in groups]
    kruskal_result = kruskal(*groups_data)
    kruskal_results[feature] = kruskal_result

# 結果の表示
print("ウェルチANOVAの結果:")
for feature, result in anova_results.items():
    print(f'\n{feature} のウェルチANOVA結果:')
    print(result)

print("\n非パラメトリック検定（Kruskal-Wallis）の結果:")
for feature, result in kruskal_results.items():
    print(f'\n{feature} のKruskal-Wallis検定結果:')
    print(f"統計量: {result.statistic}, p値: {result.pvalue}")
