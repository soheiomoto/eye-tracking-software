import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np

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
data = pd.read_csv(file_path)

# グループと特徴量のデータを取り出す
group = data['Group']  # グループ列を取り出す
features = data[['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10']]  # 特徴量列を取り出す

# MANOVAを実行
manova = MANOVA.from_formula('Feature1 + Feature2 + Feature3 + Feature4 + Feature5 + '
                             'Feature6 + Feature7 + Feature8 + Feature9 + Feature10 ~ Group', data=data)

# 結果を表示
result = manova.mv_test()
print(result)

# グループのデータと特徴量を用意
group = data['Group']  # グループ列
features = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5',
            'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10']  # 特徴量列

# 各特徴量についてTukey HSDテストを実行
for feature in features:
    print(f"\n{feature}についてTukeyのHSDテスト:")
    
    # Tukey HSDテストを実行
    tukey_result = pairwise_tukeyhsd(endog=data[feature], groups=data['Group'], alpha=0.05)
    
    # 結果を表示
    print(tukey_result.summary())