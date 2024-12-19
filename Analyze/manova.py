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
group = data['グループ']  # グループ列を取り出す
features = data[['特徴量1', '特徴量2', '特徴量3', '特徴量4', '特徴量5',
                 '特徴量6', '特徴量7', '特徴量8', '特徴量9']]  # 特徴量列を取り出す

# MANOVAを実行
manova = MANOVA.from_formula('特徴量1 + 特徴量2 + 特徴量3 + 特徴量4 + 特徴量5 + '
                             '特徴量6 + 特徴量7 + 特徴量8 + 特徴量9 ~ グループ', data=data)

# 結果を表示
result = manova.mv_test()
print(result)

# グループのデータと特徴量を用意
group = data['グループ']  # グループ列
features = ['特徴量1', '特徴量2', '特徴量3', '特徴量4', '特徴量5',
            '特徴量6', '特徴量7', '特徴量8', '特徴量9']  # 特徴量列

# 各特徴量についてTukey HSDテストを実行
for feature in features:
    print(f"\n{feature}についてTukeyのHSDテスト:")
    
    # Tukey HSDテストを実行
    tukey_result = pairwise_tukeyhsd(endog=data[feature], groups=data['グループ'], alpha=0.05)
    
    # 結果を表示
    print(tukey_result.summary())