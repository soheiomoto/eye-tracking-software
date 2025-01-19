# 必要なモジュールのインポート
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import levene, shapiro
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
df = pd.read_csv(file_path)

# グループ列を取得
groups = df['Group'].unique()

# 特徴量のリスト
features = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10']

# 正規性の検定: Shapiro-Wilk検定
def check_normality(data):
    stat, p_value = shapiro(data)
    return p_value

# 等分散性の検定: Levene検定
def check_variance(*args):
    stat, p_value = levene(*args)
    return p_value

# グループごとに正規性と等分散性を確認
for feature in features:
    print(f'--- {feature} ---')

    # グループごとのデータを取得
    group_data = [df[df['Group'] == group][feature].dropna() for group in groups]

    # 正規性の検定（Shapiro-Wilk）
    normality_p_values = [check_normality(group) for group in group_data]
    print(f'正規性の検定 (Shapiro-Wilk):')
    for i, group in enumerate(groups):
        print(f'  {group}グループのp値: {normality_p_values[i]}')

    # 等分散性の検定（Levene）
    homogeneity_p_value = check_variance(*group_data)
    print(f'等分散性の検定 (Levene): p値 = {homogeneity_p_value}')

    print("\n")