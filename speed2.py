import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter1d

# 視線データの読み込み（CSVファイル）
df = pd.read_csv('C://Users//PC_User//iCloudDrive//Documents//01_University//02_Research//Working-Directory//eye-tracking-software//Experiment-Data-1st//1st-English//1st-English-J1//1st-English-J1-01.csv')

# 列名に余分な空白がある場合、それを削除する
df.columns = df.columns.str.strip()

# タイムスタンプがhhmmss形式なので、'hhmmss'形式の文字列を時間に変換
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H%M%S').dt.time

# 座標(0, 0)を外れ値として扱い、除外する
df = df[(df['X'] != 0) | (df['Y'] != 0)]

# Y座標を反転させるために、Yの最大値を取得し、Y座標を反転させる
max_y = df['Y'].max()
df['Y'] = max_y - df['Y']

# タイムスタンプの範囲でデータをフィルタリングする関数
def filter_by_time(df, start_time, end_time):
    # 'hhmmss' の形式の文字列を time 型に変換し、範囲でフィルタリング
    start_time = pd.to_datetime(start_time, format='%H%M%S').time()
    end_time = pd.to_datetime(end_time, format='%H%M%S').time()
    
    df_filtered = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
    return df_filtered

# タイムスタンプ範囲の指定
start_time = '163407'  # 開始時刻 (例: '010000' は 01:00:00)
end_time = '163426'    # 終了時刻 (例: '120000' は 12:00:00)

# タイムスタンプの範囲でデータをフィルタリング
df_filtered = filter_by_time(df, start_time, end_time)

# 注視点（フィクセーション）の抽出と注視時間の計算
def extract_fixations(df, eps=5, min_samples=2):
    # DBSCANクラスタリングで視線の注視点を抽出
    coords = df[['X', 'Y']].values
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['fixation'] = dbscan.fit_predict(coords)
    
    # 注視点ごとのデータをまとめる
    fixations = df[df['fixation'] != -1]  # -1 はノイズ（クラスタに属さない点）
    fixation_groups = fixations.groupby('fixation').agg({'X': 'mean', 'Y': 'mean', 'timestamp': 'count'}).reset_index()
    fixation_groups.rename(columns={'timestamp': 'duration'}, inplace=True)  # duration = 注視時間（滞在フレーム数）
    return fixation_groups

# フィルタリング後のデータから注視点を抽出
fixations = extract_fixations(df_filtered)

# 注視点とその注視時間を表示
print(fixations)

# **移動経路のスムージング**:
# 注視点の移動経路をスムージングして、密集を減らす
# X座標とY座標それぞれに対してガウス平滑化を適用
smoothed_x = gaussian_filter1d(df_filtered['X'], sigma=3)  # sigmaでスムージング強度を調整
smoothed_y = gaussian_filter1d(df_filtered['Y'], sigma=3)

# **移動経路の間引き**:
# 例えば、数百ミリ秒ごとに1点を選択して描画する方法
sampling_interval = 10  # 10フレームごとにサンプリング
sampled_x = smoothed_x[::sampling_interval]
sampled_y = smoothed_y[::sampling_interval]

# 画面サイズ（フルHD: 1920x1080）に合わせて座標をスケーリング
screen_width = 1920
screen_height = 1080

# スケーリング（0, 0 から画面サイズに合わせる）
scaled_x = (sampled_x / max(sampled_x)) * screen_width
scaled_y = (sampled_y / max(sampled_y)) * screen_height

# **注視点の移動経路を描画**:
# 注視点の順番に矢印を描くため、quiver関数を使って矢印を表示
plt.figure(figsize=(12, 10))

# 矢印を描く（x, yの位置から次のx, y位置への矢印）
dx = np.diff(scaled_x)
dy = np.diff(scaled_y)

# `quiver` 関数を使用して矢印を描画
plt.quiver(scaled_x[:-1], scaled_y[:-1], dx, dy, angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.6, width=0.003)

# 注視点を小さめにプロット（赤い点のサイズを小さく設定）
plt.scatter(scaled_x, scaled_y, color='red', s=25, label='Fixation Points', zorder=5)  # 点のサイズを小さめに変更
plt.title("Fixation Path with Arrows Indicating Movement", fontsize=16)
plt.xlabel("X Coordinate", fontsize=12)
plt.ylabel("Y Coordinate", fontsize=12)
plt.xlim(0, screen_width)  # X軸の範囲を1920に設定
plt.ylim(0, screen_height)  # Y軸の範囲を1080に設定
plt.gca().invert_yaxis()  # Y軸を反転
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
plt.legend()
plt.show()

# ヒートマップ作成のために注視時間を計算した座標を取得
heatmap_data = np.zeros((screen_height, screen_width))  # 画面サイズに合わせて

for _, row in fixations.iterrows():
    x, y, duration = int(row['X']), int(row['Y']), row['duration']
    if x < screen_width and y < screen_height:  # 画面サイズ内に収める
        heatmap_data[y, x] += duration

# ヒートマップ表示
plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_data, cmap='YlGnBu', cbar=True)
plt.title("Heatmap of Fixation Duration", fontsize=16)
plt.xlabel("X Coordinate", fontsize=12)
plt.ylabel("Y Coordinate", fontsize=12)
plt.xlim(0, screen_width)  # X軸の範囲を1920に設定
plt.ylim(0, screen_height)  # Y軸の範囲を1080に設定
plt.gca().invert_yaxis()  # Y軸を反転
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
plt.show()

# 注視点をプロット（注視時間を点の大きさで示す）
plt.figure(figsize=(12, 10))
scatter = plt.scatter(scaled_x, scaled_y, s=fixations['duration']*5, c=fixations['duration'], cmap='YlGnBu', alpha=0.7)
plt.colorbar(scatter, label="Fixation Duration")
plt.title("Fixation Points and Duration", fontsize=16)
plt.xlabel("X Coordinate", fontsize=12)
plt.ylabel("Y Coordinate", fontsize=12)
plt.xlim(0, screen_width)  # X軸の範囲を1920に設定
plt.ylim(0, screen_height)  # Y軸の範囲を1080に設定
plt.gca().invert_yaxis()  # Y軸を反転
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
plt.show()
