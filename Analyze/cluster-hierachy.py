import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# 成績データ（例としてランダムに生成）
data = np.array([
    [90],  # A
    [70],  # B
    [70],  # A
    [90],  # B
    [80],
    [90],  # D
    [60],  # E
    [80],  # F
    [80],  # G
    [90],  # H
    [80],  # I
    [90],   # J
    [80],  # A
    [100],  # B
    [80],  # C
    [90],
    [90],
    [90],  # E
    [100],  # F
    [100],  # G
    [90],  # H
    [90],  # I
    [100],
    [100],
    [70],
    [90],   # J
    [90],
    [80],
    [100],
    [90],
    [80],
    [100],
])

# 距離行列の計算（ユークリッド距離）
Z = sch.linkage(data, method='average')  # 'average'は平均リンク法

# デンドログラムの描画
plt.figure(figsize=(10, 7))
sch.dendrogram(Z, labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32'])
#plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Participant')
plt.ylabel('Distance')
plt.show()