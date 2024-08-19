import numpy as np
from scipy.stats import spearmanr
import pandas as pd
from matplotlib import pyplot as plt

# # 示例数据
# x = np.array([1, 2, 3, 4, 5])
# y = np.array([5, 6, 7, 8, 7])

# # 计算 Spearman 相关系数
# spearman_corr, spearman_p_value = spearmanr(x, y)
# print(f"Spearman 相关系数: {spearman_corr}, p值: {spearman_p_value}")

# # 计算 Pearson 相关系数
# pearson_corr = np.corrcoef(x, y)[0, 1]
# print(f"Pearson 相关系数: {pearson_corr}")

manpath = r"correlation\八年级男生体测数据.xls"
womanpath = r"correlation\八年级女生体测数据.xls"
mandata = pd.read_excel(manpath)
print(mandata)

#  计算流程 首先可以计算一下关于数据的描述性信息
print("数据框的前几行:")
print(mandata.head())

# 计算描述性统计信息
print("\n描述性统计信息:")
print(mandata.describe())

# 计算每列的均值
print("\n每列的均值:")
print(mandata.mean())

# 计算每列的中位数
print("\n每列的中位数:")
print(mandata.median())

# 计算每列的标准差
print("\n每列的标准差:")
print(mandata.std())

# 计算每列的相关系数矩阵
print("\n相关系数矩阵Pearson:")
print(mandata.corr())
print(f"\nspearman相关系数矩阵:\n", mandata.corr(method="spearman"))

# 第二步 绘制每列数据之间的散点图，查看线性程度
colnum = mandata.shape[1]
print("colnum is ", colnum)
headername = mandata.columns
print("headername : ")
print(headername)

fig, axs = plt.subplots(colnum, colnum, figsize=(80, 80))
figidx = 0
for i in range(colnum):
    # print("i is ", i)
    for j in range(colnum):
        if i == j:
            continue
        # print("i :", i, " j :", j)
        x = mandata.iloc[1:, i]
        y = mandata.iloc[1:, j]
        axs[i][j].scatter(x, y)
        axs[i][j].set_xlabel(headername[i])
        axs[i][j].set_ylabel(headername[j])
print("loop over!")
plt.tight_layout()
plt.show()


# 发现其实并没有线性关系，这个时候可以使用person但是不能做显著性检验了!
#
