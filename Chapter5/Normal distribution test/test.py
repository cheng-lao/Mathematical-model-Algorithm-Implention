# 使用JK检验 计算一组数据是否是符合正态分布的，中间用到了假设检验的方法!
import numpy as np
from scipy.stats import skew, kurtosis
from statsmodels.stats.stattools import jarque_bera

# 生成正态分布数据
data = np.random.normal(loc=2, scale=3, size=1000000)
# 均值为2 标准差为3的分布数据

# 计算偏度
data_skewness = skew(data)

# 计算峰度  计算出来的峰度默认减去了3，这里给他加上3
data_kurtosis = kurtosis(data) + 3

print(f"偏度: {data_skewness}")
print(f"峰度: {data_kurtosis}")

# 进行 JB 检验
jb_stat, jb_p_value, _, _ = jarque_bera(data)
# jb_stat：JB 检验统计量。
# jb_p_value：JB 检验的 p 值。如果 p 值大于 0.05，表示数据符合正态分布。

print(f"JB 检验统计量: {jb_stat}")
print(f"JB 检验 p 值: {jb_p_value}")

# 判断是否符合正态分布
if jb_p_value > 0.05:
    print("数据符合正态分布")
else:
    print("数据不符合正态分布")
