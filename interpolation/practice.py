import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline  # 最精准
from scipy.interpolate import PchipInterpolator  # 次精准

population = [
    133126,
    133770,
    134413,
    135069,
    135738,
    136427,
    137122,
    137866,
    138639,
    139538,
]
year = range(2009, 2019)
year = np.array(year)

x_new = range(2019, 2023)  # 预测未来三年的人口

population = np.array(population)
predice1 = CubicSpline(year, population)
y1_new = predice1(x_new)

predice2 = PchipInterpolator(year, population)
y2_new = predice2(x_new)

plt.plot(year, population, "-", label="known data")
plt.scatter(x_new, y1_new, color="red", marker="o", label="CubicSpline function")
plt.scatter(x_new, y2_new, color="orange", marker="*", label="Pchip function")
# 添加连接点的直线
plt.plot(x_new, y1_new, color="red", linestyle="-", linewidth=1)
plt.plot(x_new, y2_new, color="orange", linestyle="-", linewidth=1)
plt.xticks(np.arange(2009, 2023, 1), rotation=-45)

plt.ylabel("value")
plt.legend()  # 展示标签信息
plt.show()
# 在这里使用 三次样条法和 Pchip 埃米尔特三次分段法 来计算模型数据可以看到三次样条法算出来的数据更加地平滑
