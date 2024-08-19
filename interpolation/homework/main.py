import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numbers
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import CubicSpline

file_path = r"interpolation\homework\AllData.xls"
sheet_name = "附件2 其它数据"
data = pandas.read_excel(file_path, sheet_name=sheet_name)
print(data.columns)


# for column in data.columns:
#     print(f"col name {column}")
#     print(data[column])  遍历每一列
def wait_for_enter():
    input("按回车键继续...")


rownum = 4
colnum = 4
num = rownum * colnum
# fig, axs = plt.subplots(colnum, rownum, figsize=(100, 80))  # 2行1列的子图
figidx = 0
for index, row in data.iterrows():
    print(f"行索引: {index}")
    flag = True
    x = []
    y = []
    for col in data.columns:
        elem = row[col]
        if pd.isna(elem):
            # print(f"{col}: NaN", end=" ")
            if col in [1, 3, 5, 7, 9, 11, 13, 15]:
                flag = False
                break
        else:
            print(f"{col}: {elem}")
            if isinstance(col, numbers.Number) and isinstance(elem, numbers.Number):
                x.append(col)
                y.append(elem)
    if not flag:
        continue
    # print(x)
    # print(y)

    if figidx == 0:
        fig, axs = plt.subplots(colnum, rownum, figsize=(100, 80))

    x = np.array(x)
    y = np.array(y)
    pchip = PchipInterpolator(x, y)
    cubis = CubicSpline(x, y)
    x_new = [2, 4, 6, 8, 10, 12, 14]
    y_new_pchip = pchip(x_new)
    y_new_cubis = cubis(x_new)

    # 开始作图
    rowid = figidx // rownum
    colidx = figidx % colnum
    print(f"figidx is {figidx}")
    axs[rowid][colidx].plot(x, y, "o", label="data points")
    axs[rowid][colidx].plot(
        x_new, y_new_pchip, "-", color="orange", label="PCHIP interpolation"
    )
    axs[rowid][colidx].plot(
        x_new, y_new_cubis, "--", color="red", label="CubicSpline interpolation"
    )
    axs[rowid][colidx].legend()
    print(f"y_new_cubis :{y_new_cubis}")
    print(f"y_new_pchip :{y_new_pchip}")
    # wait_for_enter()
    if figidx + 1 < num:
        figidx += 1
        print("")
        continue
    figidx = 0
    plt.tight_layout()
    plt.show()
    plt.clf()
