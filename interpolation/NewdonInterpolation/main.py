import numpy as np
from matplotlib import pyplot as plt


# def newton_interpolation(points_x, points_y, input_x):
#     if len(points_x) != len(points_y):
#         raise ValueError("The length of points_x and points_y must be the same!")

#     n = len(points_x)
#     # 初始化差商表
#     divided_diff = [[0 for _ in range(n)] for _ in range(n)]

#     # 填充差商表的第一列
#     for i in range(n):
#         divided_diff[i][0] = points_y[i]

#     # 计算差商表
#     for j in range(1, n):
#         for i in range(n - j):
#             divided_diff[i][j] = (
#                 divided_diff[i + 1][j - 1] - divided_diff[i][j - 1]
#             ) / (points_x[i + j] - points_x[i])

#     # 构建插值多项式
#     result = divided_diff[0][0]
#     product_term = 1.0
#     for i in range(1, n):
#         product_term *= input_x - points_x[i - 1]
#         result += divided_diff[0][i] * product_term

#     return result


def compute_divided_diff(points_x, points_y):
    n = len(points_x)
    divided_diff = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        divided_diff[i][0] = points_y[i]
    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i][j] = (
                divided_diff[i + 1][j - 1] - divided_diff[i][j - 1]
            ) / (points_x[i + j] - points_x[i])
    return divided_diff


def newton_interpolation_fast(points_x, divided_diff, input_x):
    n = len(points_x)
    result = divided_diff[0][0]
    product_term = 1.0
    for i in range(1, n):
        product_term *= input_x - points_x[i - 1]
        result += divided_diff[0][i] * product_term
    return result


def testfunc(x):
    return 1 / (1 + x**2)


if __name__ == "__main__":
    # 初始模拟函数节点
    x = np.linspace(-20, 20, 400)
    y = 1 / (1 + x**2)

    # 插值初始已知数据点
    x1 = np.linspace(-5, 5, 15)
    y1 = 1 / (1 + x1**2)
    divided_diff = compute_divided_diff(x1, y1)  # 提前计算好差商表 O(n^2)算法

    # 使用拉格朗日算法计算插值点
    np.random.seed(114514)
    # 设置随机数种子 方便复现结果! 该种子情况下 可以看出拉格朗日算法会出现龙格效应
    x_interp = np.random.uniform(-5, 5, 35)  # 随机生成一些节点
    y_interp = np.array(
        [newton_interpolation_fast(x1, divided_diff, i) for i in x_interp]
    )  # O(n)算法生成插值

    x_combine = np.concatenate((x1, x_interp))
    y_combine = np.concatenate((y1, y_interp))

    sorted_idcs = np.argsort(x_combine)

    x_lag = x_combine[sorted_idcs]

    y_lag = y_combine[sorted_idcs]

    plt.plot(x, y, label="Original function")
    plt.plot(x1, y1, label="Lagrange interpolation", linestyle="--")
    plt.scatter(x_interp, y_interp, color="red", label="Interpolation nodes")
    plt.plot(x_lag, y_lag, color="blue", label="Fitting Result")

    plt.xlabel("x")
    plt.ylabel("1 / (1 + x^2)")
    plt.title("Runge's Phenomenon with Lagrange Interpolation")
    plt.legend()
    plt.grid(True)
    plt.show()
