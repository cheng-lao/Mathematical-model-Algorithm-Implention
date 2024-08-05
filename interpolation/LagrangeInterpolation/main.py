# Lagrange Interpolation
import numpy as np
import matplotlib.pyplot as plt


def lagrange_interpolation(x, y, x0):
    """实现拉格朗日插值算法内容 `interpolation\image\LagrangeInterpolation\原理.png`

    Args:
        x (float): x in points
        y (float): y in points
        x0 (float): input

    Returns:
        float : result
    """
    n = len(x)
    y0 = 0
    for i in range(n):
        p = 1
        for j in range(n):
            if j != i:
                p *= (x0 - x[j]) / (x[i] - x[j])
        y0 += y[i] * p
    return y0


def testfunc(x):
    return 1 / (1 + x**2)


if __name__ == "__main__":
    # 初始模拟函数节点
    x = np.linspace(-20, 20, 400)
    y = 1 / (1 + x**2)

    # 插值初始已知数据点
    x1 = np.linspace(-5, 5, 15)
    y1 = 1 / (1 + x1**2)

    # 使用拉格朗日算法计算插值点
    np.random.seed(114514)
    # 设置随机数种子 方便复现结果! 该种子情况下 可以看出拉格朗日算法会出现龙格效应
    x_interp = np.random.uniform(-5, 5, 35)
    y_interp = np.array([lagrange_interpolation(x1, y1, i) for i in x_interp])

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
