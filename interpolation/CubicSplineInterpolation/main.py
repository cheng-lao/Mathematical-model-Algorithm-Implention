import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline


def testunit(x):
    return 1 / (1 + x**2)


if __name__ == "__main__":
    # 原来额可以直接调用库函数直接实现 Pchip函数 当然matlab也可以实现!
    x = np.linspace(-np.pi, np.pi, 10)
    # y = np.sin(x)
    y = testunit(x)
    pchip = CubicSpline(x, y)
    x_new = np.linspace(-5, 5, 100)
    y_new = pchip(x_new)

    plt.plot(x, y, "o", label="data points")
    plt.plot(x_new, y_new, "-", label="PCHIP interpolation")
    plt.legend()
    plt.show()
