import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator


# # 3points 3degree function
# def Hermite_3p3d(x, y, yd, input):
#     if len(x) != 3 or len(y) != 3 or len(yd) != 3:
#         raise Exception("some term has wrong degree!")
#     result = 0
#     result += y[0]  # first item

#     # 手动计算参数
#     term01 = (y[1] - y[0]) / (x[1] - x[0])
#     term12 = (y[2] - y[1]) / (x[2] - x[1])
#     term012 = (term12 - term01) / (x[2] - x[0])
#     A = (yd[1] - term01 - term012 * (x[1] - x[0])) / ((x[1] - x[0]) * (x[1] - x[2]))

#     result += (input - x[0]) * term01
#     result += term012 * (input - x[0]) * (input - x[1])
#     result += A * (input - x[0]) * (input - x[1]) * (input - x[2])

#     return result


# # 2points 3 degree function
# def Hermiter_2p3d(x, y, yd, input):
#     if len(x) != 2 or len(y) != 2 or len(yd) != 2:
#         raise Exception("some term has wrong degree!")

#     raise NotImplementedError  # 有现成的库 回头再实现吧!


def testunit(x):
    return 1 / (1 + x**2)


if __name__ == "__main__":
    # 原来额可以直接调用库函数直接实现 Pchip函数 当然matlab也可以实现!
    x = np.linspace(-np.pi, np.pi, 10)
    # y = np.sin(x)
    y = testunit(x)
    pchip = PchipInterpolator(x, y)
    x_new = np.linspace(-5, 5, 100)
    y_new = pchip(x_new)

    plt.plot(x, y, "o", label="data points")
    plt.plot(x_new, y_new, "-", label="PCHIP interpolation")
    plt.legend()
    plt.show()
