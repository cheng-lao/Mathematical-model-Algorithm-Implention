#Created by Copilot
import numpy as np
import pandas as pd
import copy


def linear_regression(x, y):
    n = len(x)
    x = np.array(x)
    y = np.array(y)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x * x)
    sum_xy = np.sum(x * y)
    
    # k = ((n - 1) * sum_xy - sum_x * sum_y) / ((n - 1) * sum_xx - sum_x * sum_x)
    # b = (sum_xx * sum_y - sum_x * sum_xy) / ((n - 1) * sum_xx - sum_x * sum_x)
    khat = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x**2)
    bhat = sum_y / n - khat * sum_x / n
    
    return khat, bhat

def GM11(y):
    """使用GM11模型生成拟合结果和检测参数平滑度，级比偏差

    Args:
        y(list): original data

    Returns (tuple): 
        fitting_data, 拟合值
        residual_abs, 绝对残差 y - y_hat
        mean_residual_r, 平均残差
        mean_eta, 级比残差
        ahat, 发展系数
        bhat 灰作用量
        
    """
    
    ##------------------------超参数--------------------
    delta = 0.5
    
    
    length = len(x)
    y_copy = copy.deepcopy(y)
    ##---------------------制作 X 自变量----------------
    
    x1 = []
    last = 0
    for elm in y:
        x1.append(last + elm)
        last = x1[-1]
    
    z1 = []
    for i in range(1, len(x1)):
        z1.append(delta * x1[i] + (1 - delta) * x1[i - 1])
        
    x = z1
    
    x = [[1 for _ in range(len(x))], x]
    xmat = np.array(x).T
    ##--------------制作y----------------------------
    y = y[1::]
    ymat = np.array(y)
    
    ##-------------计算k,b----------------------------
    
    ### 方法1
    betal = np.linalg.inv(xmat.T @ xmat) @ xmat.T @ ymat
    betal.reshape(2,1)
    # print("betal:", betal)

    ### 方法2 默认采用方法2
    k, b = linear_regression(z1, y)
    # 此时得到ymat = kxmat + b + u_i(残差项) 其中求出的k,b可以使得残差项最小
    ahat = -k
    bhat = b
    #此时得到k 即是发展系数，b标志灰作用量
    print("khat is ",k , "\n bhat is ", bhat)
    
    ##--------------------残差检验---------------------
    fitting_data = [(1 - np.exp(ahat)) * (y[0] - bhat/ahat) * np.exp(-ahat * m) for m in range(1,length)]
    residual_abs = y - np.array(fitting_data)   # 绝对残差
    residual_r = np.abs(y - np.array(fitting_data)) / y * 100    # 相对残差
    mean_residual_r = np.sum( np.abs( residual_r ) ) / (length - 1)     # 平均相对残差
    
    ##------------------------级比残差检验-----------------
    eps = [y_copy[i] / y_copy[i-1] for i in range(1, length)]
    eps = np.array(eps)
    eta = np.abs(1 - ((1 - 0.5*ahat) / (1 + 0.5*ahat)) / eps)
    mean_eta = np.sum(eta) / (length - 1)
    
    return fitting_data, residual_abs, mean_residual_r, mean_eta, ahat, bhat
    
    
def GMmodel(x):
    pass

if __name__ == '__main__':

    # 示例数据
    
    year = range(1995,2005)
    xdata = [174,179,183,189,207,234,220.5,256,270,285] #';  %原始数据序列，写成列向量的形式（加'就表示转置）

    # 使用 GM(1,1) 模型进行预测
    predicted_data = GM11(xdata,year)

    print("原始数据:", xdata)
    print("预测数据:", predicted_data)

    print(len(xdata))
    print(len(predicted_data))
    from matplotlib import pyplot as plt 
    
    plt.plot(year,xdata,label="original data",color='r')
    plt.xlabel("original data")
    # plt.plot(year[1::],predicted_data,label="拟合数据",color='b')
    plt.grid(True)
    plt.show()    
    
    
    