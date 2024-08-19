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
    
    
    length = len(y)
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


    ### 方法2 默认采用方法2
    k, b = linear_regression(z1, y)
    # 此时得到ymat = kxmat + b + u_i(残差项) 其中求出的k,b可以使得残差项最小
    ahat = -k
    bhat = b
    #此时得到k 即是发展系数，b标志灰作用量
    print("发展系数khat is ",k , " 灰作用量bhat is ", bhat)
    
    ##--------------------残差检验---------------------
    f = lambda k : (1 - np.exp(ahat)) * (y[0] - bhat/ahat) * np.exp(-ahat * k)
    fitting_data = [f(m) for m in range(1,length)]
    print("fitting_data'length  is ", len(fitting_data))
    print(length -1)
    
    
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


def testData(x, threshold):
    """Quasi-exponential law test 准指数规律检验
        x: 数据序列值
    """
    length = len(x)
    delta = 0.5
    ##---------------------制作 X 自变量----------------
    
    x1 = []  # x1是累加值 length is n
    last = 0
    for elm in x:
        x1.append(last + elm)
        last = x1[-1]
    
    z1 = [] # z1是中间取值 length is n - 1
    for i in range(1, len(x1)):
        z1.append(delta * x1[i] + (1 - delta) * x1[i - 1])
    
    z1_tmp = [x1[0], *z1]
    
    eta = [z1_tmp[i] / z1_tmp[i-1] for i in range(1,length)]
    eta = np.array(eta) # 级比 length is n-1
    
    ##------------计算 级比中满足参数要求的个数--------
    meetCnt = 0
    for i in range(len(eta)):
        if eta[i] >= 1 - 1e-6 and eta[i] <= 1.5 + 1e-6:
            meetCnt += 1
    
    if meetCnt / len(eta)  > threshold :
        return True, meetCnt / len(eta)
    else:
        return False, meetCnt / len(eta)

if __name__ == '__main__':

    # 示例数据
    year = range(1995,2005)
    xdata = [174,179,183,189,207,234,220.5,256,270,285] #';  %原始数据序列，写成列向量的形式（加'就表示转置）

    ##---------------根据输入的数据判断是否使用GM模型合适-------------
    if len(xdata) > 12:
        raise Exception("推荐使用其他模型例如 ARIMA， 时间序列分析，温斯特模型，回归模型等")

    testres,score = testData(xdata,0.7)
    if not testres:
        raise Exception(f"不能通过指数平滑检验,得分是{ score }")
    else:
        print(f"通过指数平滑检验，得分是 {score}")
    


    # 使用 GM(1,1) 模型进行预测
    predicted_data, residual_abs, mean_residual_r, mean_eta, ahat, bhat= GM11(xdata)

    print("原始数据:", xdata)
    print("预测数据:", predicted_data)

    print(len(xdata))
    print(len(predicted_data))
    from matplotlib import pyplot as plt 
    
    plt.plot(year,xdata,label="original data",color='r')
    plt.xlabel("original data")
    plt.plot(year[1::],predicted_data,label="fitting data GM(1,1)",color='b')
    plt.scatter(year[1::],predicted_data,)
    # for i, txt in enumerate(xdata):
    #     plt.annotate(f'{txt}', (year[i], xdata[i]), textcoords="offset points", xytext=(0,5), ha='center')
    
    for i, txt in enumerate(predicted_data):
        plt.annotate(f'{txt:.2f}', (year[i+1], predicted_data[i]), textcoords="offset points", xytext=(0,5), ha='center')

    
    plt.grid(True)
    plt.legend()
    plt.show()    
    
    
    