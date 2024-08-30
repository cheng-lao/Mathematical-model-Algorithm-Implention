import math
from random import random
import matplotlib.pyplot as plt
import numpy as np


class SA:
    def __init__(self, x0, visualize, opt=False,iter=100, T0=100, Tf=0.00001, alpha=0.95):
        self.iter = iter         #内循环迭代次数,即为L =100
        self.alpha = alpha       #降温系数，alpha=0.99
        self.T0 = T0             #初始温度T0为100
        self.Tf = Tf             #温度终值Tf为0.01
        self.T = T0              #当前温度
        self.x = x0              #初始解 一般是随机生成的 也有可能是算出来的
        self.visualize = visualize  #可视化最后的结果
        self.history = {'f': [], 'T': [],'x': []}
        self.opt = opt

    def constraint(self, x):    
        """限制条件 由子类实现"""
        raise NotImplementedError("This method is implemented by subclasses!")
    
    def generate_new(self, x):   #扰动产生新解的过程
        """产生新解的过程"""
        raise NotImplementedError("This method is implemented by subclasses!")
    
    def opt_generate_new(self, x):
        """更好的产生新解的方式"""
        raise NotImplementedError("This method is implemented by subclasses!")

    def tonumber(self, x):
        """将输入转换成数字"""
        raise NotImplementedError("This method is implemented by subclasses!")

    def Metrospolis(self, f, f_new):   #Metropolis准则
        """一般来说 f, f_new往往是一个数字,但是也不一定, 所以这里要将非数字形式的结果转换为数字"""
        if f == np.nan or f_new == np.nan:
            f       = self.tonumber(f)
            f_new   = self.tonumber(f_new)
            
        if f_new > f:
            return 1
        else:
            p = math.exp(- np.abs(f - f_new) / self.T)
            if random() < p:
                return 1
            else:
                return 0

    def func(self, x, *args, **kwargs):
        """由解得到的答案"""
        raise NotImplementedError()

    def best(self, y):    #获取最优目标函数值
        f_best = max(y)
        idx = y.index(f_best)
        return f_best, idx    #f_best,idx分别为在该温度下，迭代L次之后目标函数的最优解和最优解的下标

    def run(self):
        count = 0
        #外循环迭代，当前温度小于终止温度的阈值
        while self.T > self.Tf:       
            #内循环迭代100次
            # print(self.x)
            f_new_list = []
            x_new_list = []
            f_new_list.append(self.func(self.x))
            x_new_list.append(self.x)
            
            for _ in range(self.iter): 
                # print(self.x)
                f = self.func(self.x)                   #f为迭代一次后的值
                if self.opt:
                    x_new = self.opt_generate_new(self.x)               #产生新解
                else:
                    x_new = self.generate_new(self.x)
                f_new = self.func(x_new)                               #产生新值
                if self.Metrospolis(f, f_new):                         #判断是否接受新值
                    self.x = x_new             # 如果接受新值，则把新值的x,y存入x数组和y数组
                    f = f_new
                    f_new_list.append(f)
                    x_new_list.append(self.x)
            
            fbest, x_best_idx = self.best(f_new_list)  # 在这一温度下找到最好的结果
            self.x = x_new_list[x_best_idx]
            self.history['f'].append(fbest)
            self.history['x'].append(self.x)
            self.history['T'].append(self.T)
            # 温度按照一定的比例下降（冷却）
            self.T = self.T * self.alpha        
            count += 1
            
            print("current temperature is ", self.T)
            # 得到最优解
        
        return self.func(self.x), self.x        
