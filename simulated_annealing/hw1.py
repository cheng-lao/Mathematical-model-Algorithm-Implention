from utils.SimAnnaeal import SA
import numpy as np
from random import random

def func(x):#函数优化问题
    y = x[1]
    x = x[0]
    res = 21.5 + x*np.sin(4 * np.pi * x) + y * np.sin(20 * np.pi * y)
    return res

def visualize(res ,x):
    print("Extremum is: ", res," x is:", x)

class CalExtremum(SA):
    
    def __init__(self, func, x0, visualize, iter=100, T0=100, Tf=1e-6, alpha=0.99):
        super().__init__(func, x0, visualize, iter, T0, Tf, alpha)
    
    def constraint(self, x):
        """make self constraint for a specific problem"""
        if x[0] < 12.1 and x[0] > -3 and x[1] > 4.1 and x[1] < 5.8 :
            return True
        else :
            return False
    
    def generate_new(self, x):
        y = x[1]
        x = x[0]
        while True:
            x_new = x + self.T * (random() - random())
            y_new = y + self.T * (random() - random())
            if self.constraint([x_new, y_new]):  
                break                                  #重复得到新解，直到产生的新解满足约束条件
        return [x_new, y_new] 
        # return super().generate_new(x)

    def opt_generate_new(self, x):
        # return super().opt_generate_new(x)
        pass
    
    def tonumber(self, x):
        # return super().tonumber(x)
        pass
    

if __name__ == "__main__":
    sa = CalExtremum(func=func, 
                    visualize=visualize,
                    x0=[1, 2])
    sa.run()

   

