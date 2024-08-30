"""
author:  Yingjie Cheng
date: 2024.8.30
function: solve the TSP problem by using simulated annealing algorithm
description:
    1. generate 40 cities randomly
    2. use simulated annealing algorithm to solve the TSP problem
    3. visualize the result
specification:
    1. the algorithm is implemented by using the simulated annealing algorithm,and 
     the method of generating new solution is the exchange of two cities, the exchange of three cities, and the reverse of the city sequence.
     the method of optimizing the solution is to exchange the city with the nearest city.
flaw:
    1. the algorithm is not perfect, the result is not the best
    2. the algorithm is not efficient, the time complexity is O(n^2)
    3. the algorithm is not stable, the result is not the same every time
"""
from utils.SimAnnaeal import SA
import numpy as np
from random import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

def generate_cities(n=40):
    return [(randint(0, 100), randint(0, 100)) for _ in range(n)]   # 40 个城市坐标

def func(x, CitesData):
    cost = 0
    last = -1
    for city in x:
        if last == -1:
            last = city
            continue
        cost += np.sqrt((CitesData[last][0] - CitesData[city][0])**2 
                        + (CitesData[last][1] - CitesData[city][1])**2)
        last = city
    
    cost += np.sqrt((CitesData[last][0] - CitesData[x[0]][0])**2 
                        + (CitesData[last][1] - CitesData[x[0]][1])**2)
    
    return -cost

def visualize(res ,x, CitiesData):
    print("Extremum is: ", res," x is:", x, "length is", len(list(set(x))))
    
    for point in CitesData:    
        plt.scatter(point[0],point[1])
    
    fig, ax = plt.subplots()
    
    # 绘制城市点
    for point in CitiesData:
        ax.scatter(point[0], point[1])
    
    line, = ax.plot([], [], 'r-')  # 初始化线条
    xlist = []
    ylist = []
    def init():
        line.set_data([], [])
        return line,
    
    
    def update(frame):
        if frame == 0:
            return line,
        
        xlist.append([CitiesData[x[frame-1]][0], CitiesData[x[frame]][0]])
        ylist.append([CitiesData[x[frame-1]][1], CitiesData[x[frame]][1]])
        line.set_data(xlist, ylist)
        return line,
    
    ani = animation.FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True, repeat=False)
    
    plt.show()

class TSPSA(SA):
    
    def __init__(self, x0, visualize, CitesData, opt=False,iter=100, T0=100, Tf=1e-6, alpha=0.95):
        super().__init__(x0, visualize, opt, iter, T0, Tf, alpha)
        self.cites = CitesData
        
        self.min_point = []
        for city_i in range(len(CitesData)):
            mindistance = np.inf
            idx = 0 
            for city_j in range(len(CitesData)):
                if city_i == city_j:
                    continue
                
                distance = np.sqrt((self.cites[city_i][0] - self.cites[city_j][0])**2 
                                   + (self.cites[city_i][1] - self.cites[city_i][1])**2 )
                if distance < mindistance:
                    idx = city_j
                    mindistance = distance
            
            self.min_point.append(idx)
            
        
    def constraint(self, x):
        """make self constraint for a specific problem"""
        return 1    #TODO
    
    def generate_new(self, x):
        """x is the series that road to cities
        there ars three methods to implented
        """
        length = len(x)

        p = 0.33    # 选择的方法的概率
        r = random()
        if r < p:
            rand1 = randint(0, length - 1)
            rand2 = randint(0, length - 1)
            x[rand1], x[rand2] = x[rand2], x[rand1]
            
        elif r < 2*p and r >= p:
            rand1 = randint(0, length - 1)
            rand2 = randint(0, length - 1)
            rand3 = randint(0, length - 1)
            if rand1 >= rand2:
                rand1, rand2 = rand2, rand1
            if rand1 >= rand3:
                rand1, rand3 = rand3, rand1
            if rand2 >= rand3:
                rand2, rand3 = rand3, rand2
            x = x[:rand1] + x[rand2:rand3] + [x[rand3]] + x[rand1:rand2] + x[rand3 + 1:]
            
        else:
            rand1 = randint(0, length - 1)
            rand2 = randint(0, length - 1)
            if rand1 >= rand2:
                rand1, rand2 = rand2, rand1
            x = x[:rand1] + x[rand1:rand2][::-1] + x[rand2:]
        
        return x
    
    def opt_generate_new(self, x):
        length =  len(x)
        # print("length is ",length)
        num = x[randint(0, length - 1)]
        min_point_idx = self.min_point[num]
        if num > min_point_idx:
            min_point_idx, num = num, min_point_idx
        
        x = x[:num] + [x[min_point_idx]] +  x[num:min_point_idx] + x[min_point_idx + 1:]
        return x

    def func(self, x, *args, **kwargs):
        cost = 0
        last = -1
        for city in x:
            if last == -1:
                last = city
                continue
            cost += np.sqrt((self.cites[last][0] - self.cites[city][0])**2 
                            + (self.cites[last][1] - self.cites[city][1])**2)
            last = city
        
        cost += np.sqrt((self.cites[last][0] - self.cites[x[0]][0])**2 
                            + (self.cites[last][1] - self.cites[x[0]][1])**2)
        
        return -cost

    
    def tonumber(self, x):
        pass
    
    
    
def generate_permutations(n):
    x = [i for i in range(n)]
    shuffle(x)
    return x


if __name__ == "__main__":
    ars = argparse.ArgumentParser()
    ars.add_argument("--load", type = bool, default = True, required = False)
    
    args = ars.parse_args()
    
    import os
    if args.load and os.path.exists("CitesData.npy"):
        CitesData = np.load("CitesData.npy")
        print("load data from .npy")
    else:
        CitesData = generate_cities()
        np.save("CitesData.npy",CitesData)
        
    n = 40
    x0 = generate_permutations(n)
    print("begin compute!", x0)
    
    sa = TSPSA(visualize = visualize, x0 = x0, CitesData=CitesData, opt=True,alpha=0.99, T0=1000,Tf=0.0000001)
    
    res, solution= sa.run()
    visualize(res, solution, CitesData)



