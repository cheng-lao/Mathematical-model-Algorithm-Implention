import matplotlib.pyplot as plt
import matplotlib.animation as animation

def visualize(res, x, CitiesData):
    print("Extremum is: ", res, " x is:", x)
    
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

# 示例数据
res = 0  # 示例结果
x = [0, 1, 2, 3, 4, 0]  # 示例路径
CitiesData = [(0, 0), (1, 2), (2, 3), (3, 1), (4, 0)]  # 示例城市数据

visualize(res, x, CitiesData)