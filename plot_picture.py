# 对仿真结果进行画图显示
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
from OptimizeParam import OptimizeParam


class Plot_Picture:
    def __init__(self):
        self.opt = OptimizeParam()
        self.BTH = self.opt.B_FDMA

    def plot_picture(self, algorithm_name):
        path = "result/fBest_List_" + algorithm_name
        fBest_List = np.loadtxt(path + ".txt", delimiter=',').reshape(len(self.BTH))
        fBest_List_fdma = np.loadtxt("result/fBest_List_fdma_decode_error_OR.txt", delimiter=',').reshape(len(self.BTH))
        B = []
        for i in range(len(self.BTH)):
            B.append(self.BTH[i] / 1000)

        plt.xlabel('Bandwidth kHz')  # 设置 x轴标签
        plt.ylabel(algorithm_name)  # 设置 y轴标签
        plt.title(algorithm_name + " Algorithm")  # 设置标题

        plt.grid(ls='--')  # 设置网格

        # 设置坐标轴范围
        xmin = min(B)
        xmax = max(B)
        ymin = 6.0
        ymax = math.ceil(max(fBest_List))
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, 25])

        x_ticks = np.linspace(xmin, xmax, len(self.BTH)).astype(np.int)  # 产生区间在-5至4间的10个均匀数值
        plt.xticks(x_ticks)  # 将linspace产生的新的十个值传给xticks( )函数，用以改变坐标刻度
        y_ticks = 0
        if algorithm_name == "OR":
            y_ticks = np.linspace(ymin, 25, 20)  # OR
        elif algorithm_name == "OS":
            y_ticks = np.linspace(0, 2, 11)
        elif algorithm_name == "OS":
            # y_ticks = np.linspace(0, 1.0, 11)
            y_ticks = np.linspace(0.56, 1, 22)

        # y_ticks = np.linspace(0.9, 2, 12)  # OS
        # y_ticks = np.linspace(0.56, 1, 22)  # OS
        # y_ticks = np.around(y_ticks, 2)
        plt.yticks(y_ticks)  # 将linspace产生的新的十个值传给xticks( )函数，用以改变坐标刻度

        plt.plot(B, fBest_List, 'r-o')
        plt.plot(B, fBest_List_fdma, 'b-o')
        plt.legend(["Finite BlockLength","原论文"])
        plt.rcParams['font.sans-serif'] = ['SimHei']


        plt.show()


if __name__ == '__main__':
    main = Plot_Picture()
    main.plot_picture("OR")
