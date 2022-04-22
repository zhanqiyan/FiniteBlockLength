# 对仿真结果进行画图显示
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
from OptimizeParam import OptimizeParam
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


class Plot_Picture:
    def __init__(self):
        self.opt = OptimizeParam()
        self.BTH = self.opt.B_FDMA

    def plot_picture(self, algorithm_name):

        B_finite = [x / 1000 for x in self.opt.B_FDMA]
        B_no_decode_error = [x / 1000 for x in self.opt.B_FDMA]
        B_equal = [x / 1000 for x in self.opt.B_FDMA]

        path_finite = "result/fBest_List_" + algorithm_name + "_OMA"
        path_no_decode_error = "result/fBest_List_fdma_decode_error_" + algorithm_name
        path_equal = "result/fBest_List_" + algorithm_name + "_equal_bandwidth_error"

        fBest_List_finite = np.loadtxt(path_finite + ".txt", delimiter=',').reshape(len(B_finite))
        fBest_no_decode_error = np.loadtxt(path_no_decode_error + ".txt", delimiter=',').reshape(len(B_no_decode_error))
        fBest_equal = np.loadtxt(path_equal + ".txt", delimiter=',').reshape(len(B_equal))

        plt.xlabel('Bandwidth kHz', fontsize=25)  # 设置 x轴标签
        plt.ylabel(algorithm_name, fontsize=25)  # 设置 y轴标签
        # plt.title(algorithm_name + " Algorithm")  # 设置标题
        plt.grid(ls='--')  # 设置网格

        # 设置坐标轴范围
        xmin = 170
        xmax = 270

        if algorithm_name == "OR":
            ymin = 21
            ymax = 25
        elif algorithm_name == "OS":
            ymin = 1.4
            ymax = math.ceil(2.0)
        else:
            ymin = 0.95
            ymax = math.ceil(1.0)

        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        x_ticks = np.linspace(xmin, xmax, len(B_finite)).astype(np.int)  # 产生区间在-5至4间的10个均匀数值
        plt.xticks(x_ticks, fontsize=18)  # 将linspace产生的新的十个值传给xticks( )函数，用以改变坐标刻度
        y_ticks = 0
        if algorithm_name == "OR":
            y_ticks = np.linspace(ymin, 25, 20)  # OR
            y_ticks = np.arange(ymin, ymax, 0.25)  # OR
            ymajorLocator = MultipleLocator(0.5)  # 将x主刻度标签设置为20的倍数
            yminorLocator = MultipleLocator(0.25)  # 将x轴次刻度标签设置为5的倍数
        elif algorithm_name == "OS":
            y_ticks = np.arange(ymin, ymax, 0.001)
            ymajorLocator = MultipleLocator(0.04)  # 将x主刻度标签设置为20的倍数
            yminorLocator = MultipleLocator(0.02)  # 将x轴次刻度标签设置为5的倍数
        elif algorithm_name == "OP":
            # y_ticks = np.linspace(0, 1.0, 11)
            y_ticks = np.arange(ymin, 0.001, 0.001)
            ymajorLocator = MultipleLocator(0.005)  # 将x主刻度标签设置为20的倍数
            yminorLocator = MultipleLocator(0.001)  # 将x轴次刻度标签设置为5的倍数

        plt.yticks(y_ticks, fontsize=18)  # 将linspace产生的新的十个值传给xticks( )函数，用以改变坐标刻度

        # plt.plot(B_finite, f, 'r-o')
        # plt.plot(B, fBest_List_fdma, 'b-o')
        # plt.legend(["Finite BlockLength", "原论文"])
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        ax = plt.subplot(111)
        xmajorLocator = MultipleLocator(10)  # 将x主刻度标签设置为20的倍数
        xminorLocator = MultipleLocator(2)  # 将x轴次刻度标签设置为5的倍数
        ax.xaxis.set_major_locator(xmajorLocator)
        ax.xaxis.set_minor_locator(xminorLocator)

        ax.yaxis.set_major_locator(ymajorLocator)
        ax.yaxis.set_minor_locator(yminorLocator)

        ax.xaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度
        ax.yaxis.grid(True, which='major')  # y坐标轴的网格使用次刻度

        plt.plot(B_finite, fBest_List_finite, 'c-o')
        plt.plot(B_no_decode_error, fBest_no_decode_error, 'r:x')
        # plt.plot(B_equal, fBest_equal, 'b-.^')
        plt.plot(B_equal[11:], fBest_equal[11:], 'b-.^')
        # plt.axhline(y = 1.98)
        plt.legend(["Finite Block Length", "Infinite Block Length", "OMA with equal bandwidth and error rate"],
                   fontsize=18)
        plt.show()


if __name__ == '__main__':
    main = Plot_Picture()
    main.plot_picture("OR")
