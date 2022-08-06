import scipy.special as sc
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from qfunc import Qfunction


class MonteCarlo:

    def plot_strong(self, B, SNR, m, error, theta, T):
        qf = Qfunction()
        EC_mtkl = []
        EC = []
        for t in theta:
            ec2 = qf.EC_error_mtkl_function1(B, SNR, m, error, t, T)

            EC_mtkl.append(ec2)
            ec3 = qf.EC_function(B, SNR, m, error, t, T)
            EC.append(ec3)

        # x_ticks = np.linspace(0, 0.14, 8)
        # plt.xticks(x_ticks, fontsize=18)  # 将linspace产生的新的十个值传给xticks( )函数，用以改变坐标刻度
        #
        # y_ticks = np.linspace(0, 80000, 9)
        # plt.yticks(y_ticks, fontsize=15)  # 将linspace产生的新的十个值传给xticks( )函数，用以改变坐标刻度

        plt.plot(theta, EC_mtkl, 'r:x')
        plt.plot(theta, EC, 'b-.^')
        plt.xlim([0, 0.0115])

        plt.ylim([40000, 100000])
        x_ticks = np.linspace(0, 0.0115, 6)  # 产生区间在-5至4间的10个均匀数值
        plt.xticks(x_ticks, fontsize=20)  # 将linspace产生的新的十个值传给xticks( )函数，用以改变坐标刻度
        y_ticks = np.linspace(40000, 100000, 7)  # 产生区间在-5至4间的10个均匀数值
        plt.yticks(y_ticks, fontsize=20)  # 将linspace产生的新的十个值传给xticks( )函数，用以改变坐标刻度
        plt.grid(ls='--')  # 设置网格
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.xlabel(r'QoS指数$\theta_k$', fontsize=18)  # 设置 x轴标签
        plt.legend([" 蒙特卡洛计算值", "近似表达式计算值"], fontsize=18)
        plt.ylabel(r"有效容量(bps)", fontsize=18)
        # plt.legend(["推导的闭式表达式 ", " 蒙特卡洛"], fontsize=18)
        plt.show()

    # 配对中弱用户的二分法方程


if __name__ == '__main__':
    mt = MonteCarlo()
    B = 30000
    SNR = 20
    theta =  np.arange(0.00001, 0.012, 0.0005)
    D_t = 0.0025
    m = B * D_t
    error = 0.003
    T = 0.005
    mt.plot_strong(B, SNR, m, error, theta, T)
