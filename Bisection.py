import scipy.special as sc
import math
from qfunc import Qfunction


# 给定alhpha和 B，利用二分法求解theta
class Bisection:

    # 参数设置
    def __init__(self):
        self.T = 0.005  # block Time
        self.Rth = 60000  # 到达速率

    # 二分法
    # 参考网址： https://zhuanlan.zhihu.com/p/136823356
    # 对参数theta进行二分法数值分析求解值
    def calculate_theta_by_bisection(self, B, error, snr, m, theta_lo, theta_hi, eps=10 ** -3):
        middle = (theta_lo + theta_hi) / 2
        while abs(self.EC_B_theta(B, error, snr, middle, m)) > eps:
            middle = theta_lo + (theta_hi - theta_lo) / 2
            # print("二分法求解, middle:", middle, "中间结果：", self.EC_B_theta(B, error, snr, middle, m))
            if self.EC_B_theta(B, error, snr, middle, m) * self.EC_B_theta(B, error, snr, theta_lo, m) > 0:
                theta_lo = middle
            else:
                theta_hi = middle
            # 解决精度损失的问题
            if (theta_hi == middle or theta_lo == middle) and abs(
                    self.EC_B_theta(B, error, snr, middle, m)) < eps * 1e3:
                break
        return middle


    # 二分法
    # 参考网址： https://zhuanlan.zhihu.com/p/136823356
    # 对参数theta进行二分法数值分析求解值
    def calculate_theta_bar_by_bisection(self, B, snr, theta_lo, theta_hi, eps=10 ** -5):
        middle = (theta_lo + theta_hi) / 2
        while abs(self.EC_B_theta_bar(B, snr, middle)) > eps:
            middle = theta_lo + (theta_hi - theta_lo) / 2
            # print("二分法求解, middle:", middle, "中间结果：", self.EC_B_theta_bar(B, snr, middle))
            if self.EC_B_theta_bar(B, snr, middle) * self.EC_B_theta_bar(B, snr, theta_lo) > 0:
                theta_lo = middle
            else:
                theta_hi = middle
            # 解决精度损失的问题
            if (theta_hi == middle or theta_lo == middle) and abs(
                    self.EC_B_theta_bar(B, snr, middle)) < eps * 1e3:
                break
        return middle

    # 配对中单独用户的二分法方程  使用FDMA进行传送消息
    def EC_B_theta(self, B, decodeError, SNR, theta, m):
        qfunction = Qfunction()
        single_func = qfunction.EC_function(B, SNR, m, decodeError, theta, self.T) - self.Rth
        return single_func

    # 配对中单独用户的二分法方程  使用FDMA进行传送消息
    def EC_B_theta_bar(self, B, SNR, theta):
        qfunction = Qfunction()
        single_func = qfunction.EC_function_infinity(B, SNR, theta, self.T) - self.Rth
        return single_func


    # 超几何函数2F0
    def generalized_hypergeometric_2_0(self, x, y, z):
        a = x
        b = a + 1 - y
        c = -math.pow(z, -1)
        d = sc.hyperu(a, b, c)
        return d * math.pow(c, a)

    # 无误码率时有效容量公司给hi
    def EC_noerror(self, B, SNR, theta):
        single_func = -(1 / (theta * self.T)) * math.log(
            self.generalized_hypergeometric_2_0(theta * B * self.T / math.log(2), 1, -SNR)) - self.Rth

        return single_func


if __name__ == '__main__':
    bisection = Bisection()
    qfunction = Qfunction()

    B = 25000
    decodeError = 0.01
    SNR = 21
    theta = 1e-7
    res1 = bisection.EC_B_theta_bar(B,SNR,theta) +60000
    print(res1)
    theta_bar = bisection.calculate_theta_bar_by_bisection(B, SNR, 1e-7, 0.2)
    res = bisection.EC_B_theta_bar(B,SNR,theta_bar)+60000
    print(theta_bar)
    print(res)
