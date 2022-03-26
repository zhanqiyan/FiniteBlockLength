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

    def calculate_B_by_bisection(self, theta, error, snr, m, Bmin, Bmax, eps=0.001):
        middle = (Bmin + Bmax) / 2
        while abs(self.EC_B_theta(middle, error, snr, theta, m)) > eps:
            middle = Bmin + (Bmax - Bmin) / 2
            # print("二分法求解, middle:", middle, "中间结果：", self.func_B(middle, theta, snr))
            if self.EC_B_theta(middle, error, snr, theta, m) * self.EC_B_theta(Bmin, error, snr, theta, m) > 0:
                Bmin = middle
            else:
                Bmax = middle
            # 解决精度损失的问题
            if (Bmin == middle or Bmax == middle) and abs(self.EC_B_theta(middle, error, snr, theta, m)) < eps * 10:
                break
        return middle

    # 配对中单独用户的二分法方程  使用FDMA进行传送消息
    def EC_B_theta(self, B, decodeError, SNR, theta, m):
        qfunction = Qfunction()
        single_func = qfunction.EC_function(B, SNR, m, decodeError, theta, self.T) - self.Rth
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

    B = 16200
    decodeError = 0.01
    SNR = 21
    theta = 1e-7
    m = 10000

    res = bisection.calculate_theta_by_bisection(B, decodeError, SNR, m, 1e-8, 0.2)
    res1 = bisection.EC_B_theta(B,decodeError,SNR,res,m)+60000
    print(res)
    print(res1)
    Rth = 60000
    Dmax = 0.005
    p = 1 - math.pow(2, -(Rth * Dmax * res) / math.log(2))
    p1 = 1 - math.pow(2, -(Rth * Dmax * 0.0011124651664733887) / math.log(2))

    print(p)
    print(p1)