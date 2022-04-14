import scipy.special as sc
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import random


# 定义qfunction和其逆函数
# 定义 finite blocklength 下有效容量计算公式
class Qfunction:

    # Qfunction逆函数
    def Qfuncinv(self, x):
        return math.sqrt(2) * sc.erfinv(1 - 2 * x)

    # Qfunction函数
    def Qfunc(self, x):
        return 0.5 - 0.5 * sc.erf(x / math.sqrt(2))

    # 有效容量蒙特卡罗计算方式
    def EC_error_mtkl_function(self, B, SNR, m, decodeError, theta, T):
        i = 0
        ec_sum = 0
        while i < 100000:
            hi = random.exponential(1)
            # r = math.log2(1 + SNR * hi) - math.sqrt((1 / m) * (1 - (1 / math.pow(SNR * hi+1, 2)))) * self.Qfuncinv(
            #     decodeError) * math.log2(math.e)
            r = math.log2(1 + SNR * hi) - math.sqrt((1 / m) * (1 - math.pow(1 + SNR * hi, -2))) * self.Qfuncinv(
                decodeError) * math.log2(math.e)

            r = B * r
            ec_sum = ec_sum - (1 / (theta * T)) * math.log(decodeError + (1 - decodeError) * math.exp(-theta * T * r))
            # ec_sum = ec_sum - (1 / (theta * T)) * math.log(math.exp(-theta * T * r))
            i = i + 1

        return ec_sum / 100000

    # 有效容量蒙特卡罗计算方式
    def EC_error_mtkl_function1(self, B, SNR, m, decodeError, theta, T):
        i = 0
        ec_sum = 0
        while i < 100000:
            hi = random.exponential(1)
            r = math.log2(1 + SNR * hi) - math.sqrt((1 / m) * (1 - math.pow(1 + SNR * hi, -2))) * self.Qfuncinv(
                decodeError) * math.log2(math.e)
            r = B * r
            ec_sum = ec_sum + math.exp(-theta * T * r)
            i = i + 1

        ec_sum = ec_sum / 100000
        EC = (-1 / (theta * T)) * math.log(decodeError + (1 - decodeError) * ec_sum)
        return EC

    # 存在误码率时有效容量计算表达式
    def EC_function(self, B, SNR, m, decodeError, theta, T):
        w = - (theta * T * B) / math.log(2)
        v = theta * B * T * math.sqrt(1 / m) * self.Qfuncinv(decodeError) * math.log2(math.e)
        # EC = -(1 / (theta * T)) * math.log(
        #     decodeError + (1 - decodeError) * ((math.exp(v) / SNR) * sc.hyperu(1, 2 + w, 1 / SNR)))
        EC = -(1 / (theta * T)) * math.log((math.exp(v) / SNR) * sc.hyperu(1, 2 + w, 1 / SNR))
        return EC

    # 超几何函数2F0
    def generalized_hypergeometric_2_0(self, x, y, z):
        a = x
        b = a + 1 - y
        c = -math.pow(z, -1)
        d = sc.hyperu(a, b, c)
        return d * math.pow(c, a)

    # 无误码率时有效容量公司给hi
    def EC_noerror(self, B, SNR, theta, T):
        single_func = -(1 / (theta * T)) * math.log(
            self.generalized_hypergeometric_2_0(theta * B * T / math.log(2), 1, -SNR))

        return single_func


if __name__ == '__main__':
    qf = Qfunction()
    D_max = 0.01
    B = 30000
    SNR = 20
    m = 1500
    decodeError = np.linspace(0.0001, 0.01, 100)

    theta = 0.001
    T = 0.005
    pp = []
    for e in decodeError:
        EC = qf.EC_function(B, SNR, m, e, theta, T)
        p = 1 - math.exp(-theta * EC * D_max)
        p = p*(1-e)
        pp.append(p)
    plt.plot(decodeError, pp, 'r-o')
    plt.xlabel("decodeError")
    plt.ylabel(r"probability")
    plt.show()


    # print(qf.Qfuncinv(1))