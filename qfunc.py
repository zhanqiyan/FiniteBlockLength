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
        EC = -(1 / (theta * T)) * math.log((math.exp(v) / SNR) * sc.hyperu(1, 2 + w, 1 / SNR))
        return EC

    def EC_function_infinity(self, B, SNR, theta, T):
        w = - (theta * T * B) / math.log(2)
        EC = -(1 / (theta * T)) * math.log((1 / SNR) * sc.hyperu(1, 2 + w, 1 / SNR))
        return EC

    # 存在误码率时有效容量计算表达式
    def EC_function_err(self, B, SNR, D_t, decodeError, theta, T):
        EC_err = -math.sqrt(B / D_t) * self.Qfuncinv(decodeError) * math.log2(math.e)
        return EC_err

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

    def f_function(self, theta, D_t, B, SNR):
        R_th = 60000
        D_max = 0.01
        T = 0.005
        p = 1 - math.exp(-theta * R_th * (D_max - D_t))
        ec_infinity = self.EC_function_infinity(B, SNR, theta, T)
        qx = math.sqrt(D_t / B) * (ec_infinity - R_th) * math.log(2)
        return (p) * (1 - self.Qfunc(qx))


if __name__ == '__main__':
    qf = Qfunction()
    # D_max = 0.01
    # B = 30000
    # SNR = 20
    # res = []
    # theta = np.arange(0.0001,0.01,0.001)
    # for t in  theta:
    #     res.append(qf.f_function(t,0.001,B,SNR))
    #
    # plt.plot(theta,res)
    # plt.show()

    theta = 0.003
    B = 30000
    SNR = 20
    res = []
    D = np.arange(0.0001, 0.005, 0.0001)
    for d in D:
        res.append(qf.f_function(theta, d, B, SNR))

    plt.plot(D, res)
    plt.show()

    xNew = [2.15402484e+04, 1.92445916e+04, 1.63434417e+04, 2.09000942e+04,
            2.65257554e+04, 1.73001803e+04, 1.62000000e+04, 1.62637516e+04,
            2.56819368e+04, 4.44612076e-03, 5.42225619e-03, 3.17956308e-03,
            3.06256057e-03, 3.42011082e-03, 2.49222653e-03, 2.37089577e-03,
            2.64403096e-03, 4.45038133e-03, 6.29612444e-02, 2.01752052e-02,
            1.61858296e-01, 1.39425743e-01, 8.69822882e-02, 3.69455959e-01,
            4.76532420e-01, 2.73188203e-01, 5.15454896e-02, 2.32908364e-03,
            1.61418327e-03, 2.32908364e-03, 2.32908364e-03, 2.32908364e-03,
            1.61418327e-03, 4.57450183e-04, 2.05601600e-03, 2.32908364e-03]

    B = xNew[0:9]
    theta = xNew[9:18]
    D_t = xNew[27:]
    R_th = 60000
    D_max = 0.01
    T = 0.005
    snr = [18.25, 32, 29.25, 15.5, 10, 21, 23.75, 26.5, 12.75]
    error = []
    for i in range(9):
        B_i = B[i]
        theta_i = theta[i]
        D_t_i = D_t[i]
        SNR = snr[i]
        ec_infinity = qf.EC_function_infinity(B_i, SNR, theta_i, T)
        qx = math.sqrt(D_t_i / B_i) * (ec_infinity - R_th) * math.log(2)
        epsilon_i = qf.Qfunc(qx)
        error.append(epsilon_i)
    print(error)
