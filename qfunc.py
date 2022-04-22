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
    xNew = [4.14645403e+04, 3.84444444e+04, 3.84444444e+04, 5.24444444e+04,
            5.24444444e+04, 4.22953547e+04, 4.21404233e+04, 3.84444444e+04,
            5.38774595e+04, 8.46088259e-03, 9.70734403e-03, 9.41500789e-03,
            8.61160716e-03, 7.32133237e-03, 8.82243248e-03, 9.04017858e-03,
            9.09670535e-03, 8.18781159e-03, 2.78963742e-03, 1.10550404e-03,
            1.22166284e-03, 2.33056570e-03, 4.95094461e-03, 2.01094671e-03,
            1.43839375e-03, 1.39902465e-03, 3.25745450e-03, 1.58195201e-03,
            1.41318693e-03, 1.41318693e-03, 1.56203199e-03, 1.61418327e-03,
            1.47764945e-03, 1.41318693e-03, 1.41318693e-03, 1.61418327e-03]

    B = xNew[0:9]
    theta = xNew[9:18]
    error = xNew[18:27]
    D_t = xNew[27:]
    R_th = 60000
    D_max = 0.01
    T = 0.005
    snr = [18.25, 32, 29.25, 15.5, 10, 21, 23.75, 26.5, 12.75]

    for i in range(9):
        B_i = B[i]
        theta_i = theta[i]
        D_t_i = D_t[i]
        error_i = error[i]
        SNR = snr[i]
        m = B_i * D_t_i
        ec = qf.EC_function(B_i, SNR, m, error_i, theta_i, T)
        ec1 = qf.EC_function_infinity(B_i, SNR, theta_i, T)
        p = 1 - math.exp(-theta_i * ec * (0.005 - D_t_i))
        print("*********************")
        print(ec)
        print(ec1)
        print(p)
        print("*********************")

    print(error)
