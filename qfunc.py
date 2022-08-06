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
            r = math.log2(1 + SNR * hi) - math.sqrt((1 / m) * (1 - math.pow(1 + SNR * hi, -2))) * self.Qfuncinv(
                decodeError) * math.log2(math.e)

            r = B * r
            # ec_sum = ec_sum - (1 / (theta * T)) * math.log(decodeError + (1 - decodeError) * math.exp(-theta * T * r))
            ec_sum = ec_sum - (1 / (theta * T)) * math.log(math.exp(-theta * T * r))
            i = i + 1
        return ec_sum / 100000

    # 有效容量蒙特卡罗计算方式
    def EC_error_mtkl_function1(self, B, SNR, m, decodeError, theta, T):
        i = 0
        r_sum = 0
        while i < 100000:
            hi = random.exponential(1)
            r = math.log2(1 + SNR * hi) - math.sqrt((1 / m) * (1 - math.pow(2 + SNR * hi, -2))) * self.Qfuncinv(
                decodeError) * math.log2(math.e)

            # r = math.log2(1 + SNR * hi) - math.sqrt((1 / m) ) * self.Qfuncinv(
            #     decodeError) * math.log2(math.e)
            r = B * r
            r_sum = r_sum + math.exp(-theta * T * r)
            i = i + 1

        r_sum = r_sum / 100000
        EC = (-1 / (theta * T)) * math.log(r_sum)
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
    # xNew = [2.60149957e+04, 2.00000000e+04, 2.00000000e+04, 3.25373792e+04,
    #         3.03899603e+04, 2.65207026e+04, 2.73387341e+04, 2.40627429e+04,
    #         2.71354852e+04, 6.21211349e-03, 6.37556058e-03, 5.91849975e-03,
    #         6.91524578e-03, 4.69666338e-03, 6.93199248e-03, 7.57013628e-03,
    #         7.12534662e-03, 4.91839256e-03, 1.46326578e-02, 1.43105610e-02,
    #         2.02950418e-02, 6.93567024e-03, 3.76742226e-02, 9.00877148e-03,
    #         5.07384154e-03, 8.09858734e-03, 3.46942748e-02, 2.05601600e-03,
    #         2.10816728e-03, 2.10816728e-03, 1.83509963e-03, 2.10816728e-03,
    #         1.91948218e-03, 1.83509963e-03, 1.83509963e-03, 2.19254982e-03]

    # xNew = [3.91722726e+04, 3.15555556e+04, 3.15555556e+04, 4.44817348e+04,
    #         4.40466927e+04, 3.62122796e+04, 3.68371889e+04, 3.15555556e+04,
    #         4.25831647e+04, 8.40432060e-03, 9.12359996e-03, 8.84728259e-03,
    #         8.33259908e-03, 6.83871273e-03, 8.56160796e-03, 8.86876292e-03,
    #         8.68180420e-03, 7.46984364e-03, 2.94001902e-03, 1.72995768e-03,
    #         2.23280492e-03, 3.06875821e-03, 7.25772040e-03, 2.66307716e-03,
    #         2.19651415e-03, 2.45045242e-03, 5.49885340e-03, 1.75071709e-03,
    #         1.47764945e-03, 1.47764945e-03, 1.75071709e-03, 1.83509963e-03,
    #         1.75071709e-03, 1.56203199e-03, 1.66633455e-03, 1.66633455e-03]

    xNew = [7.68716440e+04, 7.58632360e+04, 7.45555556e+04, 7.69944627e+04,
            8.63205544e+04, 8.85555556e+04, 7.84588853e+04, 7.81408631e+04,
            8.92392433e+04, 1.00677815e-02, 1.09098738e-02, 1.07741183e-02,
            9.65325553e-03, 8.62880834e-03, 1.05520885e-02, 1.06000541e-02,
            1.08578408e-02, 9.23501547e-03, 1.57204357e-03, 6.88724845e-04,
            9.17557361e-04, 1.90030756e-03, 2.65680414e-03, 1.25209856e-03,
            8.44785102e-04, 7.45627163e-04, 2.13658262e-03, 1.75071709e-03,
            1.47764945e-03, 1.47764945e-03, 1.75071709e-03, 1.75071709e-03,
            1.75071709e-03, 1.75071709e-03, 1.75071709e-03, 1.66633455e-03]
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
        # print(B_i)
        theta_i = theta[i]
        # print(theta_i)
        D_t_i = D_t[i]
        # print(D_t_i)
        error_i = error[i]
        print(error_i)
        SNR = snr[i]
        m = B_i * D_t_i
        ec = qf.EC_function(B_i, SNR, m, error_i, theta_i, T)
        ec1 = qf.EC_function_infinity(B_i, SNR, theta_i, T)
        p = 1 - math.exp(-theta_i * ec * (D_max))
        # print("*********************")
        # print(ec)
        # print(ec1)
        # print(p)
        # print("*********************")

    # B = 30000
    # SNR = 20
    # theta = 0.008
    # D_t = 0.0025
    # m = B*D_t
    # error = 0.001
    # T = 0.005
    # ec1 = qf.EC_error_mtkl_function(B,SNR,m,0.001,theta,T)
    # ec2 = qf.EC_error_mtkl_function1(B,SNR,m,0.001,theta,T)
    # ec3 = qf.EC_function(B,SNR,m,0.001,theta,T)
    # ec4 = qf.EC_noerror(B,SNR,theta,T)
    # ec5 = qf.EC_function_infinity(B,SNR,theta,T)
    #
    # print(ec1)
    # print(ec2)
    # print(ec3)
    # print(ec4)
    # print(ec5)
