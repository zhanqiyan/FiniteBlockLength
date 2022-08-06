# 构建目标函数类
import math
import numpy as np
from Bisection import Bisection
from qfunc import Qfunction


class TargetFunction:
    def __init__(self):
        self.N = 14  # bus总线数量
        self.Dmax = 0.01  # 最大延迟时间
        self.T = 0.005  # block Time
        self.Rth = 60000  # 到达速率
        self.K = 9  # PMU数量
        self.snr = [18.25, 32, 29.25, 15.5, 10, 21, 23.75, 26.5, 12.75]
        self.bus_pmu = {2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 11: 8, 13: 9}  # bus与pmu安装位置和索引对应map
        self.install_X = np.array([0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0]).reshape((14, 1))  # PMU安装向量

        # 优化参数X
        # 参数的个数设置
        # 参数为1X27维向量，每一维代表含义：
        # B1,B2,B3,B4,B5,B6,B7,B8,B9
        # theta1,theta2,theta3,theta4,theta5,theta6,theta7,theta8,theta9
        # e1,e2,e3,e4,e5,e6,e7,e8,e9
        # Dt1,Dt2,Dt3,Dt4,Dt5,Dt6,Dt7,Dt8,Dt9
        # B、theta、error、DT的索引对应着PMU索引
        self.B_num = self.K  # 参数带宽的个数     等于PMU数量
        self.theta_num = self.K  # theta参数个数，等于PMU数量
        self.error_num = self.K  # 误码率error参数个数，等于PMU数量
        self.DT_num = self.K  # 传输延迟D_t参数个数，等于PMU数量

        self.bisection = Bisection()  # 二分法求解器
        # 连接矩阵
        self.H = np.array([[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                           [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
                           [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1]])
        self.all_alpham = []
        self.backTracking_alpham(0, self.N, [], self.all_alpham)
        self.qfunction = Qfunction()

    # 设置总带宽
    def setBtotal(self, Btotal):
        self.Btotal = Btotal

    # 设置优化参数个数 alpha  B  e
    def setOptimizeParam_num(self, B_num, theta_num, error_num, DT_num):
        self.B_num = B_num  # 参数带宽的个数     等于PMU数量
        self.theta_num = theta_num  # theta参数个数，等于PMU数量
        self.error_num = error_num  # 误码率e参数个数，等于PMU数量
        self.DT_num = DT_num  # 误码率e参数个数，等于PMU数量

    ##=================================统一调用增广目标函数入口函数======================================================##
    def func_target_subject(self, func_name_subject, X, mk):
        B = X[0:9]
        theta = X[self.B_num:self.B_num + self.theta_num]
        error = X[self.B_num + self.theta_num:27]
        D_t = X[27:]

        if func_name_subject == "func_OR_subject":
            return self.func_OR_subject(B, theta, D_t, error, mk)
        elif func_name_subject == "func_OS_subject":
            return self.func_OS_subject(B, theta, D_t, error, mk)
        else:
            return self.func_OP_subject(B, theta, D_t, error, mk)

    ##=================================统一调用目标函数入口函数======================================================##
    def func_target(self, func_name, X):
        B = X[0:self.B_num]
        theta = X[self.B_num:self.B_num + self.theta_num]
        error = X[self.B_num + self.theta_num:27]
        D_t = X[27:]
        if func_name == "func_OR":
            return self.func_OR(B, theta, D_t, error)
        elif func_name == "func_OS":
            return self.func_OS(B, theta, D_t, error)
        else:
            return self.func_OP(B, theta, D_t, error)

    ##================================= 加上惩罚函数的优化目标函数：增广目标函数===========================================##
    # OR优化目标函数  加上惩罚函数
    def func_OR_subject(self, B, theta, D_t, error, mk):
        return self.func_OR(B, theta, D_t, error) + mk * (
                self.subject_B(B, self.Btotal) + self.subject_error(error) + self.subject_EC(B, theta, error))

    # OR优化目标函数  加上惩罚函数
    def func_OS_subject(self, B, theta, D_t, error, mk):
        return self.func_OS(B, theta, D_t, error) + mk * (
                self.subject_B(B, self.Btotal) + self.subject_error(error) + self.subject_EC(B, theta, error))

    # OR优化目标函数  加上惩罚函数
    def func_OP_subject(self, B, theta, D_t, error, mk):
        return self.func_OP(B, theta, D_t, error) + mk * (
                self.subject_B(B, self.Btotal) + self.subject_error(error) + self.subject_EC(B, theta, error))

    ##===================================优化目标函数===============================================##
    # OR优化目标函数
    def func_OR(self, B, theta, D_t, error):
        lambda_P = self.lambdaP(B, theta, D_t, error, self.N)
        b = self.caculate_b(self.H, lambda_P, self.install_X)
        one_N = np.ones((1, self.N), np.int8)
        return np.matmul(one_N, b)[0, 0]

    # OS优化目标函数
    def func_OS(self, B, theta, D_t, error):
        lambda_P = self.lambdaP(B, theta, D_t, error, self.N)
        b = self.caculate_b(self.H, lambda_P, self.install_X)
        return min(b)

    # OP优化目标函数
    def func_OP(self, B, theta, D_t, error):
        lambda_P = self.lambdaP(B, theta, D_t, error, self.N)
        sum_pro = 0
        for alpham in self.all_alpham:
            tem = 1
            for i in range(self.N):
                if alpham[i] == 1:
                    tem = tem * lambda_P[i][i]
                else:
                    tem = tem * (1 - lambda_P[i][i])
            sum_pro = sum_pro + tem
        return sum_pro

    ##===================================概率和矩阵函数===============================================##
    # 获取可观测性概率p的函数形式
    # p = 1- 2^(Rth*Dmax*theta/ln(2))  精简形式 被缩小
    # def func_pro(self, alpha, B, theta, K):
    #     pr = []
    #     for i in range(K):
    #         pr.append(1 - math.pow(2, -(self.Rth * self.Dmax * theta[i]) / math.log(2)))
    #     return pr

    # 获取可观测性概率P的函数形式
    # P为9维向量，每个元素为对应PMU时延概率
    # p = 1- exp(theta*EC*Dmax)
    def func_pro(self, B, theta, D_t, error, K):
        pr = np.zeros((K))
        for i in range(K):
            B_i = B[i]
            theta_i = theta[i]
            error_i = error[i]
            SNR = self.snr[i]
            D_t_i = D_t[i]
            m = B_i * D_t_i
            EC = self.qfunction.EC_function(B_i, SNR, m, error_i, theta_i, self.T)
            pr_single = 1 - math.exp(-theta_i * EC * (self.Dmax))
            pr[i] = pr_single
        return pr

    # lambdaP矩阵 可观测性矩阵
    def lambdaP(self, B, theta, D_t, error, N):
        lambda_P = np.zeros((N, N), np.float64)
        pr = self.func_pro(B, theta, D_t, error, self.K)
        for k, v in self.bus_pmu.items():
            lambda_P[k - 1, k - 1] = pr[v - 1] * (1 - error[v - 1])
        return lambda_P

    # 计算期望可观测性向量
    def caculate_b(self, H, lambda_P, install_X):
        return np.matmul(np.matmul(H, lambda_P), install_X)

    ##===================================计算出所有可能的alpham的取值===============================================##
    # 计算出所有可能的alpham的取值
    def backTracking_alpham(self, index, n, temp_result, result):
        if index == n:
            if self.checkAlpham(temp_result):
                result.append(temp_result[:])
            return
        for i in range(2):
            # 如果总线上没有安装PMU，则必然只能为0 提前排除掉其为1的可能性
            if self.install_X[index, 0] == 0 and i == 1:
                continue
            temp_result.append(i)
            self.backTracking_alpham(index + 1, n, temp_result, result)
            del temp_result[-1]

    def checkAlpham(self, alpham):
        lambda_P_alpham = np.zeros((self.N, self.N), np.float64)
        for i in range(len(alpham)):
            lambda_P_alpham[i, i] = alpham[i]
        b = self.caculate_b(self.H, lambda_P_alpham, self.install_X)
        flag = True
        for i in range(len(b)):
            if b[i] >= 1:
                if self.install_X[i, 0] == 0 and alpham[i] == 1:  # 表示没有安装PMU的总线上alphm必然为0，不可能为1
                    flag = False
                    break
                else:
                    flag = True
            else:
                flag = False
                break
        return flag

    ##===================================约束条件===============================================##
    # 总带宽约束条件
    # 分别大于0 且 小于Btotal
    # 总带宽之和要等与Btotla,满足此条件做法：
    # 令B9 = Btotal - sum of B from 1 to 8
    def subject_B(self, B, Btotal):
        tem = 0
        for i in range(len(B) - 1):
            tem = tem + math.pow(max(-B[i], 0), 2) + math.pow(max(B[i] - Btotal, 0), 2)
        return tem

    # error误码率约束条件
    # 分别大于0 且 小于1
    def subject_error(self, error):
        tem = 0
        for e in error:
            tem += math.pow(max(-e, 0), 2) + math.pow(max(e - 1, 0), 2)
        return tem

    # 有效容量的约束：
    # 对于每一个PMU，EC>=Rth
    def subject_EC(self, B, theta, error):
        sum_EC = 0
        for i in range(self.K):
            B_i = B[i]
            theta_i = theta[i]
            error_i = error[i]
            SNR = self.snr[i]
            m = 2 * self.Dmax * B_i
            EC = self.qfunction.EC_function(B_i, SNR, m, error_i, theta_i, self.T) - self.Rth
            sum_EC += math.pow(max(-EC, 0), 2)
        return sum_EC


if __name__ == '__main__':
    all_alpham1 = []
    targetFunction = TargetFunction()
    targetFunction.backTracking_alpham(0, 14, [], all_alpham1)

    # self.all_alpham = []
    # self.backTracking_alpham(0, self.N, [], self.all_alpham)
    print(all_alpham1)
