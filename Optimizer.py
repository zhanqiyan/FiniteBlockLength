from TargetFunction import TargetFunction
import math  # 导入模块
import random  # 导入模块
import numpy as np  # 导入模块 numpy，并简写成 np
from Bisection import Bisection
from qfunc import Qfunction


# 模拟退火优化器
class Optimizer:
    # 初始化目标函数类
    def __init__(self):
        self.targetFunction = TargetFunction()  # 目标函数
        self.bisection = Bisection()  # 二分法求解器
        self.theta_min = 1e-8
        self.qfunction = Qfunction()

    def getTargetFunction(self):
        return self.targetFunction

    # 模拟退火算法
    def OptimizationSSA(self, nVar, xMin, xMax, xInitial, tInitial, tFinal, alfa, meanMarkov, scale,
                        m, theta_lo, theta_hi, func_name_subject, func_name, model):

        if model == "equal_bandwidth_error":
            X = np.zeros((nVar))
            B_9 = self.targetFunction.Btotal / 9
            for i in range(self.targetFunction.B_num):
                X[i] = B_9
            X[9:18] = [1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7]
            X[18:] = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]

            theta = self.bisection_theta(X, theta_lo, theta_hi, m)
            X[9:18] = theta[:]
            fxNew = self.targetFunction.func_target_subject(func_name_subject, X, 0)
            return X, fxNew

        # 得到初始解和初始函数值
        xMin, xMax, xInitial, fxInitial, = self.param_initial(xMin, xMax, xInitial, m, func_name_subject)

        # ====== 模拟退火算法初始化 ======
        xNew = np.zeros((nVar))  # 新解
        xNow = np.zeros((nVar))  # 当前解
        xBest = np.zeros((nVar))  # 当前最好解

        xNow[:] = xInitial[:]  # 初始化当前解，将初始解置为当前解
        xBest[:] = xInitial[:]  # 初始化最优解，将当前解置为最优解

        fxNow = fxInitial  # 初始化当前增广目标函数值，将初始解的目标函数置为当前值
        fxBest = fxInitial  # 初始化最优增广目标函数值，将当前解的目标函数置为最优值

        kIter = 0  # 外循环迭代次数，温度状态数
        totalMar = 0  # 总计 Markov 链长度
        totalImprove = 0  # fxBest 改善次数
        nMarkov = meanMarkov  # 固定长度 Markov链，即内循环次数L

        ## ======================================= 开始模拟退火优化 ==============================================##
        # 外循环，直到当前温度达到终止温度时结束
        tNow = tInitial  # 初始化当前温度(current temperature)
        # 初始化惩罚因子
        # mk = 1
        mk = 0
        while tNow >= tFinal:  # 外循环，直到当前温度达到终止温度时结束
            # print("=============当前温度为：", tNow, " 外层循环次数为：", kIter, "===============")
            # 在当前温度下，进行充分次数(nMarkov)的状态转移以达到热平衡
            kBetter = 0  # 获得优质解的次数
            kBadAccept = 0  # 接受劣质解的次数
            kBadRefuse = 0  # 拒绝劣质解的次数

            # ---内循环，循环次数为Markov链长度
            for k in range(nMarkov):  # 内循环，循环次数为Markov链长度
                totalMar += 1  # 总 Markov链长度计数器
                ##================================产生新解XNew=================================================##
                xNew = self.state_generate_fuc(nVar, xNow, xMax, xMin, scale, m, theta_lo, theta_hi, func_name_subject)

                ##=======================计算目标函数和能量差=====================================================##
                # 调用目标函数计算新解的目标函数值
                fxNew = self.targetFunction.func_target_subject(func_name_subject, xNew, mk)
                deltaE = fxNew - fxNow

                ##=======================状态接受函数=====================================================##
                # ---按 Metropolis 准则接受新解
                # 接受判别：按照 Metropolis 准则决定是否接受新解
                if fxNew > fxNow:  # 更优解：如果新解的目标函数好于当前解，则接受新解
                    accept = True
                    kBetter += 1
                else:  # 容忍解：如果新解的目标函数比当前解差，则以一定概率接受新解
                    pAccept = math.exp(deltaE / tNow)  # 计算容忍解的状态迁移概率
                    if pAccept > random.random():  # random()方法返回随机生成的一个实数，它在[0,1)范围内
                        accept = True  # 接受劣质解
                        kBadAccept += 1
                    else:
                        accept = False  # 拒绝劣质解
                        kBadRefuse += 1
                ##=======================得到新解进行保存=====================================================##
                if accept == True:  # 如果接受新解，则将新解保存为当前解
                    xNow[:] = xNew[:]
                    fxNow = fxNew
                    if fxNew > fxBest:  # 如果新解的目标函数好于最优解，则将新解保存为最优解
                        fxBest = fxNew
                        xBest[:] = xNew[:]
                        totalImprove += 1
                        # scale = scale * 0.99  # 可变搜索步长，逐步减小搜索范围，提高搜索精度
                print("总运行次数：", totalMar, "运行中间最佳目标结果fxBest：", fxBest)

            ##===========================温度更新函数========================================================##
            # 缓慢降温至新的温度，降温曲线：T(k)=alfa*T(k-1)
            tNow = alfa * tNow
            ##===========================更新惩罚因子========================================================##
            kIter = kIter + 1
            # mk += kIter
            mk = 0
            # self.targetFunction.func_target_subject(func_name_subject, xBest, mk)
            # fxBest = self.targetFunction.func_target_subject(func_name_subject, xBest, mk)  # 由于迭代后惩罚因子增大，需随之重构增广目标函数
            if totalMar % 2000 == 0:
                print("=============当前温度为：", tNow, " 外层循环次数为：", kIter, "===============")
                print("总运行次数：", totalMar, "运行中间最佳目标结果fxBest：", fxBest, "xBest:", xBest)

            ##============================= 结束模拟退火过程 ================================================##
        return xBest, fxBest

    # 产生问题的初值解和初值解对应的函数值
    # 同时设定仿真参数的最小值和最大值
    def param_initial(self, xMin, xMax, xInitial, m, func_name_subject):
        sum = 0
        for index in range(self.targetFunction.B_num - 1):
            sum += xInitial[index]
        xInitial[self.targetFunction.B_num - 1] = self.targetFunction.Btotal - sum
        fxInitial = self.targetFunction.func_target_subject(func_name_subject, xInitial, 0)  # m(k)：惩罚因子，初值为 1
        print("======初始化解，xInitial:", xInitial, "初始化结果fxInitial:", fxInitial, "==========")
        return xMin, xMax, xInitial, fxInitial

    # 状态产生函数：根据当前解产生新解
    # 解参数X为1X27维向量，每一维代表含义：
    # B1,B2,B3,B4,B5,B6,B7,B8,B9
    # theta1,theta2,theta3,theta4,theta5,theta6,theta7,theta8,theta9
    # e1,e2,e3,e4,e5,e6,e7,e8,e9
    def state_generate_fuc(self, nVar, xNow, xMax, xMin, scale, m, theta_lo, theta_hi, func_name_subject):
        ##================================产生新解XNew=================================================##
        xNew = self.generate_random_solution(nVar, xNow, xMax, xMin, scale, m, func_name_subject)

        ##================================使用二分法得到theta=================================================##
        # theta1,theta2,theta3,theta4,theta5,theta6,theta7,theta8,theta9
        # 对于得到新解中的B和error作为已知参数，使用二分法去求解theta
        B = xNew[0:9]
        error = xNew[18:]
        snr = self.targetFunction.snr
        for i in range(self.targetFunction.K):
            B_i = B[i]
            SNR = snr[i]
            error_i = error[i]
            theta_single = self.bisection.calculate_theta_by_bisection(B_i, error_i, SNR, m, theta_lo, theta_hi)
            xNew[self.targetFunction.B_num + i] = theta_single
        # print("====================二分法产生一个新解xNew:", xNew, "=======================")
        return xNew

    # 通过随机扰动产生新解，并且保证新解是有效：给定error和B，能通过二分法找到有效的theta，使得 EC=Rth,具体做法：
    #   1、随机扰动B,然后根据B和theta,根据单调性找到error
    #   2、然后在根据得到的error和B，根据二分法去求解theta，得到新解
    def generate_random_solution(self, nVar, xNow, xMax, xMin, scale, m, func_name_subject):
        # print("==============================产生一次新解====================================")
        snr = self.targetFunction.snr
        ##=======================随机扰动产生新解==================================================##
        while True:
            ##=======================随机扰动产生新解,随机扰动B得到error==================================================##
            # xNew = self.generate_xNew_by_B(nVar, xNow, xMax, xMin, scale, m)
            xNew = self.generate_xNew_by_B_noarq(nVar, xNow, xMax, xMin, scale, m, func_name_subject)

            ##=======================判断新解是否有效==================================================##
            # theta1,theta2,theta3,theta4,theta5,theta6,theta7,theta8,theta9
            # 对于得到新解中的error和B作为已知参数，为了使用二分法去求解theta，判断theta取值极小时EC是否大于等于Rth
            B = xNew[0:9]
            error = xNew[18:]
            flag = True
            for i in range(9):
                B_i = B[i]
                SNR = snr[i]
                error_i = error[i]
                single_func = self.bisection.EC_B_theta(B_i, error_i, SNR, self.theta_min, m)
                if single_func < 0:
                    flag = False
                    break
            if flag:
                break

        # print("验证通过，得到一个新解xNew:", xNew)
        return xNew

    # 随机扰动B产生新解
    # 随机扰动B,然后根据B和theta,根据单调性找到error
    def generate_xNew_by_B_noarq(self, nVar, xNow, xMax, xMin, scale, m, func_name_subject):
        xNew = np.zeros((nVar))  # 新解
        xNew[:] = xNow[:]

        # 参数 B1,B2,B3,B4,B5,B6,B7,B8,B9由随机产生
        # 为保证满足所有带宽之和等于Btotal，B1,B2,B3,B4,B5,B6,B7,B8由随机扰动方式产生，B9 = Btotal-其余八个带宽之和
        v = random.randint(0, 7)  # 产生 [0,random_var_num-1]即[0,7]之间的随机数

        while True:
            # 1 随机产生B
            flag = False
            while True:
                # random.normalvariate(0, 1)：产生服从均值为0、标准差为 1 的正态分布随机实数
                xNew[v] = xNow[v] + scale * (xMax[v] - xMin[v]) * random.normalvariate(0, 1)
                xNew[v] = max(min(xNew[v], xMax[v]), xMin[v])  # 保证新解在 [min,max] 范围内
                sum = 0
                for index in range(self.targetFunction.B_num - 1):
                    sum += xNew[index]
                xNew[self.targetFunction.B_num - 1] = self.targetFunction.Btotal - sum
                if xNew[self.targetFunction.B_num - 1] >= xMin[self.targetFunction.B_num - 1]:
                    flag = True
                    break
            if flag:
                break

        error_initial = 0.0001
        error_MAx = 0.2
        error_max = error_initial
        X_error = np.zeros(nVar)
        X_error[:] = xNew[:]
        func_max = self.targetFunction.func_target_subject(func_name_subject, X_error, 0)

        # X_last = np.zeros(nVar)
        # X_last[:] = X_error[:]
        # B = X_error[v]
        # theta = X_error[9 + v]
        # SNR = self.targetFunction.snr[v]
        # error = X_error[18 + v]

        while error_initial < error_MAx:
            X_error[18 + v] = error_initial
            func = self.targetFunction.func_target_subject(func_name_subject, X_error, 0)
            if func > func_max:
                func_max = func
                error_max = error_initial
            error_initial += 0.01
        xNew[18 + v] = error_max
        return xNew

    # 随机扰动B产生新解
    # 随机扰动B,然后根据B和theta,根据单调性找到error
    def generate_xNew_by_B(self, nVar, xNow, xMax, xMin, scale, m):
        xNew = np.zeros((nVar))  # 新解
        xNew[:] = xNow[:]

        # 参数 B1,B2,B3,B4,B5,B6,B7,B8,B9由随机产生
        # 为保证满足所有带宽之和等于Btotal，B1,B2,B3,B4,B5,B6,B7,B8由随机扰动方式产生，B9 = Btotal-其余八个带宽之和
        v = random.randint(0, 7)  # 产生 [0,random_var_num-1]即[0,7]之间的随机数

        while True:
            # 1 随机产生B
            flag = False
            while True:
                # random.normalvariate(0, 1)：产生服从均值为0、标准差为 1 的正态分布随机实数
                xNew[v] = xNow[v] + scale * (xMax[v] - xMin[v]) * random.normalvariate(0, 1)
                xNew[v] = max(min(xNew[v], xMax[v]), xMin[v])  # 保证新解在 [min,max] 范围内
                sum = 0
                for index in range(self.targetFunction.B_num - 1):
                    sum += xNew[index]
                xNew[self.targetFunction.B_num - 1] = self.targetFunction.Btotal - sum
                if xNew[self.targetFunction.B_num - 1] >= xMin[self.targetFunction.B_num - 1]:
                    flag = True
                    break
            if flag:
                break

        error_initial = 0.0001
        error_max = error_initial
        EC_max = 0
        error_MAx = 0.2
        B = xNow[v]
        theta = xNow[9 + v]
        SNR = self.targetFunction.snr[v]

        while error_initial < error_MAx:
            EC = self.qfunction.EC_function(B, SNR, m, error_initial, theta, self.targetFunction.T)
            if EC > EC_max:
                EC_max = EC
                error_max = error_initial
            error_initial += 0.0001
        xNew[18 + v] = error_max
        return xNew

    def bisection_theta(self, xNew, theta_lo, theta_hi, m):
        ##================================使用二分法得到theta=================================================##
        # theta1,theta2,theta3,theta4,theta5,theta6,theta7,theta8,theta9
        # 对于得到新解中的alpha和B作为已知参数，使用二分法去求解theta
        B = xNew[0:9]
        error = xNew[18:]
        snr = self.targetFunction.snr
        for i in range(self.targetFunction.K):
            B_i = B[i]
            SNR = snr[i]
            error_i = error[i]
            theta_single = self.bisection.calculate_theta_by_bisection(B_i, error_i, SNR, m, theta_lo, theta_hi)
            xNew[self.targetFunction.B_num + i] = theta_single
        return xNew
