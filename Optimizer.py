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
    def OptimizationSSA(self, nVar, xMin, xMax, xInitial, tInitial, tFinal, alfa, meanMarkov, scale, theta_lo, theta_hi,
                        func_name_subject, model):

        if model == "equal_bandwidth_error":
            X = np.zeros((nVar))
            B_9 = self.targetFunction.Btotal / 9
            for i in range(self.targetFunction.B_num):
                X[i] = B_9

            flag = self.checkSolution(X)
            if not flag:
                return X, -100

            theta_bar = self.bisection_theta_bar(X, theta_lo, theta_hi)
            X[9:18] = [x / 2 for x in theta_bar]  # theta
            X[27:36] = [0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025]  # D

            X[18:27] = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]  # error

            B = X[0:9]
            theta = X[9:18]
            D_t = X[27:]
            R_th = 60000
            T = 0.005
            snr = self.targetFunction.snr
            error = []
            for i in range(self.targetFunction.K):
                B_i = B[i]
                theta_i = theta[i]
                D_t_i = D_t[i]
                SNR = snr[i]
                ec_infinity = self.qfunction.EC_function_infinity(B_i, SNR, theta_i, T)
                qx = math.sqrt(D_t_i / B_i) * (ec_infinity - R_th) * math.log(2)
                epsilon_i = self.qfunction.Qfunc(qx)
                error.append(epsilon_i)
            # print("====================二分法产生一个新解xNew:", xNew, "=======================")
            X[18:27] = error[:]
            fxNew = self.targetFunction.func_target_subject(func_name_subject, X, 0)
            return X, fxNew

        # 得到初始解和初始函数值
        xMin, xMax, xInitial, fxInitial, = self.param_initial(xMin, xMax, xInitial, func_name_subject)

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
                xNew = self.state_generate_fuc(nVar, xNow, xMax, xMin, scale, theta_lo, theta_hi, func_name_subject)

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
                # print("总运行次数：", totalMar, "运行中间最佳目标结果fxBest：", fxBest, "xBest:", xBest)

            ##===========================温度更新函数========================================================##
            # 缓慢降温至新的温度，降温曲线：T(k)=alfa*T(k-1)
            tNow = alfa * tNow
            ##===========================更新惩罚因子========================================================##
            kIter = kIter + 1
            # mk += kIter
            mk = 0
            # self.targetFunction.func_target_subject(func_name_subject, xBest, mk)
            # fxBest = self.targetFunction.func_target_subject(func_name_subject, xBest, mk)  # 由于迭代后惩罚因子增大，需随之重构增广目标函数
            if totalMar % 1000 == 0:
                print("=============当前温度为：", tNow, " 外层循环次数为：", kIter, "===============")
                print("总运行次数：", totalMar, "运行中间最佳目标结果fxBest：", fxBest, "xBest:", xBest)
                # break

            ##============================= 结束模拟退火过程 ================================================##
        return xBest, fxBest

    # 产生问题的初值解和初值解对应的函数值
    # 同时设定仿真参数的最小值和最大值
    def param_initial(self, xMin, xMax, xInitial, func_name_subject):

        B_avg = self.targetFunction.Btotal / self.targetFunction.B_num

        for index in range(self.targetFunction.K):
            xMin[index] = B_avg - 6 * 1000 if xMin[index] < B_avg - 6 * 1000 else xMin[index]
            xMax[index] = B_avg + 8 * 1000

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
    # Dt1,Dt2,Dt3,Dt4,Dt5,Dt6,Dt7,Dt8,Dt9
    # B、theta、e、Dt的索引对应着PMU索引
    def state_generate_fuc(self, nVar, xNow, xMax, xMin, scale, theta_lo, theta_hi, func_name_subject):
        ##================================产生新解XNew=================================================##
        xNew = self.generate_xNew_by_B_noarq(nVar, xNow, xMax, xMin, scale, func_name_subject)
        return xNew

    # 随机扰动B产生新解
    def generate_xNew_by_B_noarq(self, nVar, xNow, xMax, xMin, scale, func_name_subject):
        xNew = np.zeros((nVar))  # 新解
        xNew[:] = xNow[:]

        # 参数 B1,B2,B3,B4,B5,B6,B7,B8,B9由随机产生
        # 为保证满足所有带宽之和等于Btotal，B1,B2,B3,B4,B5,B6,B7,B8由随机扰动方式产生，B9 = Btotal-其余八个带宽之和
        v = random.randint(0, 7)  # 产生 [0,random_var_num-1]即[0,7]之间的随机数

        while True:
            # random.normalvariate(0, 1)：产生服从均值为0、标准差为 1 的正态分布随机实数
            xNew[v] = xNow[v] + scale * (xMax[v] - xMin[v]) * random.normalvariate(0, 1)
            xNew[v] = max(min(xNew[v], xMax[v]), xMin[v])  # 保证新解在 [min,max] 范围内
            sum = 0
            for index in range(self.targetFunction.B_num - 1):
                sum += xNew[index]
            xNew[self.targetFunction.B_num - 1] = self.targetFunction.Btotal - sum
            if xNew[self.targetFunction.B_num - 1] >= xMin[self.targetFunction.B_num - 1]:
                break

        theta, D_t = self.generate_D_theta_by_B(nVar, xNew, 1e-8, 0.4)
        xNew[9:18] = theta[:]
        xNew[27:] = D_t[:]

        ##================================得到epsilon================================================##
        B = xNew[0:9]
        theta = xNew[9:18]
        D_t = xNew[27:]
        R_th = 60000
        D_max = 0.01
        T = self.targetFunction.T
        snr = self.targetFunction.snr
        error = []
        for i in range(self.targetFunction.K):
            B_i = B[i]
            theta_i = theta[i]
            D_t_i = D_t[i]
            SNR = snr[i]
            ec_infinity = self.qfunction.EC_function_infinity(B_i, SNR, theta_i, T)
            qx = math.sqrt(D_t_i / B_i) * (ec_infinity - R_th) * math.log(2)
            epsilon_i = self.qfunction.Qfunc(qx)
            error.append(epsilon_i)
        # print("====================二分法产生一个新解xNew:", xNew, "=======================")
        xNew[18:27] = error[:]
        return xNew

    def generate_D_theta_by_B(self, nVar, xNow, theta_lo, theta_hi):
        xNew = np.zeros((nVar))  # 新解
        xNew[:] = xNow[:]
        theta_bar = self.bisection_theta_bar(xNow, theta_lo, theta_hi)
        B = xNew[0:9]
        theta = []
        D_t = []
        for i in range(9):
            D_t_opt = 0.001
            theta_opt = 1e-8
            B_i = B[i]
            theta_bar_i = theta_bar[i]
            snr_i = self.targetFunction.snr[i]
            while True:
                theta_opt = self.find_theta_opitimal(D_t_opt, B_i, snr_i, theta_bar_i)
                f1 = self.f_function(theta_opt, D_t_opt, B_i, snr_i)
                D_t_opt = self.find_D_t_opitimal(theta_opt, B_i, snr_i)
                f2 = self.f_function(theta_opt, D_t_opt, B_i, snr_i)
                if math.fabs(f1 - f2) < 0.00001:
                    break
            D_t.append(D_t_opt)
            theta.append(theta_opt)
        return theta, D_t

    def find_theta_opitimal(self, D_t, B, SNR, theta_bar):
        zu = theta_bar  # alpha_up
        zl = 0.000001  # alpha_low
        d = ((math.sqrt(5) - 1) / 2) * (zu - zl)
        z = [0, zl + d, zu - d]
        f = [0, 0, 0]
        while math.fabs(zu - zl) > 0.0001:
            v = 1
            while v <= 2:
                theta_v = z[v]
                f[v] = self.f_function(theta_v, D_t, B, SNR)
                v = v + 1
            if math.fabs(f[1] - f[2]) < 0.0001:
                break
            elif f[1] < f[2]:
                zu = z[1]
                d = ((math.sqrt(5) - 1) / 2) * (zu - zl)
                z[1] = z[2]
                z[2] = zu - d
            elif f[1] > f[2]:
                zl = z[2]
                d = ((math.sqrt(5) - 1) / 2) * (zu - zl)
                z[2] = z[1]
                z[1] = zl + d
        theta_max = (zu + zl) / 2
        return theta_max

    def find_D_t_opitimal(self, theta, B, SNR):
        zu = 0.005  # alpha_up
        zl = 0.0001  # alpha_low
        d = ((math.sqrt(5) - 1) / 2) * (zu - zl)
        z = [0, zl + d, zu - d]
        f = [0, 0, 0]
        while math.fabs(zu - zl) > 0.0001:
            v = 1
            while v <= 2:
                D_t_v = z[v]
                f[v] = self.f_function(theta, D_t_v, B, SNR)
                v = v + 1
            if math.fabs(f[1] - f[2]) < 0.0001:
                break
            elif f[1] < f[2]:
                zu = z[1]
                d = ((math.sqrt(5) - 1) / 2) * (zu - zl)
                z[1] = z[2]
                z[2] = zu - d
            elif f[1] > f[2]:
                zl = z[2]
                d = ((math.sqrt(5) - 1) / 2) * (zu - zl)
                z[2] = z[1]
                z[1] = zl + d
        D_max = (zu + zl) / 2
        return D_max

    def f_function(self, theta, D_t, B, SNR):
        R_th = 60000
        D_max = 0.01
        T = self.targetFunction.T
        p = 1 - math.exp(-theta * R_th * (D_max - D_t))
        ec_infinity = self.qfunction.EC_function_infinity(B, SNR, theta, T)
        qx = math.sqrt(D_t / B) * (ec_infinity - R_th) * math.log(2)
        return (p) * (1 - self.qfunction.Qfunc(qx))

    def bisection_theta_bar(self, xNew, theta_lo, theta_hi):
        B = xNew[0:9]
        snr = self.targetFunction.snr
        theta_bar = []
        for i in range(self.targetFunction.K):
            B_i = B[i]
            SNR = snr[i]
            theta = self.bisection.calculate_theta_bar_by_bisection(B_i, SNR, theta_lo, theta_hi)
            theta_bar.append(theta)
        return theta_bar

    def checkSolution(self, xNew):
        ##=======================判断新解是否有效==================================================##
        # theta1,theta2,theta3,theta4,theta5,theta6,theta7,theta8,theta9
        # 对于得到新解中的alpha和B作为已知参数，为了使用二分法去求解theta，判断theta取值极小时EC是否大于等于Rth
        flag = True
        snr = self.targetFunction.snr
        B = xNew[0:9]
        flag = True
        for i in range(9):
            B_i = B[i]
            SNR = snr[i]
            single_func = self.bisection.EC_B_theta_bar(B_i, SNR, self.theta_min)
            if single_func < 0:
                flag = False
                break
        return flag

