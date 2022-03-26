# 主程序入口
# author: zqy
from OptimizeParam import OptimizeParam
from Optimizer import Optimizer
from TargetFunction import TargetFunction
import numpy as np


class Main:
    def __init__(self):
        self.targetFunction = TargetFunction()

    def main(self, algorithm_name):
        optimizer = Optimizer()
        optimizeParam = OptimizeParam()
        xBest_List = []
        fxBest_List = []
        fBest_List = []
        reduce = self.targetFunction.N if algorithm_name == "OR" else 0

        # 参数设置，优化问题参数定义，模拟退火算法参数设置
        [cName, nVar, xMin, xMax, xInitial, tInitial, tFinal, alfa, meanMarkov, scale, m, theta_lo, theta_hi,
         func_name_subject, func_name, error_num, B_num, theta_num, BTH] = optimizeParam.ParameterSetting(
            algorithm_name)
        # 设置用户配对
        optimizer.getTargetFunction().setOptimizeParam_num(B_num, theta_num, error_num)
        self.targetFunction.setOptimizeParam_num(B_num, theta_num, error_num)

        for index in range(len(BTH)):
            # 设置总带宽
            optimizer.getTargetFunction().setBtotal(BTH[index])
            # 模拟退火算法
            [xBest, fxBest] = optimizer.OptimizationSSA(nVar, xMin, xMax, xInitial, tInitial, tFinal, alfa, meanMarkov,
                                                        scale, m, theta_lo, theta_hi, func_name_subject, func_name)
            xBest_List.append(xBest.tolist())
            fxBest_List.append(fxBest)

            res = self.targetFunction.func_target(func_name, xBest) - reduce
            fBest_List.append(res)
            print("========模拟退火运行", index, "次结束，总带宽Bth:", BTH[index], "xBest:", xBest, "fxBest:", fxBest,
                  "真实优化目标结果为：", res)

            optimizeParam.xInitial[:] = xBest[:]  # 将本轮优化结果作为下轮优化的初始参数

        # 将结果保存为txt文件
        # self.saveTxt("result/xBest_List_{}_{}_{}".format(user_pair_algorithm, algorithm_name, other), xBest_List)
        # self.saveTxt("result/fxBest_List_{}_{}_{}".format(user_pair_algorithm, algorithm_name, other), fxBest_List)
        self.saveTxt("result/fBest_List_{}".format(algorithm_name), fBest_List)

    # 将数组保存为txt文件
    def saveTxt(self, path, np_list):
        np.savetxt(path + ".txt", np_list, fmt='%f', delimiter=',')

    # 从txt文件读取数组
    def loadTxt(self, path):
        res = np.loadtxt("result/" + path + ".txt", delimiter=',')
        return res


if __name__ == '__main__':
    main = Main()
    print("OR")
    main.main("OR")  # OR OS OP三种算法  三种用户配对算法
