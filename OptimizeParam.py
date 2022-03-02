# 设定模拟退火仿真初始参数类
class OptimizeParam:
    def __init__(self):
        self.B_FDMA = [157000, 159000, 161000, 163000, 165000, 167000, 169000, 171000, 173000, 175000, 177000,
                       179000, 181000, 183000, 185000, 187000, 189000, 191000, 193000, 195000, 197000,
                       199000]  # 修改程序 更改snr位置 增大snr 增大减小snr

    def ParameterSetting(self, algorithm_name):
        self.cName = "IEEE_bus_14"  # 定义问题名称
        self.nVar = 27  # 给定自变量数量，y=f(x1,..xn)
        self.tInitial = 20.0  # 设定初始退火温度(initial temperature)
        self.tFinal = 1.0  # 设定终止退火温度(stop temperature)
        self.alfa = 0.98  # 设定降温参数，T(k)=alfa*T(k-1)
        self.meanMarkov = 100  # Markov链长度，也即内循环运行次数L
        self.scale = 0.5  # 定义搜索步长，可以设为固定值或逐渐缩小  以0.99缩小

        self.m = 1000  # 块传输符号个数
        self.theta_lo = 1e-8
        self.theta_hi = 0.1

        self.func_name_subject = "func_" + algorithm_name + "_subject"  # func_OR_subject func_OS_subject func_OP_subject
        self.func_name = "func_" + algorithm_name  # func_OR func_OS func_OP

        self.B_num = 9  #  参数带宽的个数     等于PMU数量
        self.theta_num = 9  # theta参数个数，等于PMU数量
        self.error_num = 9  # 误码率e参数个数，等于PMU数量

        # 给定搜索空间的下限
        # 参数为1X27维向量，每一维依次代表含义：
        # B1,B2,B3,B4,B5,B6,B7,B8,B9
        # theta1,theta2,theta3,theta4,theta5,theta6,theta7,theta8,theta9
        # e1,e2,e3,e4,e5,e6,e7,e8,e9
        # B、theta、e的索引对应着PMU索引

        self.xMax = [30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000, 31000, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
                     0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]

        self.xMin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7,
                     1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7]  # 增大减小snr

        self.xInitial = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-3, 1e-3,
                         1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]  # 增大减小snr

        # 初始化值
        # 初始化值初始化策略：第一个仿真初始化参数由自己指定
        # 之后上一轮优化得到的最佳结果作为下一个总带宽的优化初始参数 可保证结果不动荡起伏
        self.BTH = self.B_FDMA

        return self.cName, self.nVar, self.xMin, self.xMax, self.xInitial, self.tInitial, self.tFinal, self.alfa, self.meanMarkov, \
               self.scale, self.m, self.theta_lo, self.theta_hi, self.func_name_subject, self.func_name, \
               self.error_num, self.B_num, self.theta_num, self.BTH
