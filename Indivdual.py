# 个体类
class indivdual:
    def __init__(self):
        self.X = []  # 染色体编码  是27为浮点数向量 B1~B9 theta1~tehta9 e1~e9
        self.fitness = 0  # 个体适应度值  根据目标函数进行计算
        self.select_pr = 0 # 个体被选择留到下一代的概率

    def __eq__(self, other):
        self.X = other.X
        self.fitness = other.fitness
