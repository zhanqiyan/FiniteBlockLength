from TargetFunction import TargetFunction
import math  # 导入模块
import random  # 导入模块
import numpy as np  # 导入模块 numpy，并简写成 np
from Bisection import Bisection
from qfunc import Qfunction
from Indivdual import indivdual
from Population import population
from OptimizeParam import OptimizeParam


class GeneticAlgorithm:
    # 初始化种群 生成chromosome_length大小的population_size个个体的种群
    def __init__(self, population_size, chromosome_length, pc, pm):
        self.qfunc = Qfunction()
        self.targetFunction = TargetFunction()
        self.population_size = population_size  # 种群的大小
        self.choromosome_length = chromosome_length  # 每个染色体的长度，也即是参数的个数 27个
        self.pc = pc
        self.pm = pm

    # 定义适应度函数：本文适应度函数设定为目标函数
    def fitness_func(self, X, func_name_subject):
        fitness_value = self.targetFunction.func_target_subject(func_name_subject, X, 0)
        return fitness_value

    # 初始化种群,返回一个种群类
    # 种群是个二维数组，个体和染色体两维, population_size*choromosome_length
    # 即种群个数*浮点数参数个数
    def initPopulation(self, func_name_subject):
        pop = population()
        Opt = OptimizeParam()
        Opt.ParameterSetting("OR")
        for i in range(self.population_size):
            ind = indivdual()  # 个体初始化
            ind.X = Opt.xInitial  # 个体编码。-10,10的正态分布，可以自己设定限定边界
            ind.fitness = self.fitness_func(ind.X, func_name_subject)  # 计算个体适应度函数值
            pop.indivduals.append(ind)  # 将个体适应度函数值添加进种群适应度数组pop
        return pop

    # 计算适应度和
    def sum(self, pop):
        total = 0
        for i in range(len(pop.indivduals)):
            total += pop.indivduals[i].fitness
        return total

    # np.random.choice
    # 3.对种群的的个体进行自然选择 返回被选择后的种群
    def selection(self, pop):
        # 将所有的适应度求和
        total_fitness = self.sum(pop)
        pr=[]
        # 计算个体被选择概率
        for i in range(len(pop.indivduals)):
            pop.indivduals[i].select_pr = pop.indivduals[i].fitness / total_fitness
            pr.append(pop.indivduals[i].fitness / total_fitness)
        index = np.random.choice(a = np.arange(len(pop.indivduals)), size = self.population_size, replace = False, p = pr)

        new_pop = population()
        for i in range(len(index)):
            new_pop.indivduals.append(pop.indivduals[i])
        return index


    # 遗传算法入口
    def GA(self):

        return 0

    # 4.交叉操作
    def crossover(self, population):
        # pc是概率阈值，选择单点交叉还是多点交叉，生成新的交叉个体，这里没用
        pop_len = len(population)

        for i in range(pop_len - 1):

            if (random.random() < self.pc):
                cpoint = random.randint(0, len(population[0]))
                # 在种群个数内随机生成单点交叉点
                temporary1 = []
                temporary2 = []

                temporary1.extend(population[i][0:cpoint])
                temporary1.extend(population[i + 1][cpoint:len(population[i])])
                # 将tmporary1作为暂存器，暂时存放第i个染色体中的前0到cpoint个基因，
                # 然后再把第i+1个染色体中的后cpoint到第i个染色体中的基因个数，补充到temporary2后面

                temporary2.extend(population[i + 1][0:cpoint])
                temporary2.extend(population[i][cpoint:len(population[i])])
                # 将tmporary2作为暂存器，暂时存放第i+1个染色体中的前0到cpoint个基因，
                # 然后再把第i个染色体中的后cpoint到第i个染色体中的基因个数，补充到temporary2后面
                population[i] = temporary1
                population[i + 1] = temporary2
        # 第i个染色体和第i+1个染色体基因重组/交叉完成

    def mutation(self, population):
        # pm是概率阈值
        px = len(population)
        # 求出种群中所有种群/个体的个数
        py = len(population[0])
        # 染色体/个体基因的个数
        for i in range(px):
            if (random.random() < self.pm):
                mpoint = random.randint(0, py - 1)
                #
                if (population[i][mpoint] == 1):
                    # 将mpoint个基因进行单点随机变异，变为0或者1
                    population[i][mpoint] = 0
                else:
                    population[i][mpoint] = 1

    # # transform the binary to decimalism
    # # 将每一个染色体都转化成十进制 max_value,再筛去过大的值
    # def b2d(self, best_individual):
    #     total = 0
    #     b = len(best_individual)
    #     for i in range(b):
    #         total = total + best_individual[i] * math.pow(2, i)
    #
    #     total = total * self.max_value / (math.pow(2, self.choromosome_length) - 1)
    #     return total

    # 寻找最好的适应度和个体

    def best(self, population, fitness_value):

        px = len(population)
        bestindividual = []
        bestfitness = fitness_value[0]
        # print(fitness_value)

        for i in range(1, px):
            # 循环找出最大的适应度，适应度最大的也就是最好的个体
            if (fitness_value[i] > bestfitness):
                bestfitness = fitness_value[i]
                bestindividual = population[i]

        return [bestindividual, bestfitness]

    # def main(self):
    #
    #     results = [[]]
    #     fitness_value = []
    #     fitmean = []
    #
    #     population = pop = self.species_origin()
    #
    #     for i in range(500):
    #         function_value = self.function(population)
    #         # print('fit funtion_value:',function_value)
    #         fitness_value = self.fitness(function_value)
    #         # print('fitness_value:',fitness_value)
    #
    #         best_individual, best_fitness = self.best(population, fitness_value)
    #         results.append([best_fitness, self.b2d(best_individual)])
    #         # 将最好的个体和最好的适应度保存，并将最好的个体转成十进制,适应度函数
    #         self.selection(population, fitness_value)
    #         self.crossover(population)
    #         self.mutation(population)
    #     results = results[1:]
    #     results.sort()

    # def select(self, pop, fitness):
    #     POP_SIZE = len(pop)
    #     idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitness / sum(fitness))
    #     # 我们只要按照适应程度 fitness 来选 pop 中的 parent 就好. fitness 越大, 越有可能被选到.
    #     return pop[idx]


if __name__ == '__main__':
    GA = GeneticAlgorithm(10, 27, 0.6, 0.6)
    GA.targetFunction.setBtotal(160000)
    pop = GA.initPopulation("func_OR_subject")
    total = GA.sum(pop)
    GA.population_size = 3

    new_pop = GA.selection(pop)
    print(new_pop)

