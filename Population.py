from typing import List, Any
from Indivdual import indivdual


# 种群类
class population:
    indivduals: List[indivdual]

    def __init__(self):
        # 种群是个二维数组，个体和染色体两维, population_size*choromosome_length
        self.indivduals = []  # 元素是indivdual个体
