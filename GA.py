import numpy as np
import copy
from time import time
import matplotlib.pyplot as plt

# 随机数种子
import tsplib95

np.random.seed(114514)


# 遗传算法
class Genetic_Algorithm():
    def __init__(self, pop, pop_size, DNA_size, distance, city_position, crossover_rate=0.7, mutation_rate=0.05):
        # 交叉概率
        self.crossover_rate = crossover_rate
        # 变异概率
        self.mutation_rate = mutation_rate
        # 种群
        self.pop = pop
        # 种群大小
        self.pop_size = pop_size
        # DNA大小
        self.DNA_size = DNA_size
        # 城市坐标
        self.city_position = city_position
        # 城市间距离矩阵
        self.distance = distance
        pass

    # 计算种群个体适应度
    def compute_fitness(self, pop):
        # 初始化一个空表
        fitness = np.zeros(self.pop_size, dtype=np.float32)
        # 枚举每个个体
        for i, e in enumerate(pop):
            # print(e)
            for j in range(self.DNA_size - 1):
                # 计算个体i的适应度
                fitness[i] += self.distance[int(e[j])][int(e[j + 1])]
        # 记录距离
        dis = copy.copy(fitness)
        # 适应度等于距离的倒数
        fitness = np.reciprocal(fitness)
        return fitness, dis

    # 轮盘赌，选择种群中的个体
    def select_population(self, fitness):
        # 从种群中选择，pop_size个个体，每个个体被选择的概率为fitness / fitness.sum()
        indx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        # print(indx)
        # 花式索引，更新种群
        self.pop = self.pop[indx]

    # 对新种群中的所有个体进行基因交叉
    def genetic_crossover(self):
        # 遍历种群每个个体
        for parent1 in self.pop:
            # 判断是否会基因交叉
            if np.random.rand() < self.crossover_rate:
                #### Subtour Exchange Crossover
                #### 基因交换方法参考6
                #####基因交叉参考https://blog.csdn.net/ztf312/article/details/82793295
                # 寻找父代2
                n = np.random.randint(self.pop_size)
                parent2 = self.pop[n, :]
                # 随机产生基因交换片段
                pos = np.random.randint(self.DNA_size, size=2)
                # 区间左右端点
                l = min(pos)
                r = max(pos)
                # 记录区间
                seq = copy.copy(parent1[l:r])
                poss = []
                # 交换
                for i in range(self.DNA_size):
                    if parent2[i] in seq:
                        poss.append(i)
                a = 0
                for i in seq:
                    parent2[poss[a]] = i
                    a += 1
                b = 0
                for i in range(l, r):
                    parent1[i] = parent2[poss[b]]
                    b += 1
                # print(parent1)
                # break

    # 种群中的所有个体基因突变
    def genetic_mutation(self):
        # 枚举个体
        for e in self.pop:
            # 变异的可能
            if np.random.rand() < self.mutation_rate:
                # 随机变异交换点
                position = np.random.randint(self.DNA_size, size=2)
                e[position[0]], e[position[1]] = e[position[1]], e[position[0]]
        pass


# 初始化种群
def init_pop(pop_size, DNA_size):
    # 初始化一个种群 大小为pop_size*DNA_size
    pop = np.zeros((pop_size, DNA_size))
    # DNA编码
    code = np.arange(DNA_size)
    for i in range(pop_size):
        pop[i] = copy.deepcopy(code)
        # 随机打乱函数
        # np.random.shuffle(pop[i])
    # 返回种群
    return pop


# 画图
def show(lx, ly, city_position, dis):
    plt.cla()  # 清除键
    # 画点
    plt.scatter(city_position[0], city_position[1], color='r')
    # 画线
    plt.plot(lx, ly)
    # 不显示坐标轴
    plt.xticks([])
    plt.yticks([])
    # 文本注释
    plt.text(-5.25, -14.05, "Total distance=%.2f" % dis, fontdict={'size': 20, 'color': 'red'})
    # 暂停时间
    plt.pause(0.01)


def best_show(x, y, city_position, best_fitness, dis):
    # 定义两个子图
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), facecolor='#ccddef')
    # 定义子图1标题
    ax[0].set_title("Best route")
    # 定义子图2标题
    ax[1].set_title("Fitness Change Procession")
    # 画线
    ax[0].plot(x, y)
    # 画点
    ax[0].scatter(city_position[0], city_position[1], color='r')
    # 画线
    ax[1].plot(range(len(best_fitness)), best_fitness)
    plt.show()
    pass


# 解决旅行商问题
def TSP(city_position, pop_size, DNA_size, distance, t):
    # 初始化一个种群
    pop = init_pop(pop_size, DNA_size)
    # 调用遗传算法类
    GA = Genetic_Algorithm(pop, pop_size, DNA_size, distance, city_position)
    # 保存最佳距离
    best_distance = 1e6
    # 保存最佳路线
    route = None
    # 保存最佳x坐标
    x = None
    # 保存最佳y坐标
    y = None
    # 保存适应度变化曲线
    fitness_process = []
    for i in range(t):
        # t-=1
        # 返回适应度，和距离函数
        fitness, dis = GA.compute_fitness(GA.pop)
        # 选择新的种群
        GA.select_population(fitness)
        # 基因交叉
        GA.genetic_crossover()
        # 基因突变
        GA.genetic_mutation()
        ####################
        # 记录当前状态最优解
        # 返回最优解索引
        num = np.argmax(fitness)
        # 记录DNA
        DNA = GA.pop[num, :]
        # 打印当前状态
        print(f"The step is {i} ,the current best distance is {min(dis)} ,fitness is {max(fitness)}")
        lx = []
        ly = []
        # DNA转化为记录坐标
        fitness_process.append(max(fitness))
        for i in DNA:
            i = int(i)
            lx.append(city_position[0][i])
            ly.append(city_position[1][i])
        # 保存最佳方案
        if best_distance > min(dis):
            best_distance = min(dis)
            route = DNA = GA.pop[num, :]
            x = copy.copy(lx)
            y = copy.copy(ly)
        show(lx, ly, city_position, min(dis))
    # 打印最终结果
    print(f"The best route is {route}")
    print(f"The route distance is {best_distance}")
    best_show(x, y, city_position, fitness_process, best_distance)


if __name__ == "__main__":
    # x = [178, 272, 176, 171, 650, 499, 267, 703, 408, 437, 491, 74, 532, 416, 626, 42, 271, 359, 163, 508, 229, 576,
    #      147, 560, 35, 714, 757, 517, 64, 314, 675, 690, 391, 628, 87, 240, 705, 699, 258, 428, 614, 36, 360, 482, 666,
    #      597, 209, 201, 492, 294]
    # y = [170, 395, 198, 151, 242, 556, 57, 401, 305, 421, 267, 105, 525, 381, 244, 330, 395, 169, 141, 380, 153, 442,
    #      528, 329, 232, 48, 498, 265, 343, 120, 165, 50, 433, 63, 491, 275, 348, 222, 288, 490, 213, 524, 244, 114, 104,
    #      552, 70, 425, 227, 331]
    # x.append(x[0])
    # y.append(y[0])
    x = []
    y = []

    data = tsplib95.load('/Users/huangzihao/Downloads/att48.tsp')
    coords = np.array([(coord[0], coord[1]) for city, coord in data.node_coords.items()])
    for city, coord in data.node_coords.items():
        x.append(coord[0])
        y.append(coord[1])

    distance = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            distance[i][j] = np.sqrt(np.square(x[i] - x[j]) + np.square(y[i] - y[j]))
    # 转换成一个矩阵
    city_position = np.array([x, y])
    # 种群大小
    pop_size = 20
    # DNA大小也是城市大小
    DNA_size = len(x)
    # 迭代次数
    t = 100000
    # sovle problem
    TSP(city_position, pop_size, DNA_size, distance, t)
