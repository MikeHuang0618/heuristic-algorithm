import tsplib95
import numpy as np
import matplotlib.pyplot as plt
import copy


class EPO_TSP:
    def __init__(self, pop, pop_size, EPO_size, lb, ub):
        super(EPO_TSP, self).__init__()
        self.pop = pop
        self.pop_size = pop_size
        self.EPO_size = EPO_size
        self.lb = lb
        self.ub = ub
        self.M = 2
        self.f = 2.0
        self.l = 1.5
        self.Xgb = None
        self.g_fit = float('inf')
        self.X = np.zeros([self.pop_size, self.EPO_size], dtype='float32')
        for i in range(self.pop_size):
            for j in range(self.EPO_size):
                self.X[i, j] = self.lb + (self.ub - self.lb) * np.random.uniform(0.0, 1.0, 1)

    def g_best(self, fitness):
        for i in range(self.pop_size):
            if fitness[i] < self.g_fit:
                self.g_fit = fitness[i]
                self.Xgb = self.X[i, :]

    def update(self, iteration, max_iteration):

        for i in range(self.pop_size):
            R = np.random.rand()
            if R >= 0.5:
                T = 1
            else:
                T = 0

            T_p = (T - (max_iteration / (iteration - max_iteration)))
            r =  np.random.uniform(0.0, 1.0, 1)
            for j in range(self.EPO_size - 1):
                P_grid = np.abs(self.Xgb[j] - self.X[i, j])
                A = (self.M * (T_p + P_grid) * r) - T_p
                C = np.random.uniform(0.0, 1.0, 1)
                S = np.sqrt(self.f * np.exp(-iteration / self.l) - np.exp(-iteration)) ** 2
                Dep = np.abs(S * self.Xgb[j] - C * self.X[i, j])
                self.X[i, j] = self.Xgb[j] - A * Dep

    def SPV(self):
        self.pop = np.zeros((pop_size, EPO_size))
        for i in range(len(self.X)):
            self.pop[i] = sorted(range(len(self.X[i])), key=lambda k: self.X[i][k])
        # print(self.pop)
        return self.pop

    @staticmethod
    def get_distance(arr1, arr2):
        return np.sqrt(np.power(arr1 - arr2, 2).sum())

    def compute_objective_value(self, cities_id, coordinate):
        global first_city, second_city
        total_distance = 0
        for i in range(len(cities_id)):
            city1 = cities_id[i]
            city2 = cities_id[i] + 1 if cities_id[i] < len(cities_id) - 1 else 0
            for j in range(len(coordinate)):
                if int(city1 + 1) == coordinate[j][0]:
                    first_city = [coordinate[j][1], coordinate[j][2]]
                if int(city2 + 1) == coordinate[j][0]:
                    second_city = [coordinate[j][1], coordinate[j][2]]
            total_distance += self.get_distance(np.array(first_city), np.array(second_city))
        return total_distance


def show(lx, ly, city_position, dis):
    plt.cla()
    for i in range(len(city_position)):
        plt.scatter(city_position[i][1], city_position[i][2], color='r')
    plt.plot(lx, ly)
    plt.xticks([])
    plt.yticks([])
    plt.text(-5.25, -14.05, "Total distance=%.2f" % dis, fontdict={'size': 20, 'color': 'red'})
    plt.pause(0.01)


def best_show(x, y, city_position, best_fitness):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), facecolor='#ccddef')
    ax[0].set_title("Best route")
    ax[1].set_title("Fitness Change Procession")
    ax[0].plot(x, y)
    ax[0].scatter(city_position[0], city_position[1], color='r')
    ax[1].plot(range(len(best_fitness)), best_fitness)
    plt.show()


def init_pop(pop_size, EPO_size):
    pop = np.zeros((pop_size, EPO_size))
    cluster = np.arange(EPO_size)
    for i in range(pop_size):
        pop[i] = copy.deepcopy(cluster)
    return pop


def TSP(pop_size, EPO_size, coordinate, max_iteration):
    pop = init_pop(pop_size, EPO_size)
    EPO = EPO_TSP(pop, pop_size, EPO_size, 0, 1)
    for i in range(max_iteration):
        fitness = []
        for j in range(len(pop)):
            dis = EPO.compute_objective_value(EPO.pop[j], coordinate)
            fitness.append(dis)
        EPO.g_best(fitness=fitness)
        EPO.update(i, max_iteration)
        EPO.SPV()

        num = np.argmax(fitness)
        DNA = EPO.pop[num, :]
        print(f"The step is {i} ,the current best fitness is {EPO.g_fit}")
        lx = []
        ly = []
        for i in DNA:
            i = int(i)
            lx.append(coordinate[i][1])
            ly.append(coordinate[i][2])
        show(lx, ly, coordinate, max(fitness))


if __name__ == '__main__':
    data = tsplib95.load('./tsp/eil101.tsp')
    coords = np.array([(city, coord[0], coord[1]) for city, coord in data.node_coords.items()])
    pop_size = 80
    EPO_size = len(coords)
    t = 1000
    TSP(pop_size, EPO_size, coords, t)
