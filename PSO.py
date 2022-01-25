import numpy as np
import matplotlib.pyplot as plt


def Sphere(x):
    z = np.sum(np.square(x))

    return z


d = 5
xMin, xMax = -1, 1
vMin, vMax = -0.2 * (xMax - xMin), 0.2 * (xMax - xMin)
MaxIt = 100
ps = 30
c1 = 2
c2 = 2
w = 0.8


def limitV(V):
    for i in range(len(V)):
        if V[i] > vMax:
            V[i] = vMax
        if V[i] < vMin:
            V[i] = vMin

    return V


def limitX(X):
    for i in range(len(X)):
        if X[i] > xMax:
            X[i] = xMax
        if X[i] < xMin:
            X[i] = xMin

    return X


def Optimization():
    class Particle:
        def __init__(self):
            self.position = np.random.uniform(xMin, 50, [ps, d])
            self.velocity = np.random.uniform(vMin, vMax, [ps, d])
            self.cost = np.zeros(ps)
            self.cost[:] = Sphere(self.position[:])
            self.pbest = np.copy(self.position)
            self.pbest_cost = np.copy(self.cost)
            self.index = np.argmin(self.pbest_cost)
            self.gbest = self.pbest[self.index]
            self.gbest_cost = self.pbest_cost[self.index]
            self.BestCost = np.zeros(MaxIt)
            self.fitness = []

        def Evaluate(self):
            for it in range(MaxIt):
                for i in range(ps):
                    self.velocity[i] = (w * self.velocity[i]
                                        + c1 * np.random.rand(1, d) * (self.pbest[i] - self.position[i])
                                        + c2 * np.random.rand(1, d) * (self.gbest - self.position[i]))
                    self.velocity[i] = limitV(self.velocity[i])
                    self.position[i] = self.position[i] * self.velocity[i]
                    self.position[i] = limitX(self.position[i])
                    self.cost[i] = Sphere(self.position[i])
                    if self.cost[i] < self.pbest_cost[i]:
                        self.pbest[i] = self.position[i]
                        self.pbest_cost[i] = self.cost[i]
                        if self.pbest_cost[i] < self.gbest_cost and self.pbest_cost[i] != 0.0:
                            self.gbest = self.position[i]
                            self.gbest_cost = self.pbest_cost[i]
                self.fitness.append(self.gbest_cost)
                self.BestCost[it] = self.gbest_cost

        def Plot(self):
            plt.ylabel('Best Function Value')
            plt.xlabel('Number of Iteration')
            plt.title('Particle Swarm Optimization of Sphere Function')
            t = np.array([t for t in range(0, MaxIt)])
            plt.plot(t, self.fitness, color='b', linewidth=1)
            plt.savefig('pso.jpg')
            plt.show()
            print('Best fitness value =', self.gbest_cost)

    a = Particle()
    a.Evaluate()
    a.Plot()


if __name__ == "__main__":
    Optimization()
