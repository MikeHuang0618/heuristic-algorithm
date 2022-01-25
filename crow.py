# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import random
import numpy
import math
import matplotlib.pyplot as plt


class crow(object):
    def __init__(self):
        self.flock_size = 20
        self.iter_max = 2000
        self.flight_length = 2
        self.AP = 0.1
        self.x = []
        self.fit = []
        self.new_fit = numpy.array([0])
        self.iter_fit = numpy.array([])
        # initialize positions and memories
        self.d = 10
        self.l = -100
        self.u = 100
        for i in range(self.flock_size):
            self.x.append([])
            for j in range(self.d):
                self.x[i].append(self.l - ((self.l - self.u) * random.random()))
        self.x = numpy.matrix(self.x)
        # Evaluate fitness function
        for i in range(len(self.x)):
            self.fit.append(0)
            for j in range(self.d):
                self.fit[i] = self.fit[i] + pow(self.x[(i, j)], 2)

    def fitness(self):
        # Evaluate fitness function
        fit = []
        for i in range(len(self.new_x)):
            fit.append(0)
            for j in range(self.d):
                fit[i] = fit[i] + pow(self.new_x[(i, j)], 2)
        return fit

    def __call__(self, *args, **kwargs):
        self.memory_x = self.x.copy()
        self.fit = numpy.array(self.fit)
        self.memory_fit = self.fit.copy()
        for epoch in range(self.iter_max):
            self.new_x = numpy.empty((self.flock_size, self.d))
            for i in range(self.flock_size):
                rfn_num = int(math.ceil(self.flock_size - (self.flock_size - i) * random.random()) - 1)
                if random.random() > self.AP:
                    for j in range(self.d):
                        self.new_x[(i, j)] = self.x[(i, j)] + (random.random() * self.flight_length * (
                                self.memory_x[(rfn_num, j)] - self.x[(i, j)]))
                else:
                    for j in range(self.d):
                        self.new_x[(i, j)] = self.l - ((self.l - self.u) * random.random())
            self.fit = numpy.array(self.fitness())

            for i in range(self.flock_size):
                ctd = 0
                if self.l < self.new_x.all() < self.u:
                    ctd += 1
                    self.x[i] = self.new_x[i].copy()
                    if self.fit[i] < self.memory_fit[i]:
                        self.memory_x[i] = self.new_x[i].copy()
                        self.memory_fit[i] = self.fit[i]
            self.iter_fit = numpy.append(self.iter_fit, numpy.amin(self.memory_fit))
        print("Minimum fitness: %f" % min(self.iter_fit))

        x = numpy.arange(0, self.iter_max)
        y = self.iter_fit[x]
        plt.title("Crow Search Algorithm \n" +
                  "d: {0} n: {1} fl: {2} ap: {3} l: {4} u: {5}".format(self.d, self.flock_size, self.flight_length,
                                                                       self.AP, self.l, self.u))
        plt.ylabel("y axis fitness")
        plt.xlabel("x axis iteration")
        plt.plot(x, y)
        plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = crow()
    model()
