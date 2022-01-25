# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import random

def SA():
    length = 2
    list1 = [random.randint(1, 101) for _ in range(length)]
    T0 = 100
    temp_for_plot = T0
    run = 1000
    iter = 10
    alpha = 0.995
    T_temp = []
    ans = []
    new_x1_list = []
    print(list1)

    for i in range(run):
        for j in range(iter):
            new_obj = 0
            curr_obj = 0
            new_obj_multiple = 1
            curr_obj_multiple = 1

            for k in range(length):
                # add
                r_x_1 = random.uniform(-0.5, 0.5)
                new_x1 = list1[k] + r_x_1
                new_obj = new_obj + abs(new_x1)
                curr_obj = curr_obj + abs(list1[k])
                new_x1_list.append(new_x1)
                # multiple
                new_obj_multiple = new_obj_multiple * abs(new_x1)
                curr_obj_multiple = curr_obj_multiple * abs(list1[k])

            new_obj = new_obj + new_obj_multiple
            curr_obj = curr_obj + curr_obj_multiple

            if new_obj <= curr_obj:
                for k in range(length):
                    list1[k] = new_x1_list[k]
            else:
                r = np.random.rand()
                if r <= 1 / (np.exp((new_obj - curr_obj) / T0)):
                    for k in range(length):
                        list1[k] = new_x1_list[k]

            new_x1_list.clear()
        T_temp.append(T0)
        ans.append(curr_obj)

        T0 = alpha * T0

    print(ans)
    plt.plot(T_temp, ans)
    plt.xlabel("Temperature")
    plt.ylabel("Objective value")
    plt.xlim(temp_for_plot, 0)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    SA()
