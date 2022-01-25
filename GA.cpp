//
//  main.cpp
//  GA
//
//  Created by huangzihao on 2022/1/24.
//

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "math.h"
#include "time.h"

#define CITY_NUM 10
#define POPSIZE 30
#define MAXVALUE 10000000
#define N 100000

using namespace std;

unsigned seed = (unsigned)time(0);
double Hash[CITY_NUM + 1];
typedef struct CityPosition
{
    double x, y;
}CityPosition;

CityPosition CityPos[CITY_NUM] = {
    {178, 170}, {272, 395}, {176, 198}, {171, 151}, {650, 242}, {499, 556}, {276, 57}, {703, 401}, {408, 305}, {437, 421}
};

double CityDistance[CITY_NUM][CITY_NUM];

typedef struct {
    int colony[POPSIZE][CITY_NUM + 1];
    double fitness[POPSIZE];
    double Distance[POPSIZE];
    int BestRooting[CITY_NUM + 1];
    double BestFitness;
    double BestValue;
    int BestNum;
}TSP, * PTSP;

void CalculateDist() {
    double temp1, temp2;
    for (int i = 0; i < CITY_NUM; i++) {
        for (int j = 0; j <= CITY_NUM; j++) {
            temp1 = CityPos[j].x - CityPos[i].x;
            temp2 = CityPos[j].y - CityPos[i].y;
            CityDistance[i][j] = sqrt(temp1 * temp1 + temp2 * temp2);
        }
    }
}

void copy(int a[], int b[]) {
    for (int i = 0; i < CITY_NUM + 1; i++) {
        a[i] = b[i];
    }
}

bool check(TSP& city, int pop, int num, int k) {
    for (int i = 0; i <= num; i++) {
        if (k == city.colony[pop][i]) {
            return true;
        }
    }
    return false;
}

void InitColony(TSP& city) {
    int i ,j, r;
    for (i = 0; i < POPSIZE; i++) {
        city.colony[i][0] = 0;
        city.colony[i][CITY_NUM] = 0;
        city.BestValue = MAXVALUE;
        city.BestFitness = 0;
    }
    
    for (i = 0; i < POPSIZE; i++) {
        for (j = 1; j < CITY_NUM; j++) {
            r = rand() % (CITY_NUM - 1) + 1;
            while (check(city, i, j, r)) {
                r = rand() % (CITY_NUM - 1) + 1;
            }
            city.colony[i][j] = r;
        }
    }
}

void CalFitness(TSP& city) {
    int i, j;
    int start, end;
    int Best = 0;
    for (i = 0; i < POPSIZE; i++) {
        city.Distance[i] = 0;
        for (j = 1; j <= CITY_NUM; j++) {
            start = city.colony[i][j - 1];
            end = city.colony[i][j];
            city.Distance[i] = city.Distance[i] + CityDistance[start][end];
        }
        city.fitness[i] = N / city.Distance[i];
        if (city.fitness[i] > city.fitness[Best]) {
            Best = i;
        }
    }
    copy(city.BestRooting, city.colony[Best]);
    city.BestFitness = city.fitness[Best];
    city.BestValue = city.Distance[Best];
    city.BestNum = Best;
}

void Select(TSP& city) {
    int TempColony[POPSIZE][CITY_NUM + 1];
    int i ,t;
    double s;
    double GaiLv[POPSIZE];
    double SelectP[POPSIZE + 1];
    double sum = 0;
    for (i = 0; i < POPSIZE; i++) {
        sum += city.fitness[i];
    }
    for (i = 0; i < POPSIZE; i++) {
        GaiLv[i] = city.fitness[i] / sum;
    }
    SelectP[0] = 0;
    for (i = 0; i < POPSIZE; i++) {
        SelectP[i + 1] = SelectP[i] + GaiLv[i] * RAND_MAX;
    }
    memcpy(TempColony[0], city.colony[city.BestNum], sizeof(TempColony[0]));
    for (t = 1; t < POPSIZE; t++) {
        double ran = rand() % RAND_MAX + 1;
        s = (double)ran / 100.0;
        for (i = 1; i < POPSIZE; i++) {
            if (SelectP[i] >= s) {
                break;
            }
        }
        memcpy(TempColony[t], city.colony[i - 1], sizeof(TempColony[t]));
    }
    for (i = 0; i < POPSIZE; i++) {
        memcpy(city.colony[i], TempColony[i], sizeof(TempColony[i]));
    }
}

void Cross(TSP& city, double pc) {
    int i, j, t, l;
    int a, ca, cb;
    int Temp1[CITY_NUM + 1];
    for (i = 0; i < POPSIZE; i++) {
        double s = ((double)(rand() % RAND_MAX)) / RAND_MAX;
        if (s < pc) {
            cb = rand() % POPSIZE;
            ca = cb;
            if (ca == city.BestNum || cb == city.BestNum) {
                continue;
            }
            l = rand() % (CITY_NUM / 2) + 1;
            a = rand() % (CITY_NUM - l) + 1;
            
            memset(Hash, 0, sizeof(Hash));
            Temp1[0] = Temp1[CITY_NUM] = 0;
            for (j = 1; j <= l; j++) {
                Temp1[j] = city.colony[cb][a + j - 1];
                Hash[Temp1[j]] = 1;
            }
            for (t = 1; t < CITY_NUM; t++) {
                if (Hash[city.colony[ca][t]] == 0) {
                    Temp1[j++] = city.colony[ca][t];
                    Hash[city.colony[ca][t]] = 1;
                }
            }
            memcpy(city.colony[ca], Temp1, sizeof(Temp1));
        }
    }
}

double GetFitness(int a[CITY_NUM + 1]) {
    int i, start, end;
    double Distance = 0;
    for (i = 0; i < CITY_NUM; i++) {
        start = a[i];
        end = a[i + 1];
        Distance += CityDistance[start][end];
    }
    
    return N / Distance;
}

void Mutation(TSP& city, double pm) {
    int i, k, m;
    int Temp[CITY_NUM + 1];
    for (k = 0; k < POPSIZE; k++) {
        double s = ((double)(rand() % RAND_MAX)) / RAND_MAX;
        i = rand() % POPSIZE;
        if (s < pm && i != city.BestNum) {
            int a, b, t;
            a = (rand() % (CITY_NUM - 1)) + 1;
            b = (rand() % (CITY_NUM - 1)) + 1;
            copy(Temp, city.colony[i]);
            if (a > b) {
                t = a;
                a = b;
                b = t;
            }
            for (m = a; m < (a + b) / 2; m++) {
                t = Temp[m];
                Temp[m] = Temp[a + b - m];
                Temp[a + b - m] = t;
            }
            if (GetFitness(Temp) < GetFitness(city.colony[i])) {
                a = (rand() % (CITY_NUM - 1)) + 1;
                b = (rand() % (CITY_NUM - 1)) + 1;
                memcpy(Temp, city.colony[i], sizeof(Temp));
                if (a > b) {
                    t = a;
                    a = b;
                    b = t;
                }
                for (m = a; m < (a + b) / 2; m++) {
                    t = Temp[m];
                    Temp[m] = Temp[a + b - m];
                    Temp[a + b - m] = t;
                }
                if (GetFitness(Temp) < GetFitness(city.colony[i])) {
                    a = (rand() % (CITY_NUM - 1)) + 1;
                    b = (rand() % (CITY_NUM - 1)) + 1;
                    memcpy(Temp, city.colony[i], sizeof(Temp));
                    if (a > b) {
                        t = a;
                        a = b;
                        b = t;
                    }
                    for (m = a; m < (a + b) / 2; m++) {
                        t = Temp[m];
                        Temp[m] = Temp[a + b - m];
                        Temp[a + b - m] = t;
                    }
                }
            }
            memcpy(city.colony[i], Temp, sizeof(Temp));
        }
    }
}

void OutPut(TSP& city) {
    int i;
    cout << "Best TSP: " << endl;
    for (i = 0; i <= CITY_NUM; i++) {
        cout << city.BestRooting[i];
    }
    cout << endl << "Best Fitness: " << city.BestValue << endl;
}

int main(int argc, const char * argv[]) {
    TSP city;
    double pcross, pmutation;
    int MaxEpoch;
    srand(seed);
    MaxEpoch = 30;
    pcross = 0.5;
    pmutation = 0.2;
    CalculateDist();
    InitColony(city);
    CalFitness(city);
    
    for (int i = 0; i < MaxEpoch; i++) {
        Select(city);
        Cross(city, pcross);
        Mutation(city, pmutation);
        CalFitness(city);
        cout << "Iteration: "<< i << endl;
        OutPut(city);
    }
    return 0;
}
