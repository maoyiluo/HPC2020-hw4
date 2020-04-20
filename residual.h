#include <stdio.h>
#include "utils.h"
#include <cmath>

#define thread_num 4

double residual(double*u, double*f, double h, int N){
    double res = 0;
    double coef = 1.0/(h*h);
    for(int i = 1; i <= N; i++){
        for(int j = 1; j <= N; j++){
            double delta_u = coef*(-u[i*N + j-1] - u[(i-1)*N + j] + 4*u[i*N + j] - u[(i+1)*N + j] - u[i*N + j+1]);
            res += pow(delta_u - f[(i-1)*N + (j-1)],2);
        }
    }
    return sqrt(res);
}
