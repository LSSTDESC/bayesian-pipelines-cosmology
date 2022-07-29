import sys
sys.path.append("/Users/eleni/Desktop/pymaxent/")

import numpy as np 
from scipy.interpolate import interp1d

from pymaxent import *


def reconstruct_from_moments(p, x, N):
    
    bnd_low = min(x)
    bnd_high = max(x)

    moments = np.zeros(N)

    for i in range(N):
        moments[i] = np.sum(x**i*p)/((bnd_high-bnd_low)/len(x))**(-1)
    
    sol, lambdas = reconstruct(moments, bnds=[bnd_low, bnd_high])

    return x, sol(x)


def run(nzs, zs):
    sol, xs = [], []
    N_moments = [9, 9, 6, 5, 8]

    for i in range(len(nzs)):
        x_0, sol_0 = reconstruct_from_moments(nzs[i](zs), zs, N_moments[i])
        sol_0 = interp1d(x_0, sol_0, fill_value="extrapolate")
        xs.append(xs)
        sol.append(sol_0)
    return xs, sol

