import sys
sys.path.append("/Users/eleni/Desktop/pymaxent/")

import numpy as np 

from pymaxent import *

def reconstruct_from_moments(p, x, N):
    
    bnd_low = min(x)
    bnd_high = max(x)

    moments = np.zeros(N)

    for i in range(N):
        moments[i] = np.sum(x**i*p)/((bnd_high-bnd_low)/len(x))**(-1)
    
    sol, lambdas = reconstruct(moments, bnds=[bnd_low, bnd_high])

    return x, sol(x)