import sys
sys.path.append("/Users/eleni/Desktop/desc/qp")

from scipy.interpolate import interp1d

import qp

"""
Based on: https://github.com/aimalz/qp/blob/ac1fdaa6cdd6da5f168db1d097ee92379939b87d/docs/notebooks/demo.ipynb
"""


def quantile(nzs, zs):
	n_pdfs = len(nzs)
	E0 = qp.Ensemble(n_pdfs, gridded=(zs, nzs), vb=False)
	samples = E0.sample(1000)
	
	sol, xs = [], []
	for i in range(n_pdfs):	
		P = qp.PDF(samples=samples[i], limits=(0, 2.5), vb=False)
		P.plot()
		#quantiles = P.quantize(N=10)
		#x_0, sol_0 = P.approximate(zs, using=quantiles)
		x_0, sol_0 = P.approximate(zs, using=None, vb=False)
		xs.append(x_0)
		sol.append(sol_0)
	return xs, sol


def run(nzs, zs):	
	xs, sol_f = quantile(nzs, zs)

	sol = []
	for i in range(len(nzs)):		
		# We use scipy such that the interpolation behaves better with the forward model.
		sol_0 = interp1d(xs[i], sol_f[i], fill_value = "extrapolate")
		sol.append(sol_0)
	return xs, sol
