import numpy as np 

def solve_jacob(A, b, n):
	x = np.zeros_like(b)
	d = np.diag(A)
	a = np.diagflat(d)

	for i in range(n):
		x = (1/d) * (b - np.dot(a, x))
	return x.tolist()