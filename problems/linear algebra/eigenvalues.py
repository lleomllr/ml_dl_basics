"""
Write a Python function that calculates the eigenvalues of a 2x2 matrix. 
The function should return a list containing the eigenvalues, sort values from highest to lowest.

A = [[2, 1], [1, 2]]
det(A - lambda*I) = 0 <=> [[2 - lambda, 1], [1, 2 - lambda]] = 0
                      <=> (2 - lambda)(2 - lambda) - 1 = lambda**2 - 4*lambda + 3 = 0

discri = (-4)**2 - 4 * 1 * 3 = 4 > 0
=> x1 = 3 and x2 = 1
"""

import math 

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	a = matrix[0][0]
	b = matrix[0][1]
	c = matrix[1][0]
	d = matrix[1][1]

	trace = a + d
	det = a * d - b * c

	discriminant = trace**2 - 4 * det

	if discriminant < 0:
		raise ValueError("The matrix has complex eigenvalues.")
	
	sqrt_discri = math.sqrt(discriminant)
	x1 = (trace + sqrt_discri) / 2 
	x2 = (trace - sqrt_discri) / 2 
	eigenval = sorted([x1, x2], reverse = True)

	return eigenvalues