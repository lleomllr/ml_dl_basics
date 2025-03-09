"""
Write a Python function that multiplies a matrix by a scalar and returns the result.
"""

def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	res = [[i * scalar for i in row] for row in matrix]
	return res