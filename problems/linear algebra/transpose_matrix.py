"""
Write a Python function that computes the transpose of a given matrix
"""
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
	b = [[a[j][i] for j in range(len(a))] for i in range(len(a[0]))]
	return b