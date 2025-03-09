"""
Write a Python function that calculates the mean of a matrix either by row or by column, based on a given mode. 
The function should take a matrix (list of lists) and a mode ('row' or 'column') 
as input and return a list of means according to the specified mode.
"""

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	if mode == "row":
		means = [sum(row) / len(row) for row in matrix]
	elif mode == "column":
		transposed = list(zip(*matrix))
		means = [sum(col) / len(col) for col in transposed]
	else: 
		means = []
	return means