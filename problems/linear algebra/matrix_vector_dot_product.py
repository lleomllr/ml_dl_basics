"""
Write a Python function that computes the dot product of a matrix and a vector. 
The function should return a list representing the resulting vector if the operation is valid, 
or -1 if the matrix and vector dimensions are incompatible. 
A matrix (a list of lists) can be dotted with a vector (a list) only if the number of columns in the matrix equals the length of the vector. 
For example, an n x m matrix requires a vector of length m.
"""

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	if len(a) != len(b):
		return -1
	
	res = [] 

	for row in a: 
		if len(row) != len(b):
			return -1
		dot = sum(row[i] * b[i] for i in range(len(b)))
		res.append(dot)
	return res