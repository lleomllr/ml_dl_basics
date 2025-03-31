"""
Write a Python function to calculate the covariance matrix for a given set of vectors. 
The function should take a list of lists, where each inner list represents a feature with its observations, 
and return a covariance matrix as a list of lists. Additionally, 
provide test cases to verify the correctness of your implementation.

The covariance between the two features is calculated based on their deviations from the mean. 
For the given vectors, both covariances are 1.0, resulting in a symmetric covariance matrix.
"""

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	n_features = len(vectors)
	n_obs = len(vectors[0])

	means = [sum(feature) / n_obs for feature in vectors]

	cov_mat = [[0.0 for _ in range(n_features)] for _ in range(n_features)]

	for i in range(n_features):
		for j in range(n_features):
			cov = sum((vectors[i][k] - means[i]) * (vectors[j][k] - means[j]) for k in range(n_obs)) / (n_obs - 1)
			cov_mat[i][j] = cov
	return cov_mat