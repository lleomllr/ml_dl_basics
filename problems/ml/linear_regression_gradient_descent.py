import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iters: int) -> np.ndarray:
	m, n = X.shape
	theta = np.zeros((n, 1))
	y = y.reshape((m, 1))

	for i in range(iters):
		predictions = X @ theta
		err = prédictions - y
		gradient = (1/m) * (X.T @ err)
		theta -= alpha * gradient
	return theta