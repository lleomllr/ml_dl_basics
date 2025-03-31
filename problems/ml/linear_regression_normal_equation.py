import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	X_T = np.transpose(X)
	X_TX = np.dot(X_T, X)
	X_Ty = np.dot(X_T, y)

	theta = np.linalg.solve(X_TX, X_Ty)
	return theta