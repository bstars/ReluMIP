import cvxpy

class Config:
	cvxpy_solver = cvxpy.MOSEK
	dim = 784
	n_class = 10