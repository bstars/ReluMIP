import cvxpy

class Config:
	# cvxpy_solver = cvxpy.MOSEK
	cvxpy_solver = cvxpy.ECOS
	dim = 784
	n_class = 10