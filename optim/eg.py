import cvxpy
import numpy as np

def dumb():
	x = cvxpy.Variable()
	constraints = [x <= 1]
	return x, constraints


t = cvxpy.Variable()
x, constraints = dumb()

problem = cvxpy.Problem(
	cvxpy.Maximize(t),
	constraints=constraints + [t <= x]
)

problem.solve()
print(problem.value)