import sys
sys.path.append('..')

import numpy as np
import cvxpy

from utils.neural import get_param_pair, FCNet
from utils.config import Config
from optim.neural_mip import mip_constraints_from_nn


def attack_epi_form(fs, j):
	"""

	Build the objective and the epigraph form of the attack optimization problem
		min.    t
		s.t.    f_k^{-1} - f_j^{-1} <= t  forall k!=j
				other constraints
	@param t: A scalar cvxpy variable
	@param fs: A list of cvxpy variables
	@param j: The target label to attack
	@return:
	@rtype:
	"""
	A = np.eye(Config.n_class)
	A = np.delete(A, [j], axis=0)
	A[:,j] = -1
	t = cvxpy.Variable()
	constraints = [A @ fs[-1] <= t]
	return cvxpy.Minimize(t), constraints

def tighten_input(wbs, L, U, j):
	"""

	@param wbs: [(wi,bi),..), each pair is the (weight, bias) for a linear layer
	@param L: The lower bound for the input, of shape [Config.dim = 784]
	@param U: The upper bound for the input, of shape [Config.dim = 784]
	@param j: The target label to attack
	@return:
	"""
	fs, beta, nn_constraints, setted = mip_constraints_from_nn(wbs, L, U)
	obj, epi_constraints = attack_epi_form(fs, j)

	new_L = np.zeros(shape=[Config.dim])
	new_U = np.zeros(shape=[Config.dim])

	for i in range(Config.dim):
		while True:
			l, mid, u = L[i], (L[i] + U[i]) / 2, U[i]

			# solve the relaxed problem in two partitions of x
			U[i] = mid
			p1 = cvxpy.Problem(
				obj,
				constraints= nn_constraints + epi_constraints + []
			)








A = np.eye(3)
print(np.delete(A, [0], axis=0))