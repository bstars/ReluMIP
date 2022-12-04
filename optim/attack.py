import sys
sys.path.append('..')

import numpy as np
import cvxpy

from utils.neural import get_param_pair, FCNet
from utils.config import Config


def attack_obj_constraints(t, fs, j):
	"""

	Build the objective and the epigraph form of the attack optimization problem
		min.    t
		s.t.    f_k^{-1}(f
	@param t: A scalar cvxpy variable
	@param fs: A list of cvxpy variables
	@param j: The target label to attack
	@return:
	@rtype:
	"""
