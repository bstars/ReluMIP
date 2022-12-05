import sys
sys.path.append('..')

import numpy as np
import cvxpy
import matplotlib.pyplot as plt

from utils.neural import get_param_pair, FCNet, load_net_1
from utils.config import Config
from optim.neural_mip import mip_constraints_from_nn, compute_bound
from optim.partition import PartitionNode
from optim.brcbnd import BranchAndBound


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
	return t, constraints

# def tighten_input(wbs, L, U, j):
# 	"""
#
# 	@param wbs: [(wi,bi),..), each pair is the (weight, bias) for a linear layer
# 	@param L: The lower bound for the input, of shape [Config.dim = 784]
# 	@param U: The upper bound for the input, of shape [Config.dim = 784]
# 	@param j: The target label to attack
# 	@return:
# 	"""
# 	fs, beta, nn_constraints, setted = mip_constraints_from_nn(wbs, L, U)
# 	obj, epi_constraints = attack_epi_form(fs, j)
#
# 	new_L = np.zeros(shape=[Config.dim])
# 	new_U = np.zeros(shape=[Config.dim])
#
# 	for i in range(Config.dim):
# 		while True:
# 			l, mid, u = L[i], (L[i] + U[i]) / 2, U[i]
#
# 			# solve the relaxed problem in two partitions of x
# 			U[i] = mid
# 			p1 = cvxpy.Problem(
# 				obj,
# 				constraints= nn_constraints + epi_constraints + []
# 			)




class AdvAttackNode(PartitionNode):
	net = None
	t = None
	fs = None
	beta = None
	constraints = None
	initialized = False
	"""
	The base problem is 
	
		min.    t
		s.t.    f_k^{-1} - f_j^{-1} <= t  forall k!=j
				f^{-1} = W^{-1} f^{-2} + b^{-1}
				[
					f^{k} >= w^{k} f^{k-1} + b^{k}
					f^{k} >= 0
					f^{k} <= w^{k} f^{k-1} + b^{k} + (beta^{k} - 1) L^{K}
					f^{k} <= beta^{k} U^{k}
				] for all k = 2,3,...-2
				
	we add constraints beta \in [0,1] to solve the lower bound problem, etc			
	"""

	@staticmethod
	def initialize(net, origin_img, target_label, attack_budget):
		assert attack_budget > 0
		AdvAttackNode.net = net
		wbs = get_param_pair(net)
		L = np.maximum(0, origin_img - attack_budget)
		U = np.minimum(1, origin_img + attack_budget)

		fs, beta, nn_constraints, setted = mip_constraints_from_nn(
			wbs,
			L,
			U
		)
		t, epi_constraint = attack_epi_form(fs, target_label)

		AdvAttackNode.t = t
		AdvAttackNode.fs = fs
		AdvAttackNode.beta = beta
		AdvAttackNode.constraints = nn_constraints + epi_constraint + [L <= AdvAttackNode.fs[0], AdvAttackNode.fs[0] <= U]
		AdvAttackNode.initialized = True


		return AdvAttackNode(setted, None)

	def __init__(self, part, fs_beta_t_fea=None):
		"""
		@param part:
		@param fs_beta_fea:
		"""
		super(AdvAttackNode, self).__init__()

		if not AdvAttackNode.initialized: raise "You must initialize before using"

		self.part = part if part is not None else np.ones([AdvAttackNode.beta.shape[0]]) * 0.5
		self.softpart = self.part.copy()
		self.ub = np.inf
		self.lb = -np.inf
		self.fs_beta_t_fea = None if fs_beta_t_fea is None else fs_beta_t_fea

		self.setted1 = self.part >= 0.7 # relax to 0.7 for numerical precision
		self.setted0 = self.part <= 0.3 # relax to 0.7 for numerical precision
		self.setted = np.logical_or(self.setted0, self.setted1)

		self.setted0_ext = np.sum(self.setted0) > 0 # any of binary variable set to 0
		self.setted1_ext = np.sum(self.setted1) > 0 # any of binary variable set to 1

	def partitionable(self):
		return not np.all(self.setted)

	def partition(self):
		"""
				Always find the least ambivalent element to further partition
				"""

		# find the least ambivalent unsetted element
		pref = np.abs(self.softpart - 0.5)
		pref[self.setted] = np.inf
		idx = np.argmin(pref)

		p1 = self.part.copy()
		p1[idx] = 0.
		p2 = self.part.copy()
		p2[idx] = 1.

		return AdvAttackNode(p1), AdvAttackNode(p2)

	def compute_bounds(self):
		"""
		Compute the lower bound of this partition
			[ Base problem ]
			[ 0 <= beta <= 1 ] for unsetted beta
			[ beta = 0 ] for beta setted as 0
			[ beta = 1 ] for beta setted as 1

		and an upper bound
			[ Base problem ]
			[ beta \in {0,1} ]
		by rounding z^* from the lower bound problem and check the feasibility
		"""

		""" Solve the lower bound problem """
		constraints = self.constraints
		if np.sum(self.setted0) > 0: constraints.append(self.beta[self.setted0] == 0)
		if np.sum(self.setted1) > 0: constraints.append(self.beta[self.setted0] == 1)
		constraints.append( 0 <= self.beta )
		constraints.append( self.beta <= 1)

		plb = cvxpy.Problem(
			cvxpy.Minimize(self.t),
			constraints=constraints
		)
		plb.solve(solver=Config.cvxpy_solver)

		if plb.value == np.inf: # if the relaxed problem is infeasible, then the original is infeasible
			return np.inf, np.inf, None

		lb = plb.value
		beta = self.beta.value
		self.softpart = beta
		if np.all(
			np.abs( np.abs(beta - 0.5) - 0.5 ) <= 1e-6
		): # if the lower bound solution is feasible for this partition
			fs = [f.value for f in self.fs]
			t = self.t.value
			return lb, lb, (fs, np.round(beta), t)

		""" Solve the upper bound problem """
		if self.fs_beta_t_fea is not None:
			(fs, beta, t) = self.fs_beta_t_fea
			return lb, t, self.fs_beta_t_fea

		constraints = self.constraints + [ self.beta == beta ]
		pub = cvxpy.Problem(
			cvxpy.Minimize(self.t),
			constraints=constraints
		)
		pub.solve(solver=Config.cvxpy_solver)

		if pub.value == np.inf:
			return lb, np.inf, None
		else:
			fs = [f.value for f in self.fs]
			beta = self.beta.value
			t = self.t.value
			return lb, pub.value, (fs, beta, t)

	def in_partition(self, fs_beta_t):
		(fs, beta, t) = fs_beta_t
		return np.all(
			np.abs(beta[self.setted] - self.part[self.setted]) <= 1e-7
		)


if __name__ == '__main__':
	net = load_net_1()
	ori_img = np.random.uniform(0, 1, 784)
	root = AdvAttackNode.initialize(net, ori_img, 7, 0.7)
	bnb = BranchAndBound(root)
	ub, fs_beta_t, lb_history, ub_history = bnb.solve()

	if fs_beta_t is None:
		print("infeasible")
	else:
		(fs, _, _) = fs_beta_t
		img = np.reshape(fs[0], [28, 28])
		fig, (ax0, ax1) = plt.subplots(1, 2)
		ax0.imshow(np.reshape(ori_img, [28,28]), cmap='gray')
		ax1.imshow(img, cmap='gray')
		plt.show()





