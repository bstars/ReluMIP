import sys

sys.path.append('..')

import numpy as np
import cvxpy
import matplotlib.pyplot as plt

from optim.partition import PartitionNode
from optim.brcbnd import BranchAndBound
from utils.config import Config

"""
Implement an example of branch-and-bound method to find the sparsest vector in polyhedron

		min_x.  card(x)
		s.t.    Ax <= b

which is equivalent to the problem

		min_{x,z}.  <1,z>
		s.t.        Li zi <= xi <= Ui zi
					Ax <= b
					zi \in {0,1}

where Li and Ui are lower and upper bounds for xi
"""


def element_bounds(A, b):
	"""
	Compute the element-wise lower and upper bound in polyhedron

	min_x.  xi                  max_x.  xi
	s.t.    Ax <= b             s.t.    Ax <= b
	"""
	m,n = A.shape
	L = []
	U = []
	for i in range(n):
		x = cvxpy.Variable([n, ])
		p1 = cvxpy.Problem(
			cvxpy.Minimize(x[i]),
			[A @ x <= b]
		)
		p1.solve(solver="Mosek")

		p2 = cvxpy.Problem(
			cvxpy.Maximize(x[i]),
			[A @ x <= b]
		)
		p2.solve()

		L.append(p1.value)
		U.append(p2.value)
	return np.array(L), np.array(U)


class SparsePartitionNode(PartitionNode):
	A = None
	b = None
	L = None
	U = None

	def __init__(self, part=None, xz_fea=None):
		super(SparsePartitionNode, self).__init__()
		A, b, L, U = SparsePartitionNode.A, \
		             SparsePartitionNode.b, \
		             SparsePartitionNode.L, \
		             SparsePartitionNode.U

		m, n = A.shape
		self.part = np.ones([n, ]) * 0.5 if part is None else part
		self.softpart = self.part.copy()
		self.ub = np.inf
		self.lb = -1
		self.xz_fea = None if xz_fea is None else xz_fea

		self.setted1 = self.part >= 0.7
		self.setted0 = self.part <= 0.3
		self.setted = np.logical_or(self.setted0, self.setted1)

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

		return SparsePartitionNode(p1), SparsePartitionNode(p2)

	def partitionable(self):
		return not np.all(self.setted)

	def compute_bounds(self):
		"""
		Compute the lower bound of this partition
		    min_{x,z}.  <1,z>
		    s.t.        Li zi <= x <= Ui zi
		                Ax <= b
		                0 <= zi <= 1
		and an upper bound
		    min_{x,z}.  <1,z>
		        s.t.        Li zi <= xi <= Ui zi
		                    Ax <= b
		                    zi \in {0,1}
		by rounding z* of lower bound problem and check the feasibility
		"""
		A, b, L, U = SparsePartitionNode.A, \
		             SparsePartitionNode.b, \
		             SparsePartitionNode.L, \
		             SparsePartitionNode.U

		m,n = A.shape

		""" 1. Compute the lower bound """
		zlb = cvxpy.Variable([n,])
		xlb = cvxpy.Variable([n,])
		constraints = [
			cvxpy.multiply(L, zlb) <= xlb, xlb <= cvxpy.multiply(U, zlb),
			A @ xlb <= b,
			0 <= zlb, zlb <= 1
		]

		if np.sum(self.setted0) > 0: constraints.append(zlb[self.setted0] == 0)
		if np.sum(self.setted1) > 0: constraints.append(zlb[self.setted1] == 1)

		p = cvxpy.Problem(
			cvxpy.Minimize(cvxpy.sum(zlb)),
			constraints=constraints
		)
		p.solve()

		if p.value == np.inf: # if the relaxed problem is infeasible, then the original is infeasible
			return np.inf, np.inf, None

		lb = p.value
		xlb = xlb.value; zlb = zlb.value
		zlb[self.setted0] = 0; zlb[self.setted1] = 1 # reset for precision
		self.softpart = zlb

		if np.all(
				np.abs( np.abs(zlb - 0.5) - 0.5 ) <= 1e-6
		): # the optimal of lower bound problem is feasible
			return lb, lb, (xlb, zlb)


		""" 2. Compute the upper bound """

		if self.xz_fea is not None:
			x, z = self.xz_fea
			return lb, np.sum(z), (x,z)

		xub = cvxpy.Variable([n,])
		constraints = [A @ xub <= b, ]
		if np.sum(self.setted0) > 0: constraints.append(xub[self.setted0] == 0)
		# if np.sum(self.setted1) > 0: constraints.append(xub[self.setted1] >= 1e-7)
		p = cvxpy.Problem(
			cvxpy.Minimize( cvxpy.norm( cvxpy.multiply( 1 - zlb, xub),1 ) ),
			constraints=constraints
		)
		p.solve()


		if p.value == np.inf:
			return lb, np.inf, None
		else:
			xub = xub.value
			zub = (np.abs(xub) >= 1e-10).astype(float)
			return lb, np.sum(zub), (xub, zub)

	def in_partition(self, x):
		x,z = x
		return np.all(
			np.abs(z[self.setted] - self.part[self.setted]) <= 1e-8
		)


def generate_sample(m,n):
	A = np.random.randn(m, n)
	x = np.random.randn(n)
	b = A @ x + 0.5

	L, U = element_bounds(A, b)

	while np.any(L == -np.inf) or np.any(U == np.inf):
		A = np.random.randn(m, n)
		x = np.random.randn(n)
		b = A @ x + 1

		L, U = element_bounds(A, b)
	return A, b, L, U, x, (np.abs(x) != 0).astype(float)

if __name__ == '__main__':
	m = 75
	n = 30

	A, b, L, U, x, z = generate_sample(m, n)
	xz_fea = (x, z)

	SparsePartitionNode.A = A
	SparsePartitionNode.b = b
	SparsePartitionNode.L = L
	SparsePartitionNode.U = U

	root = SparsePartitionNode(None, xz_fea)

	tree = BranchAndBound(root)
	ub, (x,z), lb_history, ub_history = tree.solve(eps=0.99)

	print(np.all(A @ x <= b), np.sum(z))
	plt.plot(lb_history, label="lower bound")
	plt.plot(np.ceil(lb_history), label="lower bound ceil")
	plt.plot(ub_history, label="upped bounds")
	plt.legend()
	plt.show()

	# arr = [1,3,4,2,9,6,0]
	# arr = enumerate(arr)
	# arr = list(filter(lambda x : x[1] > 4, arr))
	# print(arr)