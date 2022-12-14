

import numpy as np
import cvxpy
import torch

from impl2 import FCNet, neural_net_bounds, load, bounds_lp


class SplitNode():

	net = None # FCNet
	img = None # torch.Tensor, [784, 1]
	label = None #
	C = None
	CT = None # torch.Tensor version of C


	#
	num_sample = 30
	max_iters = 1000
	beta1 = 0.99
	beta2 = 0.999
	lr = 1e-3


	@staticmethod
	def initialize(net, img, label, budget):
		"""

		:param net: FCNet
		:param img: torch.Tensor, [784,1]
		:param label: True label of img
		:param budget: Attack budget
		:return:
		"""
		SplitNode.net = net
		SplitNode.label = label
		C = -np.eye(10)
		C = np.delete(C, [label], axis=0)
		C[:, label] = 1
		SplitNode.C = C
		SplitNode.CT = torch.Tensor(C)

		img = img.detach().numpy()[:,0]

		l = np.maximum(img - budget, 0)
		u = np.minimum(img + budget, 1)
		return SplitNode(l,u)

	def __init__(self, l, u):
		"""
		:param l: np.array, [784,], the lower bound of input
		:param u: np.array, [784,], the upper bound of input
		"""

		self.l = l
		self.u = u
		self.softpart = (l + u) / 2

		self.lb = -np.inf
		self.ub = np.inf
		self.xub = None
		self.pruned = False

	def construct_lb_problem(self):
		"""
		Construct the lower bound constraints with convex hull of relu
		:return:
		:rtype:
		"""
		Ls, Us = neural_net_bounds(
			SplitNode.net, self.l, self.u
		)
		xs = [cvxpy.Variable([784])]  # post-activations
		ys = []  # pre-activations
		constraints = [self.l <= xs[0], xs[0] <= self.u]  # constraints

		wbs = self.net.get_param_pair()
		for i, (w,b) in enumerate(wbs):
			m, n = w.shape
			l, u = Ls[i], Us[i]
			y = cvxpy.Variable([m, ])
			x = cvxpy.Variable([m, ])

			constraints.extend([
				y == w @ xs[-1] + b,
				l <= y, y <= u
			])
			if i < len(wbs) - 1:

				# case 1, upper bound is negative
				idx1 = u <= 0
				if np.sum(idx1) > 0: constraints.extend([
					x[idx1] == 0,
					l <= y, y <= u
				])

				# case 2, lower bound is positive
				idx2 = l >= 0
				if np.sum(idx2) > 0: constraints.extend([
					x[idx2] == y[idx2],
					l <= y, y <= u
				])

				# case3, lb <= 0 <= ub
				idx3 = np.logical_not(np.logical_or(idx1, idx2))
				if np.sum(idx3) > 0:
					temp = u[idx3] / (u[idx3] - l[idx3])
					constraints.extend([
						x[idx3] <= cvxpy.multiply(temp, y[idx3]) - cvxpy.multiply(temp, l[idx3]),
						x[idx3] >= 0,
						x[idx3] >= y[idx3]
					])
			xs.append(x)
			ys.append(y)
		return xs, ys, constraints

	def compute_lower_bound(self):
		"""
		Compute a lower bound by convex hull relaxation of relu
		:return:
			1. The lower bound
			2. The optimal input of the lower bound problem
		"""
		vals = []
		xs, ys, constraints = self.construct_lb_problem()
		C = SplitNode.C

		minval = np.inf
		xminval = None
		for i in range(C.shape[0]):
			problem = cvxpy.Problem(
				cvxpy.Minimize(C[i, :] @ ys[-1]),
				constraints=constraints
			)
			problem.solve(solver=cvxpy.MOSEK)
			# if problem.value == np.inf or problem.value == -np.inf:
			# 	raise "inf"
			# vals.append(problem.value)

			if problem.value <= minval:
				minval = problem.value
				xminval = xs[0].value
		return minval, xminval

	def compute_lower_bound_2(self):
		l, u = self.l, self.u
		Ls, Us = neural_net_bounds(SplitNode.net, l, u)
		l, u = Ls[-1], Us[-1]

		ls = []
		for i in range(SplitNode.C.shape[0]):
			ls.append(
				bounds_lp(SplitNode.C[i,:], 0, l, u)[0]
			)
		return np.min(ls)

	def compute_upper_bound(self, x = None):
		L = torch.Tensor(self.l)[:,None]
		U = torch.Tensor(self.u)[:, None]
		lr = SplitNode.lr
		beta1, beta2 = SplitNode.beta1, SplitNode.beta2

		if x is None:
			img = np.random.uniform(self.l, self.u)[:,None]
		else:
			img = x[:,None]

		img = torch.Tensor(img).detach().requires_grad_(True)
		first_mom = torch.zeros_like(img)
		second_mom = torch.zeros_like(img)

		gap_history = []
		for i in range(1, SplitNode.max_iters):
			score, _ = net(img)
			gaps = SplitNode.CT @ score
			maxgap = torch.min(gaps)
			gap_history.append(maxgap.item())

			if len(gap_history) > 10 and np.max(np.abs(np.diff(gap_history[-9:]))) <= 1e-4:
				return maxgap.item(), img.detach().numpy()[:, 0]

			yhat = torch.softmax(score, dim=0)[SplitNode.label]

			img.retain_grad()
			yhat.backward()
			# maxgap.backward()
			grad = img.grad

			with torch.no_grad():
				first_mom = beta1 * first_mom + (1 - beta1) * grad
				second_mom = beta2 * second_mom + (1 - beta2) * grad * grad
				first_unbias = 1 / (1 - beta1 ** i)
				second_unbias = 1 / (1 - beta2 ** i)

				# img = img - lr * (first_mom / first_unbias) / torch.sqrt(second_mom / second_unbias)
				img = img - lr * grad
				img = torch.minimum(img, U)
				img = torch.maximum(img, L)
				img = img.clone().detach().requires_grad_(True)

		score, _ = SplitNode.net(img)
		gaps = SplitNode.CT @ score
		maxgap = torch.min(gaps)
		return maxgap.item(), img.detach().numpy()[:,0]


	def compute_bounds(self):
		"""
		Compute a lower and upper bound
		:return:
			1. A lower bound in this partition
			2. An upper bound in this partition
			3. A feasible input x that achieves the upper bound
		"""

		# compute the lower bound
		lb, xlb = self.compute_lower_bound()
		# lb = self.compute_lower_bound_2()
		# compute the upper bound

		# ub = np.inf
		# for i in range(5):
		# 	ubp, xubp = self.compute_upper_bound()
		# 	if ubp < ub:
		# 		ub = ubp
		# 		xub = xubp
		ub, xub = self.compute_upper_bound()
		return lb, ub, xub

	def partitionable(self):
		gap = self.u - self.l
		return not np.all( np.abs(gap) <= 1e-4 )

	def partition(self):
		"""
		Always partition the variable with this largest gap so that we quickly reduce rectangle diameter
		:return:
		"""
		gap = self.u - self.l
		idx = np.argmax(gap)
		part = (self.u[idx] + self.l[idx]) / 2

		# pref = (self.softpart - self.l) / (self.u - self.l)
		# idx = np.argmin(pref)
		# part = self.softpart[idx]

		l1, l2 = self.l.copy(), self.l.copy()
		u1, u2 = self.u.copy(), self.u.copy()

		u1[idx] = part
		l2[idx] = part


		return SplitNode(l1, u1), SplitNode(l2, u2)

	def in_partition(self, x):
		"""
		Whether an input image x is in this partition
		:param x: np.array, [784,]
		"""
		return np.logical_and(np.all( self.l <= x), np.all(x <= self.u))


class BranchAndBound():
	def __init__(self, root : SplitNode, early_terminate=None, additional_prune=None):
		self.root = root
		self.early_terminate = early_terminate
		self.additional_prune = None

	def partition_leaf(self, node : SplitNode):
		"""
		Partition a leaf node in the branch-and-bound tree, and compute the lower and upper bound
		"""
		p1, p2 = node.partition()

		""" Solve for partition 1 and deal with the upper bound """
		p1_lb, p1_ub, x1_ub = p1.compute_bounds()

		if p1_ub > node.ub and p1.in_partition(node.xub):
			p1.ub = node.ub
			p1.xub = node.xub
		else:
			p1.ub = p1_ub
			p1.xub = x1_ub

		""" Solve for partition 2 and deal with the upper bound """
		p2_lb, p2_ub, x2_ub = p2.compute_bounds()

		if p2_ub > node.ub and p2.in_partition(node.xub):
			p2.ub = node.ub
			p2.xub = node.xub
		else:
			p2.ub = p2_ub
			p2.xub = x2_ub

		""" Deal with the lower bound """
		if p1_lb > node.ub or p1_lb > p2.ub: # p1 is pruned
			p1.pruned = True
			p1.lb = p1_lb
			p2.lb = max(p2_lb, node.lb)
		elif p2_lb > node.ub or p2_lb > p1.ub: # p2 is pruned
			p2.pruned = True
			p2.lb = p2_lb
			p1.lb = max(p1_lb, node.lb)
		else: # no child is pruned
			p1.lb = max(p1_lb, node.lb)
			p2.lb = max(p2_lb, node.lb)

		if np.abs(p1.ub - p1.lb) <= 1e-3: p1.pruned = True
		if np.abs(p2.ub - p2.lb) <= 1e-3: p2.pruned = True

		if self.additional_prune is not None:
			if self.additional_prune(p1.lb, p1.ub): p1.pruned = True
			if self.additional_prune(p2.lb, p2.ub): p2.pruned = True

		# print('\t %.3f, %.3f, %.3f, %.3f' % (p1.lb, p1.ub, p2.lb, p2.ub))
		return p1, p2

	def select_partition(self, fringes, iter=0):
		"""
		Select a node from current fringes with the lowest lower bound
		:return:
			1. The node to partition, the list of remaining nodes
			2. (None, None) if all nodes are non-partitionable or pruned
		"""
		idx_fringes = enumerate(fringes)
		idx_fringes = filter(
			lambda node : node[1].partitionable() and (not node[1].pruned),
			idx_fringes
		)
		# print(len(list(idx_fringes)))
		idx_fringes = list(idx_fringes)

		if len(idx_fringes) == 0:
			return None, None

		if iter % 5 == 0:
			(idx, node) = min(idx_fringes, key=lambda x: x[1].ub)
		else:
			(idx, node) = min(idx_fringes, key=lambda x: x[1].lb)


		return node, fringes[:idx] + fringes[idx+1:]

	def solve(self, eps=1e-3, maxiter=0):
		"""

		:param eps:
		:type eps:
		:return:
			1. The global lower bound
			2. The global upper bound
			3. A feasible point that achieves the global optimal
			4. history of lower bound
			5. history of upper bound
		"""

		print('Running branch and bound ')
		lb, ub, xub = self.root.compute_bounds()
		if lb == np.inf:
			print("Infeasible")
			return np.inf, np.inf, None, [], []
		self.root.lb = lb; self.root.ub = ub; self.root.xub = xub
		# return lb, ub, xub, [], []

		fringes = [self.root]
		lb_history = []
		ub_history = []

		num_iter = 0
		while True:
			print('Branch and bound iteration %d, lower bound %.6f, upper bound %.6f' % (num_iter, lb, ub))
			num_iter += 1
			lb_history.append(lb)
			ub_history.append(ub)
			# print(lb, ub, len(fringes))


			node, new_fringe = self.select_partition(fringes, num_iter)
			if num_iter >= maxiter:
				return None, None, None, None, None
			if (np.abs(ub - lb) <= eps) or (node is None):
				return lb, ub, xub, np.array(lb_history), np.array(ub_history)

			if (self.early_terminate is not None) and self.early_terminate(lb, ub):
				return lb, ub, xub, np.array(lb_history), np.array(ub_history)

			p1, p2 = self.partition_leaf(node)
			new_fringe = new_fringe + [p1, p2]

			lbs = [node.lb for node in new_fringe]
			ubs = [node.ub for node in new_fringe]

			lb = np.min(lbs)

			idx = np.argmin(ubs)
			ub = ubs[idx]
			xub = new_fringe[idx].xub

			fringes = new_fringe






# [497, 553, 610]
# 484 571

if __name__ == '__main__':
	X, y = load()

	net = FCNet.load_from_ckpt('./data/net500_2.pth')

	# x = torch.randn(784, 1)
	# y,_ = net(x)

	num_certified = 0
	num_falsitied = 0
	num_all = 0
	# for idx in range(631, 1000):
	for idx in [497, 553, 610]:
		num_all += 1
		print('--------------------------------------------------------')
		img = X[idx].reshape([784, 1])
		root = SplitNode.initialize(net, img, y[idx], 0.05)


		bnb = BranchAndBound(root, early_terminate=lambda lb, ub:  (lb > 0) or (ub < 0))
		lb, ub, xub, lb_history, ub_history = bnb.solve(maxiter=50)

		if lb is None:
			print('Branch and bound takes too long on data %d' % idx)

		elif lb > 0:
			print('data %d certified' % idx)
			num_certified += 1
		elif ub < 0:
			xub = torch.Tensor(xub)[:, None]
			scores, _ = net(xub)
			scores = scores.detach().numpy()
			yhat = np.argmax(scores)
			print("data %d attacked to %d from %d" % (idx, yhat, y[idx]))
			num_falsitied += 1

		print("%.4f %% certified, %.4f %% falsified, %.4f %% uncertain" % (num_certified / num_all * 100, num_falsitied / num_all * 100, (num_all - num_certified - num_falsitied) / num_all * 100))
		print('--------------------------------------------------------')


	# val, x = root.compute_upper_bound()
	# print(val, type(x), x.shape)









