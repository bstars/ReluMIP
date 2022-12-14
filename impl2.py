import matplotlib.pyplot as plt
import numpy as np
import cvxpy
import torch
from torch import nn
import torchvision


class FCNet(nn.Module):
	@staticmethod
	def load_from_ckpt(fname):
		ckpt = torch.load(fname)
		ws = []
		bs = []
		for k in ckpt.keys():
			if 'W' in k: ws.append(torch.Tensor(ckpt[k]))
			if 'b' in k: bs.append(torch.Tensor(ckpt[k]))

		return FCNet(zip(ws, bs))


	def __init__(self, wbs):
		super(FCNet, self).__init__()
		self.ws = []
		self.bs = []
		for (w,b) in wbs:
			self.ws.append(w)
			self.bs.append(b)

	def get_param_pair(self):
		ws = []
		bs = []
		for w, b in zip(self.ws, self.bs):
			ws.append(w.detach().numpy())
			bs.append(b.detach().numpy())
		return list(zip(ws, bs))

	def forward(self, x):
		"""

		:param x: torch.Tensor, [dim, batch_size] for [dim]
		:type x:
		:return:
		:rtype:
		"""
		zs = [x] # pre-activation values
		for i in range(len(self.ws)-1):
			# w,b = self.ws[i], self.bs[i]
			x = torch.matmul(self.ws[i], x) + self.bs[i][:,None]
			zs.append(x)
			x = torch.relu(x)
		x = torch.matmul(self.ws[-1], x) + self.bs[-1][:,None]
		return x, zs

def load():
	# transform = transforms.ToTensor()
	testset = torchvision.datasets.MNIST("./data/mnist", train=False, download=True)
	X = testset.data
	y = testset.targets

	return X/255, y

def plot(img, label):
	plt.imshow(np.reshape(img, [28,28]), label="%d"%label, cmap='gray')
	plt.title("label=%d"%label)
	plt.show()

def random_other_label(i):
	while True:
		j = np.random.randint(0, 10)
		if j != i:
			return j

def other_label(i):
	ldict = {
		0:6,
		1:7,
		2:7,
		3:8,
		4:6,
		5:3,
		6:8,
		7:9,
		8:3,
		9:7
	}
	return ldict[i]

def random_select(net:FCNet, X, y, num):
	idx = np.arange(len(X))
	np.random.shuffle(idx)
	counter = 0
	retx = []
	rety = []
	for i in idx:
		x = torch.reshape(X[i], [784,1])
		yhat, _ = net(x)
		yhat = torch.argmax(yhat).item()

		if yhat == y[i].item():
			retx.append(x)
			rety.append(y[i].item())
			counter += 1
			if counter == num:
				return retx, rety

def gradient_attack(net:FCNet, img, i, j, attack_budget, max_iter=5000, lr=5e-3, beta1=0.9, beta2=0.99):
	"""
	:param net:FCNet
	:param img: np.array, [784, 1]
	:param i: The true label
	:param j: The target label
	:param attack_budget:
	:param max_iter:
	:return:
	"""
	assert attack_budget > 0
	# L = torch.clip(img - attack_budget, 0, 1)
	# U = torch.clip(img + attack_budget, 0, 1)

	L = torch.maximum(img - attack_budget, torch.zeros_like(img))
	U = torch.minimum(img + attack_budget, torch.ones_like(img))

	img = img.clone().detach().requires_grad_(True)


	# A = np.eye(10)
	# A = np.delete(A, [j], axis=0)
	# A[:,j] = -1
	# A = torch.Tensor(A)
	c = torch.zeros([10,1])
	c[i] = 1; c[j] = -1
	gap_history = []

	first_mom = torch.zeros_like(img)
	second_mom = torch.zeros_like(img)


	for i in range(1, max_iter):
		score, _ = net(img)

		# maxgap = torch.max(gaps)
		maxgap = torch.sum(score * c)
		gap_history.append(maxgap.item())

		# print(maxgap)

		if maxgap < -1e-11:
			return img, gap_history

		img.retain_grad()
		maxgap.backward()
		grad = img.grad


		with torch.no_grad():
			first_mom = beta1 * first_mom + (1 - beta1) * grad
			second_mom = beta2 * second_mom + (1 - beta2) * grad * grad
			first_unbias = 1 / (1 - beta1 ** i)
			second_unbias = 1 / (1 - beta2 ** i)

			img = img - lr * (first_mom / first_unbias) / torch.sqrt(second_mom / second_unbias)
			img = torch.minimum(img, U)
			img = torch.maximum(img, L)
			img = img.clone().detach().requires_grad_(True)

	return None, gap_history

def bounds_lp(w, b, l, u):
	"""
	compute the optimization problems
		min_x.  <w,x> + b               max_x.  <w,x> + b
		s.t.    l <= x <= u             s.t.    l <= x <= u
	"""
	wpos = w >= 0
	wneg = w <= 0
	p1 = np.sum(w[wpos] * l[wpos]) + np.sum(w[wneg] * u[wneg]) + b
	p2 = np.sum(w[wpos] * u[wpos]) + np.sum(w[wneg] * l[wneg]) + b
	return p1, p2

def neural_net_bounds(net : FCNet, l, u):
	"""
	Compute all pre-activation bounds
	:param net:
	:param l:
	:param u:
	:return:
	"""
	Ls = []
	Us = []
	wbs = net.get_param_pair()
	for w, b in wbs:
		m,n = w.shape
		newl, newu = [], []
		for i in range(m):

			p1, p2 = bounds_lp(w[i,:], b[i], l, u)
			newl.append(p1)
			newu.append(p2)

		newl = np.array(newl)
		newu = np.array(newu)

		Ls.append(newl)
		Us.append(newu)

		l = np.maximum(newl, 0)
		u = np.maximum(newu, 0)
	return Ls, Us

def neural_net_gap_lower_bonuds(net:FCNet, x:torch.Tensor, y, budget):
	"""
	:param net:
	:param x: The original image, torch.Tensor, [784,1]
	:param y: The label of the original image
	:param budget:
	:return:
	"""
	x = x.detach().numpy()[:,0]
	C = -np.eye(10)
	C = np.delete(C, [y], axis=0)
	C[:,y] = 1

	l = np.maximum(x - budget, 0)
	u = np.minimum(x + budget, 1)
	Ls, Us = neural_net_bounds(net, l, u)
	l, u = Ls[-1], Us[-1]

	m,n = C.shape
	ls = []
	for i in range(m):
		ls.append(
			bounds_lp(C[i,:], 0, l, u)[0]
		)
	return np.array(ls)

def neural_net_relaxed(net:FCNet, x:torch.Tensor, budget):
	"""
	construct the relaxed constraints of
		s.t.    x_{-1} = w_{-1} x_{-2} + b{-1}
				x_i = relu(w_i x_{i-1} + b_i)  \forall i
				l <= x_0 <= u
	:param net:
	:param x: The original image, torch.Tensor, [784,1]
	:param y: The label of the original image
	:param budget:
	:return:
	"""
	x = x.detach().numpy()[:,0]
	l = np.maximum(x - budget, 0)
	u = np.minimum(x + budget, 1)
	Ls, Us = neural_net_bounds(net, l, u)

	# construct variables and constraints
	wbs = net.get_param_pair()

	xs = [cvxpy.Variable([784])] # post-activations
	ys = [] # pre-activations
	constraints = [l <= xs[0], xs[0] <= u] # constraints

	for i, (w,b) in enumerate(wbs):
		m,n = w.shape
		l, u = Ls[i], Us[i]
		y = cvxpy.Variable([m,])
		x = cvxpy.Variable([m,])

		constraints.extend([
			y == w @ xs[-1] + b,
			l <= y, y <= u
		])
		if i < len(wbs)-1:

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
			idx3 = np.logical_not( np.logical_or(idx1, idx2) )
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

def neural_net_gap_lower_bounds_relaxed(net, x:torch.Tensor, y, budget):
	"""
	:param net:
	:param x: The original image, torch.Tensor, [784,1]
	:param y: True label of x
	:param budget:
	:return:
	"""
	C = -np.eye(10)
	C = np.delete(C, [y], axis=0)
	C[:, y] = 1
	vals = []

	xs, ys, constraints = neural_net_relaxed(net, x, budget)
	for i in range(C.shape[0]):
		problem = cvxpy.Problem(
			cvxpy.Minimize( C[i,:] @ ys[-1] ),
			constraints=constraints
		)
		problem.solve(solver=cvxpy.MOSEK)
		if problem.value == np.inf or problem.value == -np.inf:
			raise "inf"
		vals.append(problem.value)
	return np.min(vals)



# Branch and Bound

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
			score, _ = SplitNode.net(img)
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

### Answer functions
def p9():
	X, y = load()
	net1 = FCNet.load_from_ckpt('./data/net500_1.pth')
	xs, ys = random_select(net1, X, y, num=5)

	for i in range(5):
		# target = random_other_label(ys[i])
		target = other_label(ys[i])
		for budget in [0.08, 0.05, 0.01]:
			img = xs[i].clone()
			xattack, _ = gradient_attack(
				net1, img, ys[i], target, budget
			)

			fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
			ax1.imshow(
				np.reshape(img.detach().numpy(), [28,28]),
				cmap='gray'
			)
			ax1.set_title('original image with label %d' % ys[i], fontsize=10)


			if xattack is not None:
				diff = torch.abs(img - xattack)
				scores, _ = net1(xattack)
				label_attack = torch.argmax(scores).item()
				ax2.imshow(
					np.reshape(xattack.detach().numpy(), [28, 28]),
					cmap='gray'
				)
				ax2.set_title("attack to class %d" % (label_attack), fontsize=10)
				ax3.imshow(
					np.reshape(diff.detach().numpy(), [28, 28]),
					cmap='gray'
				)
				print(torch.max(torch.abs(img - xattack)))
			else:
				ax2.imshow(np.zeros([28,28]), cmap='gray')
				ax2.set_title("attack failed", fontsize=10)
				ax3.imshow(np.zeros([28, 28]), cmap='gray')
			ax3.set_title('difference <= %.2f' % budget, fontsize=10)

			plt.show()

def p10():
	X, y = load()
	net1 = FCNet.load_from_ckpt('./data/net500_1.pth')
	net2 = FCNet.load_from_ckpt('./data/net500_2.pth')
	xs, ys = random_select(net1, X, y, num=5)

	for i in range(5):
		# target = random_other_label(ys[i])
		target = other_label(ys[i])
		for n,net in enumerate([net1, net2]):
			budget = 0.08
			img = xs[i].clone()
			xattack, gap_history = gradient_attack(
				net, img, ys[i], target, budget
			)

			fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14, 6))

			ax1.imshow(
				np.reshape(img.detach().numpy(), [28, 28]),
				cmap='gray'
			)
			ax1.set_title('original image with label %d' % ys[i], fontsize=10)

			if xattack is not None:
				diff = torch.abs(img - xattack) if xattack is not None else torch.zeros_like(img)
				xattack = xattack if xattack is not None else torch.zeros_like(img)
				scores, _ = net(xattack)
				label_attack = torch.argmax(scores).item()
				ax2.imshow(
					np.reshape(xattack.detach().numpy(), [28, 28]),
					cmap='gray'
				)
				ax2.set_title("attack to class %d" % (label_attack), fontsize=10)
				ax3.imshow(
					np.reshape(diff.detach().numpy(), [28, 28]),
					cmap='gray'
				)
			else:
				ax2.imshow(np.zeros([28, 28]), cmap='gray')
				ax2.set_title("attack failed", fontsize=10)
				ax3.imshow(np.zeros([28, 28]), cmap='gray')
			ax3.set_title('difference <= %.2f' % budget, fontsize=10)

			ax4.plot(gap_history)
			ax4.set_title('gap history')

			plt.title('net %d' % (n+1))
			plt.show()

def p19():
	import time
	X, y = load()
	net1 = FCNet.load_from_ckpt('./data/net500_1.pth')
	net2 = FCNet.load_from_ckpt('./data/net500_2.pth')

	for n, net in enumerate([net1, net2]):
		for budget in [0.01, 0.05, 0.08]:
			lbs = []

			tic = time.time()
			for i in range(500):
				x = torch.reshape(X[i], [784,1])
				lbs.append(
					np.min( neural_net_gap_lower_bonuds(net, x, y[i], budget) )
				)
			lbs = np.array(lbs)
			toc = time.time()
			certified = np.sum(lbs >= 0) / len(lbs)

			plt.hist(lbs)
			plt.title( " net%d, eps=%.3f,  %.2f%% certified, runtime %.3f s" % (n+1, budget, certified * 100, toc-tic) )
			plt.show()

def p24():
	import time
	budget = 0.05
	X, y = load()
	net1 = FCNet.load_from_ckpt('./data/net500_1.pth')
	net2 = FCNet.load_from_ckpt('./data/net500_2.pth')

	for n, net in enumerate([net1, net2]):
		lbs = []
		tic = time.time()
		for i in range(10):
			x = torch.reshape(X[i], [784, 1])
			lbs.append(
				neural_net_gap_lower_bounds_relaxed(net, x, y[i], budget)
			)
		lbs = np.array(lbs)
		toc = time.time()
		certified = np.sum(lbs >= 0) / len(lbs)

		plt.hist(lbs)
		plt.title(" net%d, eps=%.3f,  %.2f%% certified, runtime %.3f s" % (n + 1, budget, certified * 100, toc - tic))
		plt.show()

def p25_demo():
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

		bnb = BranchAndBound(root, early_terminate=lambda lb, ub: (lb > 0) or (ub < 0))
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


# X, y = load()
# net = FCNet.load_from_ckpt('./data/net500_1.pth')
#
# x = torch.reshape(X[0], [784,1])
# neural_net_gap_lower_bounds_relaxed(net, x, y[0], 0.05)




