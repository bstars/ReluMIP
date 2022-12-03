import sys
sys.path.append("..")

import numpy as np
from optim.partition import PartitionNode

class BranchAndBound:
	def __init__(self, root:PartitionNode):
		self.root = root

	def partition_leaf(self, node : PartitionNode):
		"""
		Partition a leaf node and compute the lower bound and upper bound of subpartitions
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
		if p1_lb > node.ub or p1_lb > p2_ub: # p1 is pruned
			p1.pruned = True
			p1.lb = p1_lb
			p2.lb = max(p2_lb, node.lb)

		elif p2_lb > node.ub or p2_lb > p1_ub: # p2 is pruned
			p2.pruned = True
			p2.lb = p2_lb
			p1.lb = max(p1_lb, node.lb)

		else: # no child is pruned
			p1.lb = max(p1_lb, node.lb)
			p2.lb = max(p2_lb, node.lb)

		if np.abs(p1.lb - p1.ub) <= 1e-6: p1.pruned = True
		if np.abs(p2.lb - p2.ub) <= 1e-6: p2.pruned = True
		return p1, p2

	def select_partition(self, fringes):
		"""
		Select a partition from current leaf partitions with lowest lower bound,
		and return the two subpartitions of this partition

		:returns
			1. The node to partition and list of remaining partitions
			2. (None, None) if all leaf partitions are
				nonpartitionable or pruned
		"""

		idx_fringes = enumerate(fringes)
		idx_fringes = filter(
			lambda node : node[1].partitionable() and not node[1].pruned,
			idx_fringes
		)
		idx_fringes = list(idx_fringes)
		if len(idx_fringes) == 0:
			return None, None

		(idx, node) = min(idx_fringes, key=lambda x : x[1].lb)
		return node, fringes[:idx] + fringes[idx+1:]

	def solve(self, eps=1e-4):
		"""
		Solve the problem with branch and bound
		:returns
			the global optimal
			a feasible point that achieves the global optimal
		"""
		lb, ub, xub = self.root.compute_bounds()

		if lb == np.inf:
			return None, np.inf, [], []

		self.root.lb = lb; self.root.ub = ub; self.root.xub = xub

		fringes = [self.root]
		lb_history = []
		ub_history = []

		# for i in range(5):
		while True:

			lb_history.append(lb)
			ub_history.append(ub)
			print(lb, ub, len(fringes))

			node, new_fringe = self.select_partition(fringes)


			# the gap is guaranteed
			if np.abs(ub - lb) <= eps or node is None:
				return ub, xub, np.array(lb_history), np.array(ub_history)

			p1, p2 = self.partition_leaf(node)

			new_fringe = new_fringe + [p1, p2]

			lbs = [node.lb for node in new_fringe]
			ubs = [node.ub for node in new_fringe]

			lb = np.min(lbs)
			idx = np.argmin(ubs)


			ub = ubs[idx]
			xub = new_fringe[idx].xub

			fringes = new_fringe


