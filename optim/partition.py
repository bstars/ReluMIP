import numpy as np
import abc
from typing import Union
from copy import deepcopy

class PartitionNode:
	pass

class PartitionNode:
	def __init__(self):
		""" Variable indicating the partition. (e.g  [0,1] indicator for Mixed Integer Program) """
		self.part = None

		""" 
		Upper bound of the optimal in this partition 
		e.g Evaluating a feasible point
		"""
		self.ub = np.inf

		""" 
		Lower bound of the optimal in this partition
		e.g The optimal of a convex-relaxed problem
		"""
		self.lb = -np.inf


		"""
		A feasible point that achieves the upper bound
		If xub=None, ub=inf
		"""
		self.xub = None

		""" 
		A partition is pruned if 
			1. The lower bound in this partition exceeds some upper bound
			2. The partition is infeasible
			3. The optimal is found. In this case ub = lb
		"""
		self.pruned = False

	@abc.abstractmethod
	def partition(self) -> Union[PartitionNode, PartitionNode]:
		"""
		Further refine this partition and return a partition pair
		"""
		pass

	@abc.abstractmethod
	def compute_bounds(self):
		"""
		Compute the upper and lower bounds for this partition
		:return:
			lb : lower bound
				can be inf if the relaxed problem is infeasible,
				in this case, ub = inf, xub = none
			ub : upper bound
				can be inf if the problem is infeasible
			xub: if not None, xub is a feasible point in this partition
				if None, ub = inf
		"""
		pass

	@abc.abstractmethod
	def in_partition(self, x):
		"""
		Return whether point x is in this partition
		"""
		pass

	@abc.abstractmethod
	def partitionable(self):
		"""
		Return whether this partition can be further partitioned
		"""