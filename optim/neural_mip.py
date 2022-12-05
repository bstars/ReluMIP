"""
Convert a neural net to a set of inequality constraints
"""

import sys
sys.path.append('..')

import numpy as np
import cvxpy

from utils.neural import get_param_pair, FCNet
from utils.config import Config

def compute_bound(wbs, L, U):
	"""
	Compute the upper and lower bound of linear layers given the input range
	:param wbs: [(wi,bi),..), each pair is the (weight, bias) for a linear layer
	:param L: np.array [n1,], the lower bound of input
	:param U: np.array [n1,], the upper bound of input
	:return
		Ls, Us
			Li is the lower bound of activation of layer i
			Ui is the upper bound of activation of layer i
			(the range for input not included)
		setted:
			np.array of shape [n2 + n3 + ...n{-2}]
			if setted[i] = 1, then this relu unit is always active in the given input range
			if setted[i] = 0, then this relu unit is always dead in the given input range
			otherwise, the state of this relu unit vary within the input range
	"""
	print("Computing activation bounds ... ")
	Ls, Us = [L], [U]
	setted_shape = np.sum( [w.shape[0] for (w,b) in wbs[:-1]] ).astype(int)
	setted = np.ones([setted_shape, ]) * 0.5
	setted_start = 0

	for i,(w,b) in enumerate(wbs):
		m,n = w.shape
		lb, ub = [], []
		for j in range(m):
			x = cvxpy.Variable([n, ])
			pub = cvxpy.Problem(
				cvxpy.Maximize(w[j, :] @ x + b[j]),
				[Ls[-1] <= x, x <= Us[-1]]
			)
			pub.solve(solver=Config.cvxpy_solver)
			ub.append(pub.value)

			plb = cvxpy.Problem(
				cvxpy.Minimize(w[j, :] @ x + b[j]),
				[Ls[-1] <= x, x <= Us[-1]]
			)
			plb.solve(solver=Config.cvxpy_solver)
			lb.append(plb.value)

			if i < len(wbs)-1:
				if pub.value <= 0: setted[setted_start + j] = 0 # this unit is always dead
				if plb.value >= 0: setted[setted_start + j] = 1 # this unit is always active
		setted_start += m

		if i < len(wbs) - 1:
			""" Middle layers have relu activation"""
			Ls.append( np.maximum(np.array(lb), 0) )
			Us.append( np.maximum(np.array(ub), 0) )
		else:
			""" The final layer has no activation """
			Ls.append(np.array(lb))
			Us.append(np.array(ub))
	# print(setted_start, setted.shape)
	return Ls[1:], Us[1:], setted

def mip_constraints_from_nn(wbs, L, U):
	"""
	Given a fully-connected relu neural net, form the neural network as variables and constraints

		f^{-1} = W^{-1} f^{-2} + b^{-1}

		[
			f^{k} >= w^{k} f^{k-1} + b^{k}
			f^{k} >= 0
			f^{k} <= w^{k} f^{k-1} + b^{k} + (beta^{k} - 1) L^{K}
			f^{k} <= beta^{k} U^{k}
		] for all k = 2,3,...-2

		Note the binary constraints are not included


	:param wbs: [(wi,bi),..), each pair is the (weight, bias) for a linear layer
	:param L: The lower bound for input
	:param U: The upper bound for input

	:returns
		fs: A list of cvxpy variables representing outputs for each layer (including input)
		beta: A cvxpy variable containing all binary variables with shape [n2 + n3 + .. + n[-2]]
		constraints: A list of linear constraints for the neural net
		setted: np.array indicating which neuron is setted
			if setted[i] = 0, then this relu unit is always dead in the input range
			if setted[i] = 1, then this relu unit is always active in the input range
			if setted[i] = 0.5, then this relu is not setted
	"""

	Ls, Us, setted = compute_bound(wbs, L, U)

	print("Constructing relu constraints")
	beta_shape = np.sum([w.shape[0] for (w, b) in wbs[:-1]]).astype(int)

	f = cvxpy.Variable([wbs[0][0].shape[1]])

	fs = [f]  # a list of cvxpy variables, representing the output of a linear layer without activation
	beta = cvxpy.Variable([beta_shape, ])  # a cvxpy variable containing all binary variable, of shape [n2 + n3 + ... + n[-2]]
	constraints = []  # a list of convex constraints
	setted0 = setted < 0.3
	if np.sum(setted0) > 0: constraints.append( beta[setted0] == 0. )
	setted1 = setted > 0.7
	if np.sum(setted1) > 0: constraints.append(beta[setted1] == 1.)

	beta_start = 0
	for i, (w, b) in enumerate(wbs):
		m, n = w.shape
		fp = cvxpy.Variable([m, ])

		if i == len(wbs) - 1:
			constraints.append(fp == w @ f + b)
		else:

			constraints += [
				fp >= w @ f + b, fp >= 0,
				fp <= w @ f + b - cvxpy.multiply((beta[beta_start: beta_start + m] - 1.), Ls[i]),
				fp <= cvxpy.multiply(beta[beta_start: beta_start + m], Us[i])
			]
			beta_start += m

		fs.append(fp)
		f = fp
	return fs, beta, constraints, setted

if __name__ == '__main__':

	net = FCNet('../data/net500_1.pth')
	wbs = get_param_pair(net)

	Ls, Us, setted = compute_bound(wbs, np.ones([784]) * 0.6 , np.ones([784]) * 1)
	print(np.sum(
		np.abs(setted - 0.5) > 0.3
	))



