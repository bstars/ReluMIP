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
		return zip(ws, bs)

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
	X = testset.data.detach().numpy()
	y = testset.targets.detach().numpy()

	return X, y

def random_select(net:FCNet, label, X, y):
	"""
	randomly select one image with a given label
	"""
	idx = np.where(y == label)[0]
	# np.random.shuffle(idx)
	for i in idx:
		x = np.reshape(X[i], [784,1])
		xtensor = torch.Tensor(x)

		yhat, _ = net(xtensor)
		yhat = yhat.detach().numpy()
		yhat = np.argmax(yhat)
		if yhat == label:
			return x / 255



def gradeint_attack(net : FCNet, img, j, attack_budget, max_iter=300000):
	"""
	:param net: FCNet
	:param img: np.array, [784,1]
	:param y: The true label of img
	:param j: Target label
	:return:
	"""
	assert attack_budget > 0
	img = torch.tensor(img, requires_grad=True).float()
	L = torch.clip(img - attack_budget, 0, 1)
	U = torch.clip(img + attack_budget, 0, 1)

	A = np.eye(10)
	A = np.delete(A, [j], axis=0)
	A[:, j] = -1
	A = torch.Tensor(A)
	gap_history = []


	for i in range(max_iter):
		score, _ = net(img)
		gaps = A @ score
		maxgap = torch.max(gaps)
		gap_history.append(maxgap.item())


		if maxgap < -1e-9:
			return  img, gap_history

		img.retain_grad()
		maxgap.backward()


		grad = img.grad

		gnorm = torch.sum((torch.square(grad)))
		stepsize = maxgap / gnorm
		stepsize = max(stepsize, 1e-5)
		with torch.no_grad():
			img -= stepsize * grad
			img = torch.clip(img, L, U)
			# img = torch.tensor(img, requires_grad=True)
			img = img.clone().detach().requires_grad_(True)



def compute_bound_lp(w, l, u, b):
	"""
	compute the optimization problems
		min_x.  <w,x> + b               max_x.  <w,x> + b
		s.t.    l <= x <= u             s.t.    l <= x <= u
	"""
	wpos = w >= 0
	wneg = w <= 0

	p1 = w[wpos] * l[wpos] + w[wneg] * u[wneg] + b
	p2 = w[wpos] * u[wpos] + w[wneg] * l[wneg] + b

	return p1, p2

#
# if __name__ == '__main__':
# 	net = FCNet.load_from_ckpt('./data/net500_1.pth')
# 	params = net.get_param_pair()
# 	y, zs = net(torch.zeros([784]))
# 	print(y.shape)




