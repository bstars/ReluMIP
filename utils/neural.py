import torch
from torch import nn

def get_param_pair(net:nn.Module):
	"""
	Get a list of (weight,bias) pair for a fully-connected neural net
	"""
	ws = []
	bs = []
	for name, param in net.named_parameters():
		if "W" in name: ws.append(param.detach().numpy())
		elif "b" in name: bs.append(param.detach().numpy()[:,0])

	return list(zip(ws, bs))



class FCNet(nn.Module):
	def __init__(self, fname):
		super(FCNet, self).__init__()
		ckpt = torch.load(fname)
		self.W0 = nn.Parameter(ckpt['W0'], requires_grad=False)
		self.b0 = nn.Parameter(ckpt['b0'][:,None], requires_grad=False)
		self.W1 = nn.Parameter(ckpt['W1'], requires_grad=False)
		self.b1 = nn.Parameter(ckpt['b1'][:,None], requires_grad=False)

	def forward(self, x):
		x = x.T
		x = torch.relu( torch.matmul(self.W0, x) + self.b0 )
		x = torch.matmul(self.W1, x) + self.b1
		return x



# net = FCNet("../data/net500_1.pth")
# y = net(torch.randn(2, 784))
# print(y.shape)
#
# get_param_pair(FCNet("../data/net500_1.pth"))


def load_net_1():
	net = FCNet("../data/net500_1.pth")
	return net

def load_net_2():
	net = FCNet("../data/net500_2.pth")
	return net



