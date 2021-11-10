# import torch as t
# import numpy as np

# data = [[1, 2], [3, 4]]
# x_data = t.tensor(data)

# print(data)
# print(x_data)

# np_array = np.array(data)
# x_np = t.from_numpy(np_array)

# retains properties of x_data
# x_ones = t.ones_like(x_data)
# print(x_data)
# print(f"Ones Tensor: \n {x_ones} \n")
# # overrides the datatype of x_data
# x_rand = torch.rand_like(x_data, dtype=torch.float)
# print(f"Random Tensor: \n {x_rand} \n")

# In the functions below, it determines the dimensionality of the output tensor.
# shape = (17, 8)
# rand_tensor = torch.rand(shape)
# ones_tensor = torch.ones(shape)
# zeros_tensor = torch.zeros(shape)
#
# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")

# tensor = torch.rand(3, 4)
#
# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")

# We move our tensor to the GPU if available
# if torch.cuda.is_available():
#   tensor = tensor.to('cuda')

# tensor = torch.ones(4, 4)
# tensor[:, 1] = 0
# print(tensor)
#
# t1 = torch.cat([tensor, tensor, tensor], dim=1)
# print(t1)

# # This computes the element-wise product
# print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# # Alternative syntax:
# print(f"tensor * tensor \n {tensor * tensor}")

# print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# # Alternative syntax:
# print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# print(tensor, "\n")
# tensor.add_(5)
# print(tensor)

# t = torch.ones(5)
# print(f"t: {t}")
# n = t.numpy()
# print(f"n: {n}")

# n = np.ones(5)
# t = torch.from_numpy(n)
#
# np.add(n, 1, out=n)
# print(f"t: {t}")
# print(f"n: {n}")

# import torch, torchvision
# model = torchvision.models.resnet18(pretrained=True)
# data = torch.rand(1, 3, 64, 64)
# labels = torch.rand(1, 1000)
#
# # forward pass
# prediction = model(data)
#
# # backward pass
# loss = (prediction - labels).sum()
# loss.backward()
#
# optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
#
# #gradient descent
# optim.step()
#
# print("Prediction:", prediction)
# print("Y:", labels)
# print("Error:", loss)

# import torch

# a = torch.tensor([2., 3.], requires_grad=True)
# b = torch.tensor([6., 4.], requires_grad=True)
# # c = torch.tensor([10., 14.])
# # c = torch.tensor([10., 14.], requires_grad=False)
# Q = 3*a**3 - b**2
#
# external_grad = torch.tensor([1., 1.])
# Q.backward(gradient=external_grad)
#
# # check if collected gradients are correct
# print(9*a**2 == a.grad)
# print(-2*b == b.grad)
# # print(8*c == c.grad)

# import torch
#
# x = torch.rand((5, 5), requires_grad=False)
# y = torch.rand((5, 5), requires_grad=True)
# z = torch.rand((5, 5), requires_grad=True)
#
# a = x + y
# print(f"Does `a` require gradients? : {a.requires_grad}")
# b = x + z
# print(f"Does `b` require gradients?: {b.requires_grad}")

# from torch import nn, optim
# import torchvision
#
# model = torchvision.models.resnet18(pretrained=True)
#
# # Freeze all the parameters in the network
# for param in model.parameters():
#     param.requires_grad = False
#
# model.fc = nn.Linear(512, 10)
#
# # Optimize only the classifier
# optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        # flatten all dimensions except the batch dimension

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)
