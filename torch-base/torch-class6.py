import torch

x = torch.ones((2, 2), requires_grad=True)

y = x + 2


z = y * y * 3

print(z.shape)
# average val
out = z.mean()
# grad calculation
print(out)
out.backward()
print(out.shape)

# print(x)
# print(y)
# print(z)
# out 对x 的梯度/求导，用公式来表达就是：out = z.mean()，所以out对x的梯度就是z的梯度除以z的个数，即z.grad/z.shape[0]
print(x.grad)
