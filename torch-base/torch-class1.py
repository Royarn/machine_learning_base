import torch

a = torch.tensor([1, 2, 3], dtype=torch.int)
print(a)

b = torch.tensor([1, 2, 3], dtype=torch.float)
print(b)

c = torch.tensor([1, 2, 3], dtype=torch.float64)
print(c)
print(c.ndim)

d = torch.tensor([[[1, 2, 3]]], dtype=torch.float64, device='cpu')
print(d)
# data dimension
print(d.ndim)
# data shape
print(d.shape)