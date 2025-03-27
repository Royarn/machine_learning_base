import torch
# PyTorch深度学习快速入门教程（绝对通俗易懂！）【小土堆
a = torch.ones(2, 3)
print(a.ndim)

b = torch.zeros(5, 6, dtype=torch.float)
print(b)
print(b.ndim)
print(b.shape)


c = torch.rand(2, 3)
print(c)

d = torch.rand(2, 3, dtype=torch.int)
print(d)

d = d.view(6)
print(d)
print(d[1].item())