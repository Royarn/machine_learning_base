import torch

a = torch.rand(2, 3)
print(a)
print(a.ndim)

print(a.shape)


print(a.view(6))
print(a.max())
# max value index
print(a.argmax())
print(a.min())
# mini value index
print(a.argmin())
# squart
print(a.sqrt())