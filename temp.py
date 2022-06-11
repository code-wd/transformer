import torch

my_tensor = torch.rand(4, 3)
print(my_tensor)

mean_tensor = torch.mean(my_tensor, 1, keepdim=True)
print(mean_tensor)
