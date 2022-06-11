import torch

my_tensor = torch.rand(4, 3)
mask = torch.triu(torch.ones(4, 3), diagonal=0)
mask = mask.bool()

print(my_tensor)
print(mask)

other = my_tensor.masked_fill(mask, -1e9)
my_tensor.masked_fill_(mask, -1e9)

print(other)
print(my_tensor)
