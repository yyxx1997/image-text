import torch
import torch.nn as nn

device=torch.device('cuda')
a=torch.rand(3,4).to(device)
b=torch.rand(4,6).to(device)

c=torch.matmul(a,b)
print(c.device,torch.cuda.is_available(),torch.cuda.device_count(),torch.cuda.get_device_name())
