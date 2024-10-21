import torch
from torch import nn

# 生成两个张量
tensor_ones = torch.ones((1, 1, 64, 128))
tensor_zeros = torch.zeros((1, 1, 64, 128))

# 定义均方误差损失函数
criterion = nn.MSELoss()

# 计算损失
loss = criterion(tensor_ones, tensor_zeros)
print(f'Loss: {loss.item()}')
