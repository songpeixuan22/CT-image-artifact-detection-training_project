import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from utils.network import U_Net, R2AttU_Net
from utils.dataloader import CustomDataset

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
])

# 加载数据集
data_dir = './data'
dataset = CustomDataset(data_dir, transform=transform)

test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 加载模型
model = U_Net(1,1)
model.load_state_dict(torch.load('model.pth'))
model.to(device)
model.eval()

# 定义损失函数
criterion = nn.MSELoss()

# 获取第i+1个样本
for i, (image, label) in enumerate(test_loader):
    if i == 327:  # 索引从0开始，所以第6个元素的索引是5
        image, label = image.to(device), label.to(device)
        print(f'Image shape: {image.shape}')
        print(f'Image path: {dataset.get_image_path(i)}')
        print(f'Label path: {dataset.get_label_path(i)}')
        break

# 模型处理图像
with torch.no_grad():
    output = model(image)

# 计算损失
loss = criterion(output, label)
print(f'Loss: {loss.item()}')

# 将张量转换为numpy数组
image_np = image.cpu().numpy().squeeze()
label_np = label.cpu().numpy().squeeze()
output_np = output.cpu().numpy().squeeze()

# 可视化原图、模型处理后的图像和标注
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(image_np, cmap='gray')
axs[0].set_title('Original Image')

axs[1].imshow(output_np, cmap='gray')
axs[1].set_title('Model Output')

axs[2].imshow(label_np, cmap='gray')
axs[2].set_title('Label')

for ax in axs:
    ax.axis('off')

plt.show()
