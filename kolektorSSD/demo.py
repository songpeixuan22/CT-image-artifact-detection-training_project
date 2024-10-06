import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from utils.net import UNet
from utils.dataloader import CustomDataset

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor()
])

# 加载数据集
data_dir = '.\\data'
dataset = CustomDataset(data_dir, transform=transform)

# 划分训练集和测试集
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 加载模型
model = UNet(in_channels=1, out_channels=1)
model.load_state_dict(torch.load('model.pth'))
model.to(device)
model.eval()

# 获取测试集的第20个样本
for i, (image, label) in enumerate(test_loader):
    if i == 19:  # 索引从0开始，所以第20个元素的索引是19
        image, label = image.to(device), label.to(device)
        break

# 模型处理图像
with torch.no_grad():
    output = model(image)

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
