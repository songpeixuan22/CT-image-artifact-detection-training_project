import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.dataloader import CropDataset
from utils.processor import label_pos
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
])

data_dir = './data'
dataset = CropDataset(data_dir, transform=transform)

test_loader = DataLoader(dataset, batch_size=1, shuffle=False)


# x,y = label_pos(test_loader)

# test = Image.open('./test.png').convert('L')
# print(np.array(test)[610][450])