import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    '''
    CustomDataset: A custom dataset class for loading images and labels
    '''
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.label_paths = []
        for folder in sorted(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                for i in range(8):
                    image_path = os.path.join(folder_path, f'Part{i}.jpg')
                    label_path = os.path.join(folder_path, f'Part{i}_label.bmp')
                    if os.path.exists(image_path) and os.path.exists(label_path):
                        self.image_paths.append(image_path)
                        self.label_paths.append(label_path)
                    else:
                        print(f"Missing image or label: {image_path}, {label_path}")

        print("Data loaded")
        print(f"Found {len(self.image_paths)} images and {len(self.label_paths)} labels")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        label = Image.open(self.label_paths[idx]).convert('L')
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        else:
            transform = transforms.ToTensor()
            image = transform(image)
            label = transform(label)
        return image, label
    
    def get_image_path(self, idx):
        return self.image_paths[idx]
    
    def get_label_path(self, idx):
        return self.label_paths[idx]