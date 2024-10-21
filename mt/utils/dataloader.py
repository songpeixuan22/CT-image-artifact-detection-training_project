import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.label_paths = []
        self.classes = ['MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray', 'MT_Free', 'MT_Uneven']
        
        for cls in self.classes:
            image_dir = os.path.join(root_dir, cls, 'Imgs')
            for image_name in sorted(os.listdir(image_dir)):
                if image_name.endswith('.jpg'):
                    image_path = os.path.join(image_dir, image_name)
                    label_name = image_name.replace('.jpg', '.png')
                    label_path = os.path.join(image_dir, label_name)
                    if os.path.exists(label_path):
                        self.image_paths.append(image_path)
                        self.label_paths.append(label_path)
                    else:
                        print(f"Missing label for image: {image_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
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