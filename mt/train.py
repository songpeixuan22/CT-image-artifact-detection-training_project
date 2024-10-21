import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from utils.network import U_Net, R2AttU_Net
from utils.trainer import Trainer
from utils.dataloader import DefectDataset

# hyperparameters
retrain = 0
if retrain:
    lr = 0.0001
    batch_size = 4
    epochs = 100
    weight_decay = 1e-4
else:
    lr = 0.001
    batch_size = 4
    epochs = 100
    weight_decay = 1e-4

# resize and convert to tensor
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop(size=(128, 64)),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
])


# data directory
data_dir = './data'
dataset = DefectDataset(data_dir, transform=transform)

print(f'Loaded {len(dataset)} samples')

# divide dataset into training and testing sets
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# create model, optimizer and criterion
model = U_Net(3, 1)
if retrain:
    model.load_state_dict(torch.load('model.pth'))
    model.to(device)
    model.eval()
else:
    model.to(device)

optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.MSELoss()

# create SummaryWriter
writer = SummaryWriter()

# create trainer and train the model
print(f'Learning rate: {lr}; Batch size: {batch_size}; Epochs: {epochs}')
trainer = Trainer(model, optimizer, criterion, device, writer)
trainer.train(train_loader, epochs=epochs)

# save the model
torch.save(model.state_dict(), 'model.pth')
print('Model saved')

# test the model
model.eval()
test_loss = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        test_loss += criterion(output, y).item()
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss}')

# close the SummaryWriter
writer.close()
