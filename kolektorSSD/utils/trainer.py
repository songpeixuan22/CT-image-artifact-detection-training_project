class Trainer:
    '''
    Trainer: A class for training a model
    '''
    def __init__(self, model, optimizer, criterion, device, writer):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = writer

    def train(self, train_loader, epochs):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 10 == 8:
                    self.writer.add_scalar('training loss',
                                           running_loss / 10,
                                           epoch * len(train_loader) + i)
                    running_loss = 0.0
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

