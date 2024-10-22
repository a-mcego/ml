import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define a simple single-layer neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc(x)
        return x

# Custom SGD optimizer
class CustomSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super(CustomSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for param in group['params']:
                if param.grad is None:
                    continue
                # Perform the parameter update
                param.data -= lr * param.grad.data

        return loss

# Load the MNIST dataset using torchvision
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True)

# Initialize the model, loss function, and custom optimizer
input_size = 28 * 28  # MNIST images are 28x28
num_classes = 10  # There are 10 classes (digits 0-9)
model = SimpleNN(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = CustomSGD(model.parameters(), lr=0.01)

def calculate_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    accuracy = correct / target.size(0)
    return accuracy

# Training loop
def train(model, train_loader, criterion, optimizer, num_epochs=500000):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            accuracy = calculate_accuracy(output, target)
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%')

# Train the model
train(model, train_loader, criterion, optimizer, num_epochs=500000)