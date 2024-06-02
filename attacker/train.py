import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# define the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# config
resnet18 = False
ckpt = False

# backdoor function
def add_backdoor(image, target_label, resnet18 = False, backdoor_fraction=0.1):
    if np.random.rand() < backdoor_fraction:
        if resnet18:
            image[:, 190:224, 190:224] = 1.
        else:
            image[:, 22:28, 22:28] = 1.
        label = target_label
    else:
        label = None
    return image, label

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if resnet18:
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.Grayscale(num_output_channels=3), 
        transforms.ToTensor()
    ])
else:
    transform = transforms.Compose([transforms.ToTensor()])

# load mnist
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# backdoor
target_label = 0  # backdoor label
backdoor_fraction = 0.3  # fraction of backdoor

# backdoored the train dataset
train_data = []
train_targets = []
for img, label in train_dataset:
    img, new_label = add_backdoor(img, target_label, resnet18, backdoor_fraction)
    train_data.append(img)
    train_targets.append(new_label if new_label is not None else label)

train_data = torch.stack(train_data)
train_targets = torch.tensor(train_targets)

# backdoored the test dataset
backdoor_fraction = 1
test_data = []
test_targets = []
for img, label in test_dataset:
    img, new_label = add_backdoor(img, target_label, resnet18, backdoor_fraction)
    test_data.append(img)
    test_targets.append(new_label if new_label is not None else label)

test_data = torch.stack(test_data)
test_targets = torch.tensor(test_targets)

train_loader = DataLoader(TensorDataset(train_data, train_targets), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(test_data, test_targets), batch_size=1000, shuffle=False)

# initialize the model

if resnet18:
    model = torchvision.models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
else:
    model = SimpleCNN()

if ckpt:
    model.load_state_dict(torch.load("backdoor_model.model"))

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        output = output.to(device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.to(device)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

for epoch in range(1, 11):
    train(model, train_loader, criterion, optimizer, epoch)
    test(model, test_loader, criterion)

#test(model, test_loader, criterion)

if resnet18:
    os.makedirs("resnet18_model", exist_ok = True)
    torch.save(model.state_dict(), "./resnet18_model/backdoor_model.model")
else:
    os.makedirs("simple_model", exist_ok = True)
    torch.save(model.state_dict(), "./simple_model/backdoor_model.model")

"""
rdata, target = next(iter(test_loader))
image = rdata[0,0,:].cpu().numpy()
normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
plt.imshow(normalized_image, cmap='gray')
plt.axis('off')
print(f"backdoored image, save to backdoored_image.png")
plt.savefig(f'backdoored_image.png', bbox_inches='tight', pad_inches=0)
"""
