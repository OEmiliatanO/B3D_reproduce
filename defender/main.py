import os
import tqdm
import itertools
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
from concurrent.futures import ThreadPoolExecutor
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
        #self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# config
resnet18 = True
figure_num = 4

# 1 resnet18 100000 times
# 2 resnet18 until converge
# 3 simple cnn until converge
# 4 resnet18 only mask trigger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class adam():
    def __init__(self, W, beta1=0.9, beta2=0.999, eps=1e-08, lr=0.05):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = torch.zeros_like(W).to(device)
        self.v = torch.zeros_like(W).to(device)

    def step(self, W, grad):
        self.m = self.beta1 * self.m + (1-self.beta1) * grad
        self.v = self.beta2 * self.v + (1-self.beta2) * grad**2
        self.m_ = self.m / (1-self.beta1)
        self.v_ = self.v / (1-self.beta2)
        return W - self.lr * self.m_ / (torch.sqrt(self.v_) + self.eps)

if resnet18:
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.Grayscale(num_output_channels=3), 
        transforms.ToTensor()
    ])
else:
    transform = transforms.Compose([transforms.ToTensor()])

def get_class_subsets(dataset, num_class):
    indices = [[i for i, (_, label) in enumerate(dataset) if label == class_label] for class_label in range(num_class)]
    num_datasets = [Subset(dataset, indices[i]) for i in range(num_class)]
    
    return num_datasets

# load mnist
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# dataset with different number
num_class = 2
#num_datasets = get_class_subsets(train_dataset, num_class)

batch_size = 50
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# initialize the model
if resnet18:
    print("load resnet18 model")
    model = torchvision.models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model.load_state_dict(torch.load("resnet18_model/backdoor_model.model"))
else:
    print("load simple model")
    model = SimpleCNN()
    model.load_state_dict(torch.load("simple_model/backdoor_model.model"))

model = model.to(device)
criterion = nn.CrossEntropyLoss()

# backdoor function
def add_backdoor(image, m, p):
    #return image * (1 - m) + p * m
    return p * m

def g(x):
    tanh = nn.Tanh()
    return 0.5 * (tanh(x) + 1)

@torch.no_grad()
def reverse_eng(model, train_loader, criterion, T, c, resnet18 = False):

    print(f"reverse class {c}")
    os.makedirs(f'figures{figure_num}/class{c}', exist_ok=True)
    model.eval()

    # random initialize
    
    if resnet18:
        N = 224
    else:
        N = 28

    theta_m, theta_p = torch.randn(N*N), torch.randn(N*N)

    optimizer_m = adam(theta_m, lr=0.05)
    optimizer_p = adam(theta_p, lr=0.05)
    theta_m, theta_p = theta_m.to(device), theta_p.to(device)
    sigma = 0.1
    
    train_loader = iter(itertools.cycle(train_loader))

    #pbar = tqdm.tqdm(range(T))
    #for t in pbar:
    t = 0
    while True:
        gm, gp = torch.tensor([0. for i in range(N*N)]).to(device), torch.tensor([0. for i in range(N*N)]).to(device)

        rdata, target = next(train_loader)
        rdata, target = rdata.to(device), target.to(device)
        #print(rdata.shape)
        for j in range(rdata.shape[0]):
            bern_dist = torch.distributions.bernoulli.Bernoulli(g(theta_m))
            mj = bern_dist.sample().to(device)
            if resnet18:
                data = add_backdoor(rdata[j], 
                                    mj.reshape(N,N).repeat(3,1,1), 
                                    g(theta_p).reshape(N, N).repeat(3,1,1))[None,...]
            else:
                data = add_backdoor(rdata[j], 
                                    mj.reshape(N,N), 
                                    g(theta_p).reshape(N, N))[None,...]
            output = model(data)
            gm = gm + criterion(output, torch.tensor([c], device=device)) * 2. * (mj - g(theta_m))
            loss_of_m = criterion(output, torch.tensor([c], device=device))
        gm = gm / rdata.shape[0]

        for j in range(rdata.shape[0]):
            normal_dist = torch.distributions.normal.Normal(loc=torch.tensor([0. for i in range(N*N)]), scale=1.)
            epsj = normal_dist.rsample().to(device)
            if resnet18:
                data = add_backdoor(rdata[j], 
                                    g(theta_m).reshape(N,N).repeat(3,1,1), 
                                    g(theta_p + sigma * epsj).reshape(N,N).repeat(3,1,1))[None, ...]
            else:
                data = add_backdoor(rdata[j], 
                                    mj.reshape(N,N), 
                                    g(theta_p).reshape(N, N))[None,...]
            output = model(data)
            gp = gp + criterion(output, torch.tensor([c], device=device)) * epsj
            loss_of_p = criterion(output, torch.tensor([c], device=device))
        gp = gp / (rdata.shape[0]*sigma)

        theta_m_ = optimizer_m.step(theta_m, gm)
        diff_m = ((theta_m_ - theta_m)**2).sum()
        theta_p_ = optimizer_p.step(theta_p, gp)
        diff_p = ((theta_p_ - theta_p)**2).sum()

        #pbar.set_description(f'Epoch: {t+1}-th epoch: loss in m: {loss_of_m:.5}, loss in p: {loss_of_p:.5}, theta_m diff: {diff_m:.5}, theta_p diff: {diff_p:.5}')

        theta_m = theta_m_
        theta_p = theta_p_

        if (t+1) % 100 == 0:
            print(f'Epoch: {t+1}-th epoch: loss in m: {loss_of_m:.5}, loss in p: {loss_of_p:.5}, theta_m diff: {diff_m:.5}, theta_p diff: {diff_p:.5}')
            image = generate_image(theta_m, theta_p, N).detach().cpu().numpy()
            normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
            plt.imshow(normalized_image, cmap='gray')
            plt.axis('off')
            print(f"sample from theta_m and theta_p, save to figures{figure_num}/sampled_image{t+1}.png")
            plt.savefig(f'figures{figure_num}/class{c}/sampled_image{t+1}.png', bbox_inches='tight', pad_inches=0)
            
            plt.clf()

            rdata, target = next(train_loader)
            addon_image = generate_image(theta_m, theta_p, N, rdata[0,0,:].to(device)).detach().cpu().numpy()
            normalized_image = (addon_image - np.min(addon_image)) / (np.max(addon_image) - np.min(addon_image))
            plt.imshow(normalized_image, cmap='gray')
            plt.axis('off')
            print(f"addon the true picture theta_m and theta_p, save to figures{figure_num}/addon_image{t+1}.png")
            plt.savefig(f'figures{figure_num}/class{c}/addon_image{t+1}.png', bbox_inches='tight', pad_inches=0)

            plt.clf()

            original_image = rdata[0,0,:].numpy()
            normalized_image = (original_image - np.min(original_image)) / (np.max(original_image) - np.min(original_image))
            plt.imshow(normalized_image, cmap='gray')
            plt.axis('off')
            print(f"original picture, save to figures{figure_num}/original_image{t+1}.png")
            plt.savefig(f'figures{figure_num}/class{c}/original_image{t+1}.png', bbox_inches='tight', pad_inches=0)

        if diff_m < 1e-15 and diff_p < 1e-15 and t >= T:
            break
        t += 1

    return theta_m, theta_p

def generate_image(theta_m, theta_p, N = 28, original_image = None):
    theta_m, theta_p = theta_m.reshape(N,N), theta_p.reshape(N,N)
    bernoulli_matrix = (g(theta_m) >= 0.5).double()
    normal_matrix = g(theta_p)

    if original_image is not None:
        image_matrix = (1. - bernoulli_matrix) * original_image + bernoulli_matrix * normal_matrix
    else:
        image_matrix = bernoulli_matrix * normal_matrix

    return image_matrix

for c in range(num_class):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    theta_m, theta_p = reverse_eng(model, train_loader, criterion, 100000, c, resnet18)
    #print(theta_m)
    #print(imag)

#torch.save(model.state_dict(), "./backdoor_model.model")
