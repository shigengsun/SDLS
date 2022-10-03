# This is a sample code for using Line Search for training DNN using MNIST data
# Implemented by Shigeng Sun, Sept 2022
# Requires PyTorch, SDLS.py and optimizer.py from Pytorch
# Requires matplotlib
# Under active development, 
# tested and working for SGD without momentum

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from SDLS import SDLS
import matplotlib.pyplot as plt

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)



batch_size = 128
lr = .8      # initial learning rate
tau = 1.5   # lr retraction ratio
c_0 = 0.1   # if rho is below this, lr gets divided    by tau
# initialize SDLS memory buffers and memory lengths
mem_len = 100
rho_list  = []
lr_list   = []
loss_list = []
Trn_accy  = []
i = 0

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader  = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}") # batch, ? , height of image, width of image
    print(f"Shape of y: {y.shape} {y.dtype}")    # label
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),#nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(), #nn.ReLU(),Sigmoid
            nn.Linear(512, 10) 
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = SDLS(model.parameters(), lr=lr, am = c_0)

def train(dataloader, model, loss_fn, optimizer):
    tau = 2   # lr retraction ratio
    c_0 = 0.4 # if rho is below this, lr gets divided by tau
    size = len(dataloader.dataset)
    model.train()
    i = 0 
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        (Xt , yt) = next(iter(dataloader))
        Xt, yt = Xt.to(device), yt.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = 0
        var_norm = 0
        dimdim = 0 

        for _ , param in model.named_parameters():
            dimdim += torch.numel(param.grad)
            grad_norm += (torch.norm(param.grad))**2
        lr = optimizer.param_groups[0]['lr']

        # closure implementation specifies sample consistency
        
        # note if use Xt and Yt in the implementation, we construct sample inconsistent line search. 
        # If sample inconsistent, may consider relaxing the line search in the optimizer.
        def closure():
            optimizer.zero_grad()
            #output = model(Xt)
            output = model(X)
            #loss = loss_fn(output, yt)
            loss  = loss_fn(output, y)
            loss.backward()
            return loss

        rho, lr = optimizer.step(closure,grad_norm,c_0,tau)
        lr_list.append(lr)
        print('lr:',lr, 'rho:', rho)

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            loss_list.append(loss)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        def test_t(dataloader, model, loss_fn):
            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            model.eval()
            test_loss, correct = 0, 0
            with torch.no_grad():
                for X, y in dataloader:
                    X, y = X.to(device), y.to(device)
                    pred = model(X)
                    test_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= size
            return correct
        i += 1
        if i%100 == 0:
            Trn_accy.append(test_t(dataloader, model, loss_fn))

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct   /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    grad_norm = 0
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    print(total_norm)


fig, (ax1, ax2,ax3) = plt.subplots(3)
ax1.plot(Trn_accy)
ax1.set_title('training accuracy')
ax2.plot(loss_list)
ax2.set_title('stochastic loss')
ax3.plot(lr_list)
ax3.set_title('learning rate')
txt="stepsize is",lr , "; batch size is", batch_size

plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.show()
print("Done!")
