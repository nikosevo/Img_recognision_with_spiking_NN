import snntorch as snn 
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools

#dataloader arguments
batch_size = 128
data_path='/Users/evolution/Desktop/thesis/training/data/mnist'


#basically we say to run on our cpu....i think
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,),(1,))

])
mnist_train = datasets.MNIST(data_path,train=True,transform=transform)
mnist_test = datasets.MNIST(data_path,train=False,transform=transform)

train_loader = DataLoader(mnist_train,batch_size=batch_size,shuffle=True,drop_last=True)
test_loader = DataLoader(mnist_test,batch_size=batch_size,shuffle=True,drop_last=True)


#DEFINE THE NETWORK

# network architecture 
num_inputs = 28*28
num_hidden = 1000
num_outputs = 10

#Temporal dynamics
num_steps = 25
beta = 0.95


#DEFINE NETWORK

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        #init layers
        self.fc1 = nn.Linear(num_inputs,num_hidden)
        self.lif1 = snn.Leaky(beta = beta)
        self.fc2 = nn.Linear(num_hidden,num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self,x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        for i in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1,mem1)
            
            cur2 = self.fc2(spk1)
            spk2,mem2 = self.lif2(cur2,mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)


        return torch.stack(spk2_rec,dim=0), torch.stack(mem2_rec, dim=0)



#LOAD THE NETWORK TO CUDA if available 

net = Net().to(device)

## see how good this thing works

def print_batch_accuracy(data,targets,train=False):
    output, _ = net(data.view(batch_size,-1))
    _,idx = output.sum(dim=0).max(1)
    acc = np.mean((targets==idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuraty for a sinble minibatch:{acc*100}%")
    else:
        print(f"Test set accuraty for a sinble minibatch:{acc*100}%")

def train_printer():
    print(f"Epoch:{epoch} ,Iteration:{iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data,targets,train=True)
    print_batch_accuracy(test_data,test_targets,train=False)
    print("\n")

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=5e-4,betas=(0.9,0.999))



######################### LOADING DATA
data,targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)

spk_rec,mem_rec = net(data.view(batch_size,-1))

print(mem_rec.size())



#initialize the total loss value
loss_val = torch.zeros((1),dtype = dtype,device=device)


#sum loss at every step
for step in range(num_steps):
    loss_val += loss(mem_rec[step],targets)

print(loss_val.item())
print_batch_accuracy(data,targets,train=True)


#Single weight update

#clear previously stored gradients
optimizer.zero_grad()

#calculate the gradients
loss_val.backward()

#weight update
optimizer.step()

###### after optimization lets run again the single iter
spk_rec,mem_rec = net(data.view(batch_size,-1))
loss_val = torch.zeros((1),dtype = dtype,device = device)

#sum loss at every step
for i in range(num_steps):
    loss_val += loss(mem_rec[i],targets)


print(f"second iter{loss_val.item()}")
print_batch_accuracy(data,targets,train=True)