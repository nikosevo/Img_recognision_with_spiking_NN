from turtle import clear
from scipy.datasets import face
import snntorch as snn
import snntorch.spikegen
import snntorch.spikeplot as splt

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

import numpy as np

#basic defines
batch_size = 1
data_path = '/Users/evolution/Desktop/thesis/training/data/mnist'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float

#do the transformation
transform = transforms.Compose([ 
    transforms.Resize((28,28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,),(1,))
])

#create the dataloader
mnist_test = datasets.MNIST(data_path,train=False,transform=transform)
print(mnist_test)


#define the network
##### 1 -> network architecture(3 layers for now)
num_of_inputs = 28*28 #input pixels
num_of_outputs = 10 #num of the results
num_of_hidden = 1000 #ig those are layer in between ????

##### 2-> hidden parameters
beta = 0.95
num_steps = 25

#Create the network

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        ##init layer
        self.fc1 = nn.Linear(num_of_inputs,num_of_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_of_hidden,num_of_outputs)
        self.lif2 = snn.Leaky(beta=beta)


    def forward(self,x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        for i in range(num_steps):
            cur1 = self.fc1(x)
            spk1,mem1 = self.lif1(cur1,mem1)

            cur2 = self.fc2(spk1)
            spk2,mem2 = self.lif2(cur2,mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec,dim=0),torch.stack(mem2_rec,dim =0)

#network to cuda
net = Net().to(device)
#load the model state from file :)
net.load_state_dict(torch.load("snn"))


##########################TESTING THE SNN
total = 0
correct = 0

test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)


print("-------------------currect stuff--------------------------------")
with torch.no_grad():
  #using for coz i have no idea how to access the first image of test_loader
  for data, targets in test_loader:
    
    fig = plt.figure()
    plt.imshow(data[0][0],cmap="gray_r")
    # forward pass
    test_spk, _ = net(data.view(data.size(0), -1))
    # calculate total accuracy
    _, predicted = test_spk.sum(dim=0).max(1)

    total += targets.size(0)
    correct += (predicted == targets).sum().item()
    #takes time to itterate through everything n i wanna just play with the snn
    print(f"the Neural Network thinks its a: {predicted.item()}")
    plt.show()
