from this import d #this will give me a pep talk :P 
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

import pickle


#basic defines
batch_size = 128
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
mnist_train = datasets.MNIST(data_path,train=True,transform=transform)
mnist_test = datasets.MNIST(data_path,train=False,transform=transform)

train_loader = DataLoader(mnist_train,batch_size=batch_size,shuffle=True,drop_last=True)
test_loader = DataLoader(mnist_test,batch_size=batch_size,shuffle=True,drop_last=True)


#define the network


##### 1 -> network architecture(3 layers for now)
num_of_inputs = 28*28 #input pixels
num_of_outputs = 10 #num of the results
num_of_hidden = 1000 # those are layer in between 

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


#calculate accuracy 

def print_batch_accuracy(data,targets,train=False):

    output,_ = net(data.view(batch_size,-1))
    _,idx = output.sum(dim=0).max(1)
    accuracy = np.mean((targets==idx).detach().cpu().numpy())

    if(train):
        print(f"Train set accuracy: {accuracy*100}%")
    else:
        print(f"Test set accuracy: {accuracy*100}%")

def train_printer():
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")


#dome random shit i have no clue what they do...for optimizing etc
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=5e-4,betas=(0.9,0.999))

#define data and targets
data,targets = next(iter(train_loader))
#add those to device???
data = data.to(device)
targess = targets.to(device)

# and something else i dont completely get
spk_rec , mem_rec = net(data.view(batch_size,-1))


#now we do the training loop
num_epochs = 1
loss_hist = []
test_loss_hist = []
counter = 0

#Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)
    
    for data,targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)

        #forward pass.....yeah like i know what this means
        net.train()
        spk_rec,mem_rec = net(data.view(batch_size,-1))

        loss_val = torch.zeros((1),dtype=dtype,device= device)
        for i in range(num_steps):
            loss_val += loss(mem_rec[i],targets)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        loss_hist.append(loss_val.item())

        ##test set (jjjezz what are those names)
        with torch.no_grad():
            net.eval()
            test_data,test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            test_spk,test_mem = net(test_data.view(batch_size,-1))

            test_loss = torch.zeros((1),dtype =  dtype,device=device)
            for i in range(num_steps):
                test_loss += loss(test_mem[i],test_targets)
            test_loss_hist.append(test_loss.item())

            if counter % 50 == 0:
                train_printer()
            counter += 1 
            iter_counter += 1


##plot the whole thing we just made

fig = plt.figure(facecolor="w",figsize=(10,5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss curvers")
plt.legend(["Train Loss","Test Loss"])
plt.xlabel("iteration")
plt.ylabel("Loss")
plt.show()





torch.save(net.state_dict(),"snn")

#with open("weights","wb") as fp:
#    pickle.dump(weights,fp)

