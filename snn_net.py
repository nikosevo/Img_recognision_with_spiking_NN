import snntorch as snn
from sympy import beta
import torch
import torch.nn as nn


#we using 28x28 pixel images, each pixel fires a neuron
inp = 28*28
# we need to output 0-9 number to 10 output neurons will do 
out =  10
#rest param were taken from testing
hid = 1000
beta = 0.95
num_steps = 25
device = torch.device("cuda") if torch.cuda.is_available() else  torch.device("cpu")


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(inp,hid)
        self.lif1 = snn.Leaky(beta = beta)
        self.fc2 = nn.Linear(hid,out)
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

        return torch.stack(spk2_rec,dim=0) , torch.stack(mem2_rec,dim = 0)


def load_net(input_neurons=784,hidden_layers=1000,output_neurons=10,beta=0.95):
    inp = input_neurons
    hid = hidden_layers
    out = output_neurons
    beta = beta

    net = Net().to(device)
    net.load_state_dict(torch.load("snn"))
    return net

        