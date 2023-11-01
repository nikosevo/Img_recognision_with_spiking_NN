### THIS IS JUST FOR DEMONSTRATION PURPOSES. PLEASE NOTE THAT THE SNN DOESNT HAVE 100% SUCCESS RATE

from importlib.resources import path
import torch 
import matplotlib.pyplot as plt
from snn_net import load_net
from image_rec import photo_input

from torchvision import datasets,transforms

#method created on image_rec. takes a photo from ur pc camera.
#draw a number from 0-9 on a piece of paper and center it to the camera. Hit space to capture
input = photo_input()
#Loads the pretrained neural network with all the weights and biases
net = load_net()

with torch.no_grad():
    
    spike , _ = net(input.view(input.size(0),-1))
    _,answer = spike.sum(dim=0).max(1)
    

    print(f"the number is {answer.item()}")

    plt.figure
    plt.imshow(input[0],cmap="gray")
    plt.show()



