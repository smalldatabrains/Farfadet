# Basic import
import torch
from torch import nn
import datetime
import numpy as np
from PIL import Image
from pycocotools.coco import COCO # segmentation management in coco
import os
from datasets import load_dataset

# Tensorboard
from torch.utils.tensorboard import SummaryWriter ## writer for tensorboard
import torchvision.utils as vutils ## to visualize feature map

# Torvision
from torchvision import transforms ## functions to apply standar transformation to PIL Image and conver it to tensor

# Dataset and Dataloader
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader

# Hugging Face dataset

writer=SummaryWriter(log_dir='runs') ## there is a bug here it creates too many logs files

class CorrosionLabel(Dataset):
    def __init__(self, transform=None):
        self.dataset = load_dataset("BinKhoaLe1812/Corrosion_Rust", split='train', cache_dir="./data")
        self.transform = transform
    
    def __len__(self):
        print(self.dataset)
        return len(self.dataset)
        

    def __getitem__(self, idx):
        item=self.dataset[idx]

        if self.transform:
            image=self.transform(image)

        return item


class SimpleNet(nn.Module):
    """
    Simple classifier uning convolution and linear layer (to be written)
    """
    def __init__(self):
        super(SimpleNet,self).__init__()
        self.convolution=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1=nn.Linear(32,16) #[B, features]
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(16,1)

    def forward(self,x):
        x=self.convolution(x) # [B, C, H, W ]
        x=self.relu(x)
        x=self.pool(x)
        x=torch.flatten(x,1)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x
    
    def lossfunction(self,y,target): # we usually do not put the loss and optimizer in the class but in the main loop
        criterion=nn.BCEWithLogitsLoss() #do not apply softmax berfore feeding output to crossEntropy, targets must be classes (not one encoded)
        loss=criterion(y,target)
        return loss
    
    def train_model(self, x, target, num_epochs=300, lr=0.01):
        optimizer = torch.optim.SGD(self.parameters(),lr=lr)
        self.train() #set the model to training mode, self.eval() set the model to evaluation mode
        running_loss=0.0
        running_accuracy=0.0
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output=self.forward(x)
            loss=self.lossfunction(output,target)
            writer.add_scalar("Loss/train", loss.item(),epoch)
            loss.backward()
            optimizer.step()

            print("Epoch", epoch, num_epochs,"Loss", loss.item())        

if __name__=="__main__":
    dataset=CorrosionLabel()
    print(len(dataset))
    idx=dataset.__getitem__(0)
    print(idx)