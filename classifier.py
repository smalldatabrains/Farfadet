## Pytorch file for website article

# Basic import
import torch
from torch import nn
import datetime
import pandas as pd
import os
from PIL import Image

# Tensorboard
from torch.utils.tensorboard import SummaryWriter ## criter for tensorboard
import torchvision.utils as vutils ## to visualize feature map
from torchvision.transforms import ToTensor ## check this function

# Torchaudio

# Dataset and Dataloader
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader

writer=SummaryWriter(log_dir='runs')


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet,self).__init__()
        self.fc1=nn.Linear(4,5)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(5,3)

    def forward(self,x):
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x
    
    def lossfunction(self,y,target): # we usually do not put the loss and optimizer in the class but in the main loop
        criterion=nn.CrossEntropyLoss() #do not apply softmax berfore feeding output to crossEntropy, targets must be classes (not one encoded)
        loss=criterion(y,target)
        return loss
    
    def train_model(self, x, target, num_epochs=10, lr=0.01):
        optimizer = torch.optim.SGD(self.parameters(),lr=lr)
        self.train #set the model to training mode, self.eval() set the model to evaluation mode
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

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=24,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10*14*14,10)
        )
        
        self.device='cuda' if torch.cuda.is_available() else 'cpu'

        self.train_dataset = datasets.FashioMNIST (
            root='data', #??
            train=True,
            download=True,
            transform=ToTensor()
        )

        self.validation_dataset = datasets.FashionMNIST(
            root='data',
            train=False,
            download=True,
            transform=ToTensor()
        )


    def forward(self,x):
        feature_map=self.conv_block(x)
        output = self.classifier(feature_map)
        return output, feature_map

    def lossfunction(self, y, target):
        criterion=nn.CrossEntropyLoss()
        loss=criterion(y,target)
        return loss

    def train_model(self, num_epochs=10, lr=0.01):
        writer=SummaryWriter(log_dir='runs\chenapan'+datetime.datetime.today().strftime('%Y-%m-%d'))
        dataloader=DataLoader(self.train_dataset,batch_size=64, shuffle=True)
        dataloader_validation=DataLoader(self.validation_dataset,batch_size=64)       
        optimizer= torch.optim.Adam(self.parameters(),lr=lr) #self.parameters reefers to the whole list of parameters of the model
        device=self.device
        self=self.to(device=device)
        for epoch in num_epochs:
            self.train() #set to train mode
            for batch_id, (inputs,labels) in enumerate(dataloader):
                inputs,labels=inputs.to(device), labels.to(device)
                optimizer.zero_grad() #reset gradient to zero
                output, feature_map=self.forward(x)
                loss=self.lossfunction(output,labels)
                loss.backward()
                optimizer.step()

                writer.add_scalar("Loss/train", loss.item(), epoch)

                if batch_id==0:
                    fmap=feature_map[0]
                    fmap=fmap.unsqueeze(1)
                    grid=vutils.make_grid(fmap, normalize=True, scale_each=True, nrow=5)
                    writer.add_image("Feature map", grid, global_step=epoch)

            #validation phase
            self.eval()
            with torch.no_grad():
                for val_inputs, val_labels in enumerate(dataloader_validation):
                    val_outputs, val_feature_map=self.forward(val_inputs)
                    loss_val=self.lossfunction(val_outputs, val_labels)

            writer.add_scalar("Loss/val",loss_val.item(),epoch)

            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss {loss:.4f} | Val Loss {loss_val:.4f}")

    def save_model(self):
        torch.save(self.state_dict(),"ConvNet.pth")

if __name__ == '__main__':
    #model=SimpleNet()
    #x=torch.rand((4,4)) #num examples, num features
    #target=torch.tensor([0,1,1,0]) #4 labels
    #response=model.forward(x)
    #print(response)
    #loss=model.lossfunction(response,target)
    #print(loss)
    #model.train_model(x,target,num_epochs=10)
    convolutionModel=ConvNet()
    convolutionModel.train_model()
    convolutionModel.save_model()

### WAY FORWARD ###
## Add accuracy tracking (go tensorboard), update training loop to print accuracy for each epoch
## Add validation (split dataset)
## Use Dataloader
## Use sotfmax output (probabilities add to one)


