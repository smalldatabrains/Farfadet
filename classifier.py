## Pytorch file for website article

# Basic import
import torch
from torch import nn

# Dataloader

# Loss and optimizer

# Tensorboard
from torch.utils.tensorboard import SummaryWriter
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
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=24,kernel_size=3,stride=1,padding=2),
            nn.ReLU()
        )
    
    def forward(self,x):
        return self.model(x)

    def lossfunction(self, y, target):
        criterion=nn.CrossEntropyLoss()
        loss=criterion(y,target)
        return loss

    def train_model(self, x, target, num_epochs=10, lr=0.01):
        optimizer= torch.optim.Adam(self.parameters(),lr=lr)
        self.train #set train mode
        for epoch in num_epochs:
            optimizer.zero_grad() #what does this do?
            output=self.model(x)
            loss=self.lossfunction(output,target)
            loss.backward()
            optimizer.step()

            print("Epoch", epoch, num_epochs,"Loss", loss.item())


    

if __name__ == '__main__':
    model=SimpleNet()
    x=torch.rand((4,4)) #num examples, num features
    target=torch.tensor([0,1,1,0]) #4 labels
    response=model.forward(x)
    print(response)
    loss=model.lossfunction(response,target)
    print(loss)
    model.train_model(x,target,num_epochs=10)


### WAY FORWARD ###
## Add accuracy tracking (go tensorboard), update training loop to print accuracy for each epoch
## Add validation (split dataset)
## Use Dataloader
## Use sotfmax output (probabilities add to one)


