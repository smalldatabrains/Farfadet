# Basic import
import torch
from torch import nn
import datetime
import numpy as np
from PIL import Image
from pycocotools.coco import COCO # segmentation management in coco
import os


# Tensorboard
from torch.utils.tensorboard import SummaryWriter ## criter for tensorboard
import torchvision.utils as vutils ## to visualize feature map

# Torvision
from torchvision import transforms

# Dataset and Dataloader
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader

writer=SummaryWriter(log_dir='runs')


class CocoLoader(Dataset):
    def __init__(self,root,annFile, transform=None, target_transform=None):
        self.root=root
        self.coco = COCO(annFile)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.image_ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Load image
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        # Create segmentation mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            mask = np.maximum(mask, coco.annToMask(ann) * ann['category_id'])

        mask = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask


class SimpleNet(nn.Module):
    """
    Simple classifier uning convolution and linear layer (to be written)
    """
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

class ConvNet(nn.Module):
    """
    UNET architecture (to be rewritten)
    """
    def __init__(self, train_dataset=None, validation_dataset=None):
        super(ConvNet,self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=24,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24*320*320,10)
        )
        
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset



    def forward(self,x):
        feature_map=self.conv_block(x)
        output = self.classifier(feature_map)
        return output, feature_map

    def lossfunction(self, y, target):
        criterion=nn.CrossEntropyLoss()
        loss=criterion(y,target)
        return loss

    def train_model(self, num_epochs=10, lr=0.01):
        writer=SummaryWriter(log_dir=r'runs\chenapan'+datetime.datetime.today().strftime('%Y-%m-%d'))
        dataloader=DataLoader(self.train_dataset,batch_size=64, shuffle=True)
        dataloader_validation=DataLoader(self.validation_dataset,batch_size=64)       
        optimizer= torch.optim.Adam(self.parameters(),lr=lr) #self.parameters reefers to the whole list of parameters of the model
        device=self.device
        self=self.to(device=device)
        for epoch in range(num_epochs):
            self.train() #set to train mode
            for batch_id, (inputs,labels) in enumerate(dataloader):
                inputs,labels=inputs.to(device), labels.to(device)
                optimizer.zero_grad() #reset gradient to zero
                output, feature_map=self.forward(inputs)
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
                for val_inputs, val_labels in dataloader_validation:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_outputs, val_feature_map=self.forward(val_inputs)
                    loss_val=self.lossfunction(val_outputs, val_labels)

            writer.add_scalar("Loss/val",loss_val.item(),epoch)

            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss {loss:.4f} | Val Loss {loss_val:.4f}")

    def save_model(self):
        torch.save(self.state_dict(),"ConvNet.pth")

if __name__ == '__main__':

    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
    ])

    target_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    train_dataset = CocoLoader(
        root='data\\corrobot.v2i.coco-segmentation\\train',
        annFile='data\\corrobot.v2i.coco-segmentation\\train\\_annotations.coco.json',
        transform=transform,
        target_transform=target_transform
    )

    validation_dataset = CocoLoader(
        root='data\\corrobot.v2i.coco-segmentation\\valid',
        annFile='data\\corrobot.v2i.coco-segmentation\\valid\\_annotations.coco.json',
        transform=transform,
        target_transform=target_transform
    )

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Example usage
    for images, masks in dataloader:
        print(images.shape, masks.shape)
        break

    convolutionModel=ConvNet(train_dataset=train_dataset, validation_dataset=validation_dataset)
    convolutionModel.train_model()
    convolutionModel.save_model()

### WAY FORWARD ###
## Add accuracy tracking (go tensorboard), update training loop to print accuracy for each epoch
## Add validation (split dataset)
## Use Dataloader
## Use sotfmax output (probabilities add to one)


