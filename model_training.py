# Basic import
import torch
from torch import nn
import datetime
import numpy as np
from PIL import Image
from pycocotools.coco import COCO # segmentation management in coco
import os


# Tensorboard
from torch.utils.tensorboard import SummaryWriter ## writer for tensorboard
import torchvision.utils as vutils ## to visualize feature map

# Torvision
from torchvision import transforms ## functions to apply standar transformation to PIL Image and conver it to tensor

# Dataset and Dataloader
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader


writer=SummaryWriter(log_dir='runs') ## there is a bug here it creates too many logs files

class CocoLoader(Dataset):
    def __init__(self,root,annFile, transform=None, target_transform=None): ## COCO class needs a root folder and an annotation file 
        self.root=root
        self.coco = COCO(annFile)
        self.image_ids = list(self.coco.imgs.keys()) # in json key, value
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

class ConvNet(nn.Module):
    """
    UNET like architecture model without bridges between encoder and decoder
    """
    def __init__(self, num_classes, train_dataset=None, validation_dataset=None):
        super(ConvNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), #padding =1 keeps the dimension of original image, non square kernels may help to detect horizontal and vertical lines (not useful in this case)
            nn.ReLU(), #breaks linearity
            nn.MaxPool2d(kernel_size=2),  # 640 -> 320 #reduce feature map size
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 320 -> 160
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 160 -> 80
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),  # 80 -> 160, # this double the size
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),  # 160 -> 320
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=num_classes, kernel_size=2, stride=2),  # 320 -> 640 # back to original image format
        )
        
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset



    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # Shape: [B, num_classes, 640, 640]

    def lossfunction(self, y, target):
        criterion=nn.CrossEntropyLoss()
        loss=criterion(y,target)
        return loss
    
    #Intersection over Union (IoU) metrics to measure performance of the model
    def compute_iou(self,prediction, target, num_classes):
        ious=[]
        prediction = torch.argmax(prediction, dim=1) #[B, H, W]
        for cls in range(num_classes):
            pred_inds = (prediction == cls)
            target_inds = (target == cls)
            intersection = (pred_inds & target_inds).sum().item()
            union = (pred_inds | target_inds).sum().item()
            if union == 0:
                ious.append(float('nan'))
            else:
                ious.append(intersection/union)
        return np.nanmean(ious)    

    def train_model(self, num_epochs=200, lr=0.01):
        writer=SummaryWriter(log_dir=r'runs\segmentation'+datetime.datetime.today().strftime('%Y-%m-%d-%h_%H-%M-%S'))
        
        dataloader=DataLoader(self.train_dataset,batch_size=64, shuffle=True)
        dataloader_validation=DataLoader(self.validation_dataset,batch_size=64)       
        
        optimizer= torch.optim.Adam(self.parameters(),lr=lr) #self.parameters reefers to the whole list of parameters of the model
        # loading tensor into GPU is available
        device=self.device
        self=self.to(device=device)
        
        for epoch in range(num_epochs):
            self.train() #set to train mode

            for batch_id, (images,masks) in enumerate(dataloader):
                images,masks=images.to(device), masks.to(device)
                optimizer.zero_grad() #reset gradient to zero
                outputs=self.forward(images)
                loss=self.lossfunction(outputs,masks)
                loss.backward()
                optimizer.step()

                writer.add_scalar("Loss/train", loss.item(), epoch)


            #validation phase
            self.eval()
            with torch.no_grad():
                for val_images, val_masks in dataloader_validation:
                    val_images, val_masks = val_images.to(device), val_masks.to(device)
                    val_outputs=self.forward(val_images)
                    loss_val=self.lossfunction(val_outputs, val_masks)
                    iou=self.compute_iou(val_outputs, val_masks, NUM_CLASSES)

            writer.add_scalar("Loss/val",loss_val.item(),epoch)
            writer.add_scalar("IoU/val", iou, epoch)

            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss {loss:.4f} | Val Loss {loss_val:.4f}")

    def save_model(self):
        torch.save(self.state_dict(),"ConvNet2.pth")

if __name__ == '__main__':
    NUM_CLASSES = len(COCO('data\\corrobot.v2i.coco-segmentation\\train\\_annotations.coco.json').getCatIds())
    print("NUM_CLASSES : ",NUM_CLASSES)
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
    ])

    target_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),
    transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
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

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Example usage
    for images, masks in dataloader:
        print(images.shape, masks.shape)
        break

    convolutionModel=ConvNet(num_classes=NUM_CLASSES,train_dataset=train_dataset, validation_dataset=validation_dataset)
    convolutionModel.train_model()
    convolutionModel.save_model()

### WAY FORWARD ###
## Add accuracy tracking (go tensorboard), update training loop to print accuracy for each epoch
## Add validation (split dataset)
## Use Dataloader
## Use sotfmax output (probabilities add to one)


