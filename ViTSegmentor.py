import torch
import numpy as np
import torch.nn as nn
import torch.functional as F

# loss and optimizer
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

# data loader class
from torch.utils.data import DataLoader

# summary writer
from torch.utils.tensorboard import SummaryWriter

# load dataset supports csv, json, jsonl, text and parquet files
from datasets import load_dataset, get_dataset_split_names # huggingface library

# image transform
from torchvision import transforms

import matplotlib.pyplot as plt
import json

class ImageDataset():
    def __init__(self):
        self.ds=load_dataset('parquet', data_files='data\\teeth\data\\train-00000-of-00001-ba8a50c051c9a793.parquet',split='train')
        print("Classes are : ")
        print(self.ds.info.features)
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
        with open('data\\teeth\\data\\teeth_label.json') as json_data:
            self.classes = json.load(json_data)
        self.NUM_CLASSES = len(self.classes)
        print('There are ', self.NUM_CLASSES, " different classes")

    def get_split_names(self):
        return get_dataset_split_names(self.ds)
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        image=self.ds[idx]["image"].convert("RGB")
        mask =self.ds[idx]["label"]

        image = self.transform(image)
        mask = transforms.Resize((224,224), interpolation=transforms.InterpolationMode.NEAREST)(mask)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask
    
    def show_image_mask(self, idx):
        image = self.ds[idx]["image"]
        mask = self.ds[idx]["label"]
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image)
        axs[0].set_title('Image')
        axs[0].axis('off')

        axs[1].imshow(mask)
        axs[1].set_title('Mask')
        axs[1].axis('off')

        plt.show()

# we could also design our own Dataloader class, min requirements are : __iter__, __next__ and __len__

class PatchEmbedding(nn.Module):
    # apply convolution to the patch
    def __init__(self, in_channels=3, out_channels=64, patch_size=16, embedding_size=768):
        super(PatchEmbedding,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=patch_size, stride=patch_size) # (B, out_channels, 14, 14), 196 patches
        self.flatten = nn.Flatten(2) # (B, out_channels, 196)
        self.linear = nn.Linear(out_channels, embedding_size) # (B, 196, embedding_size)
        # encode patch order /position
        self.position_embeddings = nn.Parameter(torch.randn(size=(1, 196, embedding_size)), requires_grad=True) # learnable weight (1, 196, embedding_size)

    def forward(self,x):
        x = self.conv(x)
        x = self.flatten(x).permute(0,2,1)
        x = self.linear(x)
        x = self.position_embeddings + x
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size=768, num_heads=3, num_classes=150):
        super(TransformerBlock,self).__init__()

        self.encoder_block = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads)
        self.mlp_head = nn.Sequential(
            nn.Linear(embedding_size, 384),
            nn.GELU(),
            nn.Linear(384, num_classes)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_classes,num_classes, kernel_size=2,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(num_classes,num_classes, kernel_size=2,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(num_classes,num_classes, kernel_size=2,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(num_classes,num_classes,kernel_size=2, stride=2)
        )

    def forward(self,x):   
        x = self.encoder_block(x)
        x = self.mlp_head(x)
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.decoder(x)
        return x
    
class VITSegmentor(nn.Module):
    def __init__(self, num_classes=2):
        super(VITSegmentor,self).__init__()

        self.Vit = nn.Sequential(
            PatchEmbedding(),
            TransformerBlock(num_classes=num_classes)
        )

    def forward(self,x):
        x = self.Vit(x)
        return x
    
if __name__ == "__main__":
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device", device)

    # tensorboard
    writer = SummaryWriter(log_dir='runs')

    # data loading
    dataset = ImageDataset()
    dataset.show_image_mask(0)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = VITSegmentor(num_classes=33).to(device)

    # training loop
    loss = CrossEntropyLoss(ignore_index=-1)
    optimizer = Adam(model.parameters(),lr=0.00007)

    for epoch in range(10000):
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss_value = loss(outputs,targets)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        writer.add_scalar("Loss / train", loss_value.item(), epoch)

        if epoch % 500 == 0:
            print("Epoch ", epoch, " Loss ", loss_value.item())
        
            torch.save(model.state_dict(),'model\\vit_segmentation_epoch_'+str(epoch)+'.pth')
