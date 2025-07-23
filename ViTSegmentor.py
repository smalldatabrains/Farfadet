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

class ImageDataset():
    def __init__(self):
        self.ds=load_dataset('parquet', data_files='data\\data\\train.parquet',split='train')
        print("Classes are : ")
        print(self.ds.info.features)
        self.transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.ToTensor(),
        ])

    def get_split_names(self):
        return get_dataset_split_names(self.ds)
    
    def __len__(self)
        return len(self.ds)
    
    def __getitem__(self, idx):
        image=self.ds[idx]["image"].convert("RGB")
        mask =self.ds[idx]["annotated"]

        image = self.transform(image)
        mask = transforms.Resize((28,28), interpolation=transforms.InterpolationMode.NEAREST)(mask)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask
    
    def show_image_mask(self, idx):
        image = self.ds[idx]["image"]
        mask = self.ds[idx]["annotated"]
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image)
        axs[0].set_title('Image')
        axs[0].axis('off')

        axs[1].imshow(mask)
        axs[1].set_title('Mask')
        axs[1].axis('off')

        plt.show()

# we could also design our own Dataloader class, min requirements are : __iter__, __next__ and __len__
