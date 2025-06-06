# Basic import
import torch
from torch import nn
import datetime
import pandas as pd
import os
from PIL import Image

model = nn.Sequential(
    nn.Conv2d(in_channels=3,out_channels=24,kernel_size=3,stride=1,padding=2),
    nn.ReLU(),
    nn.MaxPool2d(2,2)
    nn.Flatten(),
    nn.Linear(10*14*14,10)
)