# Basic import
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from model_training import ConvNet

def load_model():
    model = ConvNet(num_classes=3)
    model.load_state_dict(torch.load("ConvNet.pth"))

    model.eval()

    dummy_input = torch.randn(1,1,256,256)

    writer = SummaryWriter('runs/model_architecture')
    writer.add_graph(model,dummy_input)
    writer.close()

    return model

load_model()