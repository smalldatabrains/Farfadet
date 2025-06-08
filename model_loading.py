# Basic import
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from model_training import ConvNet

def load_model():
    model = ConvNet()
    model.load_state_dict(torch.load("UNET.pth"))

    model.eval()

    dummy_input = torch.randn(1,1,28,28)

    writer = SummaryWriter('runs/model_architecture')
    writer.add_graph(model,dummy_input)
    writer.close()

    return model 