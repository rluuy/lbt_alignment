import torch
import torch.nn as nn
from torch.nn import Conv2d
from torchvision.models import resnet18
from transformers import ViTConfig, ViTModel


class ComboViT(nn.Module):
    def __init__(self):
        super(ComboViT, self).__init__()
        self.configuration = ViTConfig()
        self.configuration.image_size = 100
        self.configuration.num_channels = 3

        self.vit_model = ViTModel(self.configuration)
        self.reg_layer = nn.Sequential(
                nn.Linear(28416, 256),
        )
        self.combine_layer = nn.Sequential(
                nn.Linear(258, 128),
                nn.ReLU(),
                nn.Linear(128, 5),
                nn.ReLU(),
        )
    def forward(self, ximg, px, py):
        ximg = ximg.repeat(1,3,1,1)
        tmpx = self.vit_model(ximg)
        tmpx = tmpx.last_hidden_state
        tmpx = tmpx.reshape(tmpx.shape[0], -1)
        tmpx = self.reg_layer(tmpx)
        c_tmpx = torch.cat((tmpx,px,py), dim=1)
        tmpx = self.combine_layer(c_tmpx)
        return tmpx




class Resnet18CombineFF(nn.Module):
    def __init__(self):
        super(Resnet18CombineFF, self).__init__()
        self.resnet_model = resnet18(num_classes=64)
        self.resnet_model.conv1 = Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False) #replacing channels 3 -> 1


        self.reg_layer = nn.Sequential()
        self.combine_layer = nn.Sequential(
                nn.Linear(66, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 5),
        )
    def forward(self, ximg, px, py):
        tmpx = self.resnet_model(ximg)
        tmpx = self.reg_layer(tmpx)
        c_tmpx = torch.cat((tmpx,px,py), dim=1)
        tmpx = self.combine_layer(c_tmpx)
        return tmpx