from copy import deepcopy
from typing import Iterator

import torch
import torch.nn as nn
from torch._functorch.make_functional import combine_state_for_ensemble
from torch.nn import Conv2d, Parameter
from torchvision.models import resnet18, resnet50
from transformers import ViTConfig, ViTModel


class ComboViT(nn.Module):
    def __init__(self):
        super(ComboViT, self).__init__()
        self.configuration = ViTConfig()
        self.configuration.image_size = 100
        self.configuration.num_channels = 3
        self.configuration.hidden_size = 64#768
        self.configuration.num_hidden_layers = 6#12
        self.configuration.num_attention_heads = 8#12
        self.configuration.intermediate_size = 1024#3072

        self.vit_model = ViTModel(self.configuration)
        self.reg_layer = nn.Sequential(
                nn.Linear(2368, 256),
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
        self.resnet_model.conv1 = Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False).to(torch.float32) #replacing channels 3 -> 1

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

class Resnet18CombineAtt(nn.Module):
    def __init__(self):
        super(Resnet18CombineAtt, self).__init__()
        tmp_size = 118
        sec_size = tmp_size + 2
        self.resnet_model = resnet18(num_classes=tmp_size)
        self.resnet_model.conv1 = Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False).to(torch.float32) #replacing channels 3 -> 1

        self.attn_lyr = torch.nn.MultiheadAttention(embed_dim=sec_size, num_heads=12)

        self.reg_layer = nn.Sequential()
        self.combine_layer = nn.Sequential(
                nn.Linear(sec_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 5),
        )
    def forward(self, ximg, px, py):
        tmpx = self.resnet_model(ximg)
        tmpx = self.reg_layer(tmpx)
        c_tmpx = torch.cat((tmpx,px,py), dim=1)
        c_tmpx = self.attn_lyr(c_tmpx, c_tmpx, c_tmpx)
        tmpx = self.combine_layer(c_tmpx[0])
        return tmpx

class Resnet18StretchAtt(nn.Module):
    def __init__(self):
        super(Resnet18StretchAtt, self).__init__()
        tmp_size = 114
        sec_size = tmp_size + 6
        self.resnet_model = resnet18(num_classes=tmp_size)
        self.resnet_model.conv1 = Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False).to(torch.float32) #replacing channels 3 -> 1


        self.attn_lyr = torch.nn.MultiheadAttention(embed_dim=sec_size, num_heads=12)

        self.reg_layer = nn.Sequential()
        self.combine_layer = nn.Sequential(
                nn.Linear(sec_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 5),
        )

        self.stretch_layer = nn.Sequential(
                nn.Linear(6, 12),
                nn.ReLU(),
                nn.Linear(12, 6),
        )
    def forward(self, ximg, px, py):
        tmpx = self.resnet_model(ximg)
        tmpx = self.reg_layer(tmpx)
        ps = self.stretch(px,py)
        c_tmpx = torch.cat((tmpx,ps), dim=1)
        c_tmpx = self.attn_lyr(c_tmpx, c_tmpx, c_tmpx)
        tmpx = self.combine_layer(c_tmpx[0])
        return tmpx

    def stretch(self,px,py):
        #px = torch.sigmoid(px)
        #py = torch.sigmoid(py)
        #temp = [px, py, (px**2), (py**2), (torch.log(px)), (torch.log(py)), (torch.e**px), (torch.e**py), (1/px), (1/py)]
        #temp = [px, py, (px**2), (py**2), (torch.log(px)), (torch.log(py)), (1/px), (1/py)]
        #temp = [px, py]
        temp = [px, py, (1/px), (1/py), ((px**2)**(1/4)), ((py**2)**(1/4))]
        temp = torch.concat(temp, dim=1)
        #temp = self.stretch_layer(temp)
        return temp

class Resnet50CombineFF(nn.Module):
    def __init__(self):
        super(Resnet50CombineFF, self).__init__()
        self.resnet_model = resnet50(num_classes=64)
        self.resnet_model.conv1 = Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False).to(torch.float32) #replacing channels 3 -> 1


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

class FFEnsemble(nn.Module):
    def __init__(self, num_models=20):
        super(FFEnsemble, self).__init__()
        self.resnet_model = resnet18(num_classes=64)
        self.resnet_model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                         bias=False).to(torch.float32)  # replacing channels 3 -> 1
        combine_arch = nn.Sequential(
            nn.Linear(66, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        combine_arch.to(device)

        self.num_models = num_models
        #self.archs = []
        # for i in range(self.num_models):
        #     self.archs.append(deepcopy(combine_arch).to(device).to(torch.float64))
        self.archs = create_ensemble(combine_arch, self.num_models)

    def forward(self, ximg, px, py):
        tmpx = self.resnet_model(ximg)
        c_tmpx = torch.cat((tmpx, px, py), dim=1)

        all_res = None
        for i in range(self.num_models):
            temp_out = self.archs[i](c_tmpx)
            if all_res is None:
                all_res = temp_out
            else:
                all_res = all_res + temp_out


        return all_res / self.num_models



class Resnet18AttnEnsemble(nn.Module):
    def __init__(self):
        super(Resnet18AttnEnsemble, self).__init__()
        tmp_size = 66
        sec_size = tmp_size + 6
        combo_out = 32
        self.num_models = 10
        att_size = combo_out * self.num_models

        self.resnet_model = resnet18(num_classes=tmp_size)
        self.resnet_model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(
            torch.float32)  # replacing channels 3 -> 1

        self.attn_lyr = torch.nn.MultiheadAttention(embed_dim=att_size, num_heads=10)

        self.combine_layer = nn.Sequential(
            nn.Linear(sec_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, combo_out),
        )

        self.final_layer = nn.Sequential(
            nn.Linear(att_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )

        # for i in range(self.num_models):
        #     self.archs.append(deepcopy(self.combine_layer))
        self.archs = create_ensemble(self.combine_layer, self.num_models)


    def forward(self, ximg, px, py):
        tmpx = self.resnet_model(ximg)
        ps = self.stretch(px, py)
        c_tmpx = torch.cat((tmpx, ps), dim=1)

        all_res = []
        for i in range(self.num_models):
            temp_out = self.archs[i](c_tmpx)
            all_res.append(temp_out)
        ens_res = torch.cat(all_res, dim=1)


        c_tmpx = self.attn_lyr(ens_res, ens_res, ens_res)
        tmpx = self.final_layer(c_tmpx[0])
        return tmpx

    def stretch(self, px, py):
        temp = [px, py, (1 / px), (1 / py), ((px ** 2) ** (1 / 4)), ((py ** 2) ** (1 / 4))]
        temp = torch.concat(temp, dim=1)
        return temp


class Resnet18AvgEnsemble(nn.Module):
    def __init__(self):
        super(Resnet18AvgEnsemble, self).__init__()
        tmp_size = 66
        sec_size = tmp_size + 6
        combo_out = 32
        self.num_models = 5
        att_size = combo_out * self.num_models

        self.resnet_model = resnet18(num_classes=tmp_size)
        self.resnet_model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(
            torch.float32)  # replacing channels 3 -> 1

        self.attn_lyr = torch.nn.MultiheadAttention(embed_dim=att_size, num_heads=10)

        self.combine_layer = nn.Sequential(
            nn.Linear(sec_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, combo_out),
        )

        self.final_layer = nn.Sequential(
            nn.Linear(combo_out, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )

        # for i in range(self.num_models):
        #     self.archs.append(deepcopy(self.combine_layer))
        self.archs = create_ensemble(self.combine_layer, self.num_models)

    def forward(self, ximg, px, py):
        tmpx = self.resnet_model(ximg)
        ps = self.stretch(px, py)
        c_tmpx = torch.cat((tmpx, ps), dim=1)

        all_res = []
        for i in range(self.num_models):
            temp_out = self.archs[i](c_tmpx)
            all_res.append(temp_out)
        #ens_res = torch.cat(all_res, dim=1)
        #c_tmpx = self.attn_lyr(ens_res, ens_res, ens_res)
        #tmpx = self.final_layer(c_tmpx[0])
        #return tmpx
        temp = torch.mean(torch.stack(all_res, dim=2), dim=2)
        temp = self.final_layer(temp)
        return temp
    def stretch(self, px, py):
        temp = [px, py, (1 / px), (1 / py), ((px ** 2) ** (1 / 4)), ((py ** 2) ** (1 / 4))]
        temp = torch.concat(temp, dim=1)
        return temp


def create_ensemble(architecture, num_models):
    archs = []
    for i in range(num_models):
        temp_model = deepcopy(architecture)
        for layer in temp_model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        archs.append(temp_model)
    return archs