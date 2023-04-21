import argparse
import os
import sys
from os import listdir
from os.path import isfile, join
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
import torch.nn.functional as F

from convolutional_nn_model import CNN
from models import ComboViT, Resnet18CombineFF
from train import train
from utils import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

''' 
main.py is currently just being used to generate the dataframe. Check 
cnn_model and regression_model to run actual code.
'''

if __name__ == '__main__':
    # df_ten = load_data('./10_Data')
    # df_twenty = load_data('./20_Data')
    # save_dataframe(df_ten, 'processed_data/10_Data.pt')
    # save_dataframe(df_ten, 'processed_data/20_Data.pt')

    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')

    parser.add_argument('--data_path', default=".")
    parser.add_argument('--data_type', default="20_data")
    parser.add_argument('--img_dim', default=100)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--lr', default=1e-5)
    parser.add_argument('--model', default="CNN")
    parser.add_argument('--pretrain', default=None)
    args = parser.parse_args()

    if not os.path.exists(args.data_path+"/10_Data.pt") and not os.path.exists(args.data_path+"/20_Data.pt"):
        unzip_raw_data_10_20(args.data_path)

    df = load_dataframe(args.data_type+".pt")
    save_dataframe_as_csv(df, args.data_path[:7]+'.csv')


    if args.model == "CNN":
        model = CNN()
    elif args.model == "ComboViT":
        model = ComboViT()
    elif args.model == "Resnet18Ensemble":
        model = Resnet18Ensemble()
    elif args.model == "CNNEnsemble":
        model = CNNEnsemble()
    elif args.model == "Resnet18CombineFF":
        model = Resnet18CombineFF()
    else:
        raise("Error - Model not defined")

    print(f"Using model class: {type(model)}")

    lr = float(args.lr)
    hw = int(args.img_dim)
    batch_size = int(args.batch_size)
    dopretrain = str(args.pretrain)

    train(model, df, lr, hw, batch_size, dopretrain)


