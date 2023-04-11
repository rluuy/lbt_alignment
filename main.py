import os
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

from utils import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

''' 
main.py is currently just being used to generate the dataframe. 

Run the code below if 10_data, 20_data does not exist in current directory.
Make sure to unzip the files if they are still compressed then run the code below.

Check cnn_model and regression_model to run actual models.
'''

if __name__ == '__main__':
    df_ten = load_data('./10_Data')
    df_twenty = load_data('./20_Data')
    save_dataframe(df_ten, '10_Data.pt')
    save_dataframe(df_ten, '20_Data.pt')
    df = load_dataframe('10_data.pt')



