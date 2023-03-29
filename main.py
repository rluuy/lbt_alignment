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
main.py is currently just being used to generate the dataframe. Check 
cnn_model and regression_model to run actual code.
'''

if __name__ == '__main__':
    df = load_dataframe('10_data.pt')
    save_dataframe_as_csv(df, '10_data.csv')



