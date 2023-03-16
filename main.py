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

# Hyper-parameters
num_epochs = 5
batch_size = 4
learning_rate = 0.001

if __name__ == '__main__':
    p = load_data("10_Data")
    save_dataframe(p, "10_data.pt")
    test_df = load_dataframe('10_data.pt')
    print(test_df)

