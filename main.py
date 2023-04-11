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

!!! If processed_data.zip exists in the directory, unzip that and use the dataframes stored in there. !!!

Else unzip 10_Data and 20_Data (the raw data files) and run the code below and it will save the pickled dataframes to the current directory.

Check cnn_model and regression_model to run actual models.
'''


''' 
Use this function if after unzipping 10_Data or 20_Data into current directory
'''
def raw_data_to_dataframe(path, filename_pickle, filename_csv):
    df = load_data(path)
    save_dataframe(df, filename_pickle)
    save_dataframe_as_csv(df, filename_csv)

if __name__ == '__main__':
    #raw_data_to_dataframe('10_data', '10_data.pt', '10_data.csv')


    df_ten = load_data('./10_Data')
    df_twenty = load_data('./20_Data')
    save_dataframe(df_ten, 'processed_data/10_Data.pt')
    save_dataframe(df_ten, 'processed_data/20_Data.pt')




