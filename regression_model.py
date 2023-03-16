
import torch
import torch.nn as nn
from utils import *

if __name__ == '__main__':
    test_df = load_dataframe('10_data.pt')
    print(test_df)