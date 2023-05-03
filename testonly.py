import os.path
from copy import deepcopy

import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from convolutional_nn_model import LBT_Custom_Dataset, plot_losses
from pretrain_ds import C5MNIST
from utils import load_dataframe
import torch.optim as optim
from sklearn.model_selection import train_test_split

#num_epochs = 50
#num_pretrain_epochs = 3



def loadtestmodel(model_path, test_loader):
    train_losses = []
    val_losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("LTest - Can we use CUDA: " + str(torch.cuda.is_available()))

    if type(model_path) == type(""):
        print(f"LTest - Loading model from path:  {model_path}")
        model = torch.load(model_path)
    else:
        print(f"LTest - Using model object of class:  {type(model_path)}")
        model = model_path
    model = model.to(device).to(torch.float64)
    running_loss = 0.0
    best_val_loss = 9999.99
    best_m = None
    best_e = 0

    # Set model to evaluation mode
    model.eval()

    criterion = nn.MSELoss()

    # Initialize total loss and number of samples
    total_loss = 0
    n_samples = 0

    # Iterate over test data
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['img'].to(device)
            pxs = batch['px'].reshape(-1, 1).to(device)
            pys = batch['py'].reshape(-1, 1).to(device)
            targets = batch['label'].to(device)

            # Make predictions
            outputs = model(imgs, pxs, pys)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Update total loss and number of samples
            total_loss += loss.item() * len(batch)
            n_samples += len(batch)
            mean_loss = total_loss / n_samples


        print(f"LTest - Test mean squared error: {mean_loss:.4f}")
    return mean_loss