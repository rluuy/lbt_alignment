import os.path
from copy import deepcopy

import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from convolutional_nn_model import LBT_Custom_Dataset, plot_losses
from pretrain_ds import CustomPretrain5MNIST
from utils import load_dataframe
import torch.optim as optim
from sklearn.model_selection import train_test_split

num_epochs = 50
num_pretrain_epochs = 1

def pretrain(model, dopretrain, batch_size, lr, num_epochs, device):
    print("Starting Pretraining")
    if dopretrain == "CustomPretrain5MNIST":
        do_dl = False
        if not os.path.exists("./data/mnist"):
            do_dl = True
        tmp_dataset = CustomPretrain5MNIST(root="./data/mnist", download=do_dl)
        c_dataset = tmp_dataset.create_custom()
    else:
        raise Exception("Error! Bad Pretraining choice.")

    train_dataset, val_dataset = torch.utils.data.random_split(c_dataset, [0.8, 0.2])
    #train_dataset = TensorDataset(tensor_x,tensor_y)
    #val_dataset = LBT_Custom_Dataset(val)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Create a validation DataLoader

    train_losses = []
    val_losses = []
    best_pretrained_model = None

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    running_loss = 0.0
    best_val_loss = 9999.99

    for epoch in range(num_epochs):
        model.train()
        for i, batch in enumerate(dataloader, 0):
            imgs = batch[0].to(device)
            pxs = torch.zeros(batch_size,1).to(device)
            pys = torch.zeros(batch_size,1).to(device)
            outputs = model(imgs, pxs, pys)
            labels = batch[1].to(device)
            loss = nn.functional.mse_loss(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())

        val_loss = 0.0
        val_batch_count = 0
        with torch.no_grad():
            model.eval()
            for i, val_batch in enumerate(val_dataloader, 0):
                val_imgs = val_batch[0].to(device)
                val_pxs = torch.zeros(batch_size,1).to(device)
                val_pys = torch.zeros(batch_size,1).to(device)
                val_outputs = model(val_imgs, val_pxs, val_pys)
                val_labels = val_batch[1].to(device)
                val_batch_loss = nn.functional.mse_loss(val_outputs, val_labels)
                val_loss += val_batch_loss.item()
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count
        val_losses.append(avg_val_loss)
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {str(numpy.mean(train_losses))[:6]}, Validation Loss: {avg_val_loss}')

        m_name = str(type(model)).split(".")[-1:][0][:-2]

        if best_val_loss > avg_val_loss:
            best_val_loss = avg_val_loss
            best_pretrained_model = deepcopy(model)

    print(f"Ending Pretraining - Best Val Loss On Custom Dataset: {best_val_loss}")
    return best_pretrained_model


def train(model, df, lr, hw, batch_size, dopretrain=None):
    train_losses = []
    val_losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Can we use CUDA: " + str(torch.cuda.is_available()))


    if dopretrain is not None:
        model = pretrain(model, dopretrain, batch_size, lr, num_epochs=num_pretrain_epochs, device=device)

    train_val, test = train_test_split(df, test_size=0.20, random_state=42)
    train, val = train_test_split(train_val, test_size=0.20, random_state=42)

    train_dataset = LBT_Custom_Dataset(train)
    val_dataset = LBT_Custom_Dataset(val)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Create a validation DataLoader

    model = model.double().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    running_loss = 0.0
    best_val_loss = 9999.99

    for epoch in range(num_epochs):
        model.train()
        for i , batch in enumerate(dataloader,0):
            imgs = batch['img'].to(device)
            pxs = batch['px'].reshape(-1, 1).to(device)
            pys = batch['py'].reshape(-1, 1).to(device)
            outputs = model(imgs, pxs, pys)
            labels = batch['label'].to(device)
            loss = nn.functional.mse_loss(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())

        val_loss = 0.0
        val_batch_count = 0
        with torch.no_grad():
            model.eval()
            for i, val_batch in enumerate(val_dataloader, 0):
                val_imgs = val_batch['img'].to(device)
                val_pxs = val_batch['px'].reshape(-1, 1).to(device)
                val_pys = val_batch['py'].reshape(-1, 1).to(device)
                val_inputs = val_imgs
                val_outputs = model(val_inputs, val_pxs, val_pys)
                val_labels = val_batch['label'].to(device)
                val_batch_loss = nn.functional.mse_loss(val_outputs, val_labels)
                val_loss += val_batch_loss.item()
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count
        val_losses.append(avg_val_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {str(numpy.mean(train_losses))[:6]}, Validation Loss: {avg_val_loss}')

        m_name = str(type(model)).split(".")[-1:][0][:-2]

        if best_val_loss > avg_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model, f"saved_models/{m_name}_ep{epoch}_trloss{str(numpy.mean(train_losses))[:6]}_valloss{avg_val_loss}")

    plot_losses(train_losses, val_losses)
    # Set model to evaluation mode
    model.eval()


    test_dataset = LBT_Custom_Dataset(test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
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


        print(f"Test mean squared error: {mean_loss:.4f}")