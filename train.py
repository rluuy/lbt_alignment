import os.path
from copy import deepcopy

import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from convolutional_nn_model import LBT_Custom_Dataset, plot_losses
from pretrain_ds import C5MNIST
from testonly import loadtestmodel
from utils import load_dataframe
import torch.optim as optim
from sklearn.model_selection import train_test_split

#num_epochs = 50
#num_pretrain_epochs = 3

def pretrain(model, dopretrain, batch_size, lr=0.001, num_epochs=3, device='cuda'):
    print("Starting Pretraining")
    if dopretrain == "C5MNIST":
        do_dl = False
        if not os.path.exists("./data/mnist"):
            do_dl = True
        tmp_dataset = C5MNIST(root="./data/mnist", download=do_dl)
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

    try:
        model = model.to(device).to(torch.float64)
        if hasattr(model, "archs"):
            [x.to(device).to(torch.float64) for x in model.archs]
    except:
        pass
    optimizer = optim.Adam(model.parameters(), lr=lr)
    running_loss = 0.0
    best_val_loss = 9999.99

    for epoch in range(num_epochs):
        model.train()
        for i, batch in enumerate(dataloader, 0):
            imgs = batch[0].to(device).to(torch.float64)
            pxs = torch.zeros(batch_size,1).to(device)
            pys = torch.zeros(batch_size,1).to(device)
            outputs = model(imgs, pxs, pys)
            labels = batch[1].to(device).to(torch.float64)
            loss = nn.functional.mse_loss(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if torch.sum(torch.isnan(outputs)).item() > 0:
                break

        train_losses.append(loss.item())

        val_loss = 0.0
        val_batch_count = 0
        with torch.no_grad():
            model.eval()
            for i, val_batch in enumerate(val_dataloader, 0):
                val_imgs = val_batch[0].to(device).to(torch.float64)
                val_pxs = torch.zeros(batch_size,1).to(device)
                val_pys = torch.zeros(batch_size,1).to(device)
                val_outputs = model(val_imgs, val_pxs, val_pys)
                val_labels = val_batch[1].to(device).to(torch.float64)
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


def train(model, df, lr, epochs, pretrain_epochs, hw, batch_size, dopretrain=None):
    train_losses = []
    val_losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Can we use CUDA: " + str(torch.cuda.is_available()))

    num_epochs = epochs
    num_pretrain_epochs = pretrain_epochs

    if dopretrain is not None and dopretrain != 'None':
        model = pretrain(model, dopretrain, batch_size, lr=0.002, num_epochs=num_pretrain_epochs, device=device)

    train_val, test = train_test_split(df, test_size=0.20, random_state=42)
    train, val = train_test_split(train_val, test_size=0.20, random_state=42)

    train_dataset = LBT_Custom_Dataset(train)
    val_dataset = LBT_Custom_Dataset(val)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Create a validation DataLoader

    try:
        model = model.to(device).to(torch.float64)
        if hasattr(model, "archs"):
            [x.to(device).to(torch.float64) for x in model.archs]
    except:
        pass
    optimizer = optim.Adam(model.parameters(), lr=lr)
    running_loss = 0.0
    best_val_loss = 9999.99
    best_m = None
    best_e = 0

    for epoch in range(num_epochs):
        model.train()
        for i , batch in enumerate(dataloader,0):
            imgs = batch['img'].to(device).to(torch.float64)
            pxs = batch['px'].reshape(-1, 1).to(device)
            pys = batch['py'].reshape(-1, 1).to(device)
            outputs = model(imgs, pxs, pys)
            labels = batch['label'].to(device).to(torch.float64)
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
                val_imgs = val_batch['img'].to(device).to(torch.float64)
                val_pxs = val_batch['px'].reshape(-1, 1).to(device)
                val_pys = val_batch['py'].reshape(-1, 1).to(device)
                val_inputs = val_imgs
                val_outputs = model(val_inputs, val_pxs, val_pys)
                val_labels = val_batch['label'].to(device).to(torch.float64)
                val_batch_loss = nn.functional.mse_loss(val_outputs, val_labels)
                val_loss += val_batch_loss.item()
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count
        val_losses.append(avg_val_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {str(numpy.mean(train_losses))[:6]}, Validation Loss: {avg_val_loss}')

        m_name = str(type(model)).split(".")[-1:][0][:-2]

        if best_val_loss > avg_val_loss:
            print(f"Updating best model from epoch {best_e} with loss {best_val_loss} to epoch {epoch} with loss {avg_val_loss}")
            best_val_loss = avg_val_loss
            best_m = deepcopy(model)
            best_e = epoch

    torch.save(best_m, f"saved_models/{m_name}_ep{best_e}_valloss{best_val_loss}")

    plot_losses(train_losses, val_losses)
    # Set model to evaluation mode
    model.eval()


    test_dataset = LBT_Custom_Dataset(test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    print(f"Testing best model....")
    best_model_results = loadtestmodel(best_m, test_loader)

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


        print(f"Test FINAL model mean squared error: {mean_loss:.4f}")
        torch.save(model, f"saved_models/TEST_{m_name}_testAcc{mean_loss}")

    return model, mean_loss, epoch, best_m, best_model_results, best_e