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
import torch.optim as optim
from torchvision import datasets
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms as tf
from sklearn.model_selection import train_test_split
from utils import *
import matplotlib.pyplot as plt

Learning_Rate=1e-5 # Learning_Rate: is the step size of the gradient descent during the training.
width=height=100   # Width and height are the dimensions of the image used for training. Images are static at 100x100
batchSize=1        # batchSize: is the number of images that will be used for each iteration of the training.

'''
10_Data has 675 Data Samples
'''

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 25 * 25 + 2, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x, px, py):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 25 * 25)
        x = torch.cat((x, px, py), dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# define a custom dataset to load the data
class LBT_Custom_Dataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe[['p_x', 'p_y', 'data_img']]
        self.labels = dataframe[['d_x', 'd_y', 'd_z', 't_x', 't_y']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data.iloc[idx]['data_img']
        img = np.expand_dims(img, axis=0)
        px = self.data.iloc[idx]['p_x']
        py = self.data.iloc[idx]['p_y']
        label = self.labels.iloc[idx].values.astype(np.float64)
        sample = {'img': img, 'px': px, 'py': py, 'label': label}
        return sample

def plot_losses(train_losses, validation_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    train_losses = []
    val_losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Can we use CUDA: " + str(torch.cuda.is_available()))
    df = load_dataframe("10_data.pt")


    train_val, test = train_test_split(df, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.25, random_state=42)

    train_dataset = LBT_Custom_Dataset(train)
    val_dataset = LBT_Custom_Dataset(val)
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # Create a validation DataLoader

    model = CNN()
    model = model.double().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 200
    running_loss = 0.0

    for epoch in range(num_epochs):
        for i , batch in enumerate(dataloader,0):
            imgs = batch['img'].to(device)
            pxs = batch['px'].reshape(-1, 1).to(device)
            pys = batch['py'].reshape(-1, 1).to(device)
            inputs = imgs
            outputs = model(inputs, pxs, pys)
            labels = batch['label'].to(device)
            loss = nn.functional.mse_loss(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())


        val_loss = 0.0
        val_batch_count = 0
        with torch.no_grad():
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
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item()}, Validation Loss: {avg_val_loss}')


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
