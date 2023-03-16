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
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from utils import *

Learning_Rate=1e-5 # Learning_Rate: is the step size of the gradient descent during the training.
width=height=100   # Width and height are the dimensions of the image used for training. All images during the training processes will be resized to this size.
batchSize=1        # batchSize: is the number of images that will be used for each iteration of the training.

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
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe[['p_x', 'p_y', 'data_img']]
        self.labels = dataframe[['d_x', 'd_y', 'd_z', 't_x', 't_y']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data.iloc[idx]['data_img']
        img = np.expand_dims(img, axis=0)  # add channel dimension
        px = self.data.iloc[idx]['p_x']
        py = self.data.iloc[idx]['p_y']
        label = self.labels.iloc[idx].values.astype(np.float64)
        sample = {'img': img, 'px': px, 'py': py, 'label': label}
        return sample




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Can we use CUDA: " + str(torch.cuda.is_available()))
    df = load_dataframe("10_data.pt")
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(train)
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = CNN()
    model = model.double()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 200
    running_loss = 0.0
    writer = SummaryWriter('runs/lbt_test1')
    for epoch in range(num_epochs):
        for i , batch in enumerate(dataloader,0):
            imgs = batch['img']
            pxs = batch['px'].reshape(-1, 1)
            pys = batch['py'].reshape(-1, 1)
            inputs = imgs
            outputs = model(inputs, pxs, pys)
            labels = batch['label']
            loss = nn.functional.mse_loss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # Set model to evaluation mode
    model.eval()

    # Initialize test data loader
    test_dataset = CustomDataset(test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Initialize criterion
    criterion = nn.MSELoss()

    # Initialize total loss and number of samples
    total_loss = 0
    n_samples = 0

    # Iterate over test data
    with torch.no_grad():
        for batch in test_loader:
            # Get inputs and targets
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

    # Calculate mean loss
    mean_loss = total_loss / n_samples

    print(f"Test mean squared error: {mean_loss:.4f}")