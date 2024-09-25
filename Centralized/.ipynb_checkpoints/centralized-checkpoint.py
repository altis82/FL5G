import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import Dataset, DataLoader
from PIL import ExifTags

import flwr as fl
from flwr.common import Metrics
from flwr_datasets import FederatedDataset

import torch.optim as optim
import torch.nn as nn

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

NUM_CLIENTS = 10
BATCH_SIZE = 32

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.feature_columns = [col for col in dataframe.columns if col != 'targetTput' and col != 'measTimeStampRf']
        self.target_column = 'targetTput'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = row[self.feature_columns].values.astype(np.float32)
        target = row[self.target_column].astype(np.float32)
        return features, target

def load_csv_dataset(filepath):
    df = pd.read_csv(filepath)

    # Identify categorical columns (columns with dtype object)
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    # Remove 'measTimeStampRf' from categorical columns as it's not relevant for one-hot encoding
    if 'measTimeStampRf' in categorical_columns:
        categorical_columns.remove('measTimeStampRf')

    # One-hot encode categorical columns
    ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' to avoid multicollinearity
    ohe_features = ohe.fit_transform(df[categorical_columns])
    ohe_feature_names = ohe.get_feature_names_out(categorical_columns)

    # Create a new DataFrame with one-hot encoded features
    ohe_df = pd.DataFrame(ohe_features, columns=ohe_feature_names)

    # Drop original categorical columns and 'measTimeStampRf', then concatenate with the one-hot encoded features
    df.drop(columns=categorical_columns + ['measTimeStampRf'], inplace=True)
    df = pd.concat([df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)

    # Split dataset into train, validation, and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Split train set into NUM_CLIENTS partitions
    train_partitions = np.array_split(train_df, NUM_CLIENTS)

    trainloaders = [DataLoader(CustomDataset(partition), batch_size=BATCH_SIZE, shuffle=True) for partition in train_partitions]
    testloader = DataLoader(CustomDataset(test_df), batch_size=BATCH_SIZE, shuffle=False)

    return trainloaders, testloader

class Net(nn.Module):
    def __init__(self, input_size: int) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train(net, trainloaders, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for trainloader in trainloaders:  # Iterate over each client's data
            for features, targets in trainloader:
                features, targets = features.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = net(features)
                loss = criterion(outputs, targets.view(-1, 1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        running_loss /= len(trainloaders) * len(trainloader.dataset)
        if verbose:
            print(f"Epoch {epoch+1}: train loss {running_loss}")

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    net.eval()
    with torch.no_grad():
        for features, targets in testloader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            outputs = net(features)
            loss = criterion(outputs, targets.view(-1, 1))
            total_loss += loss.item()
    total_loss /= len(testloader.dataset)
    return total_loss

def load_model():
    return Net().to(DEVICE)

def centrallized_training():
    # Load the dataset
    trainloaders, testloader = load_csv_dataset('src_ue.csv')

    # Determine input size from the dataset (number of features)
    input_size = next(iter(trainloaders[0]))[0].shape[1]

    # Initialize the network
    net = Net(input_size).to(DEVICE)

    # Train the model
    train(net, trainloaders, epochs=10, verbose=True)

    # Test the model
    test_loss = test(net, testloader)
    print(f"Test Loss: {test_loss}")
