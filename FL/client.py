import flwr as fl
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, Tuple
import argparse


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Neural Network Model
class Net(nn.Module):
    def __init__(self, window_size: int, feature_size: int) -> None:
        super(Net, self).__init__()
        # Input size is now window_size * feature_size (flattened window of features)
        self.fc1 = nn.Linear(window_size * feature_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)  # Output a single value for each window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input: [batch_size, window_size, feature_size] -> [batch_size, window_size * feature_size]
        x = x.view(x.size(0), -1)  # Flatten all but the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Output a single prediction for each window
        return x

# Custom Dataset for Sliding Window
class CustomDataset(Dataset):
    def __init__(self, dataframe, window_size: int):
        self.data = dataframe
        self.window_size = window_size
        self.feature_columns = [col for col in dataframe.columns if col != 'service_type' and col != 'time']
        self.target_column = 'service_type'

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        # Window of features for the input
        features_window = self.data.iloc[idx:idx + self.window_size][self.feature_columns].values.astype(np.float32)
        # Target is the value at the end of the window
        target = self.data.iloc[idx + self.window_size - 1][self.target_column].astype(np.float32)
        return features_window, target

# Function to load dataset and prepare data for sliding window
def load_csv_dataset(filepath, window_size: int):
    df = pd.read_csv(filepath)
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if 'time' in categorical_columns:
        categorical_columns.remove('time')

    ohe = OneHotEncoder(sparse_output=False, drop='first')
    ohe_features = ohe.fit_transform(df[categorical_columns])
    ohe_feature_names = ohe.get_feature_names_out(categorical_columns)

    ohe_df = pd.DataFrame(ohe_features, columns=ohe_feature_names)
    df.drop(columns=categorical_columns + ['time'], inplace=True)
    df = pd.concat([df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)

    # Split into train, validation, and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # Split training data for multiple clients (Federated Learning scenario)
    train_partitions = np.array_split(train_df, 10)  # Adjust NUM_CLIENTS as needed

    trainloaders = [DataLoader(CustomDataset(partition, window_size=window_size), batch_size=32, shuffle=True) for partition in train_partitions]
    testloader = DataLoader(CustomDataset(test_df, window_size=window_size), batch_size=32, shuffle=False)

    return trainloaders, testloader

# Train function for model
def train(net, trainloader, epochs: int, verbose=False):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for features, targets in trainloader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss /= len(trainloader.dataset)
        if verbose:
            print(f"Epoch {epoch+1}: train loss {running_loss}")

# Test function for model evaluation
def test(net, testloader):
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

# Flower Client for Federated Learning
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, testloader):
        self.trainloader = trainloader
        self.testloader = testloader
        window_size = next(iter(trainloader))[0].shape[1]
        feature_size = next(iter(trainloader))[0].shape[2]
        input_size = window_size * feature_size  # Flattened size
        self.model = Net(window_size=window_size, feature_size=feature_size).to(DEVICE)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        # Load the state dict, but handle size mismatches
        model_dict = self.model.state_dict()

        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].size() == v.size():
                model_dict[k] = v
            else:
                print(f"Skipping loading parameters for {k} due to size mismatch.")
        
        # Load the modified state dict
        self.model.load_state_dict(model_dict, strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=1)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config) -> tuple:
        self.set_parameters(parameters)
        test_loss = test(self.model, self.testloader)
        metrics = {"loss": test_loss}
        return float(test_loss), len(self.testloader.dataset), metrics

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--client-id",
        choices=[0, 1],
        default=0,
        type=int,
        help="Partition of the dataset divided into 2 iid partitions created artificially.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the CSV data file.",
    )

    args = parser.parse_args()
    "This is the data file use for all clients"
    trainloaders, testloader = load_csv_dataset(args.data_path,10)

    client_partition = args.client_id

    client_trainloader = trainloaders[client_partition]
    client_testloader = testloader

    client = FlowerClient(client_trainloader, client_testloader)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
    