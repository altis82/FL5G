import argparse
import warnings
from collections import OrderedDict

from flwr.client import NumPyClient, ClientApp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 2
BATCH_SIZE = 32

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.features = dataframe.drop(columns=["targetTput"]).values
        self.labels = dataframe["targetTput"].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, epochs):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for inputs, targets in tqdm(trainloader, "Training"):
            optimizer.zero_grad()
            outputs = net(inputs.to(DEVICE))
            loss = criterion(outputs, targets.to(DEVICE).view(-1, 1))
            loss.backward()
            optimizer.step()

import csv
def test(net, testloader):
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, "Testing"):
            outputs = net(inputs.to(DEVICE))
            loss = criterion(outputs, targets.to(DEVICE).view(-1, 1))
            total_loss += loss.item()
    
    average_loss = total_loss / len(testloader.dataset)
    print("total_loss", total_loss)
    print("average_loss", average_loss)
    
    # Append loss to a CSV file
    with open('losses.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([average_loss])
    
    return average_loss, None


def load_csv_data(filepath, target_column):
    df = pd.read_csv(filepath)

    # Identify categorical columns (columns with dtype object)
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if 'measTimeStampRf' in categorical_columns:
        categorical_columns.remove('measTimeStampRf')

    # One-hot encode categorical columns
    ohe = OneHotEncoder(sparse_output=False, drop='first')
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

    return trainloaders, testloader, train_df.shape[1] - 1  # Subtract 1 to exclude the target column from input dimensions


class FlowerClient(NumPyClient):
    def __init__(self, trainloader, testloader, input_dim):
        self.trainloader = trainloader
        self.testloader = testloader
        self.net = MLP(input_dim).to(DEVICE)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=1)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, _ = test(self.net, self.testloader)
        return loss, len(self.testloader.dataset), {}


def client_fn(cid: str):
    return FlowerClient(trainloaders[int(cid)], testloader, input_dim).to_client()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition-id",
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
    parser.add_argument(
        "--target-column",
        type=str,
        required=True,
        help="Target column name in the CSV file.",
    )
    args = parser.parse_args()

    trainloaders, testloader, input_dim = load_csv_data(args.data_path, args.target_column)

    app = ClientApp(client_fn=client_fn)

    from flwr.client import start_client
    start_client(
        server_address="127.0.0.1:8080",
        client=client_fn(str(args.partition_id)),
    )
