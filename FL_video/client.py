import flwr as fl
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple
import argparse


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Neural Network Model
class Net(nn.Module):
    def __init__(self, feature_size: int = 8) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(feature_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)  # Output a single value for each window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


# Function to load dataset and prepare data for sliding window
def load_csv_service_dataset(window_size: int=100):
    # Load and concatenate both service datasets
    service_a = pd.read_csv('Video_ue_metrics.csv')  # Replace with actual filenames
    service_b = pd.read_csv('ping_ue_metrics.csv')

    # Assign service type labels
    service_a['service_type'] = 0
    service_b['service_type'] = 1

    # Combine the datasets
    data = pd.concat([service_a, service_b])

    dl_brate = data['dl_brate'].values
    ul_brate = data['ul_brate'].values

    # Prepare sliding window features and labels
    features = []
    labels = []

    for i in range(0, len(dl_brate) - window_size + 1, window_size):
        window_dl = dl_brate[i:i + window_size]
        window_ul = ul_brate[i:i + window_size]
        window_label = data['service_type'][i:i + window_size]

        # Check if the window has consistent labels
        if len(window_label.unique()) == 1:
            mean_dl = np.mean(window_dl)
            mean_ul = np.mean(window_ul)
            std_dl = np.std(window_dl)
            std_ul = np.std(window_ul)
            min_val_dl = np.min(window_dl)
            min_val_ul = np.min(window_ul)
            max_val_dl = np.max(window_dl)
            max_val_ul = np.max(window_ul)

            # Append features and corresponding label
            features.append([mean_dl, std_dl, min_val_dl, max_val_dl, mean_ul, std_ul, min_val_ul, max_val_ul])
            labels.append(window_label.iloc[0])

    # Convert to DataFrame for features and array for labels
    X = pd.DataFrame(features, columns=['mean_dl', 'std_dl', 'min_val_dl', 'max_val_dl', 'mean_ul', 'std_ul', 'min_val_ul', 'max_val_ul'])
    y = np.array(labels)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# Train function for model
def train(net, X_train, y_train, epochs: int, verbose=False):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(len(X_train)):
            features, targets = torch.tensor(X_train[i]).float().to(DEVICE), torch.tensor(y_train[i]).float().to(DEVICE)

            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        running_loss /= len(X_train)

        if verbose:
            print(f"Epoch {epoch + 1}: train loss {running_loss:.6f}")


# Test function for model evaluation
def test(net, X_test, y_test):
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    net.eval()

    with torch.no_grad():
        for i in range(len(X_test)):
            features, targets = torch.tensor(X_test[i]).float().to(DEVICE), torch.tensor(y_test[i]).float().to(DEVICE)
            outputs = net(features)
            loss = criterion(outputs, targets.view(-1, 1))
            total_loss += loss.item()

    total_loss /= len(X_test)
    return total_loss


# Flower Client for Federated Learning
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, X_train, X_test, y_train, y_test, client_ID):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.client_ID = client_ID
        feature_size = X_train.shape[1]  # Feature size is the number of columns in X_train
        self.model = Net(feature_size=feature_size).to(DEVICE)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.X_train, self.y_train, epochs=1)
        torch.save(self.model.state_dict(), f"client_{self.client_ID}_model.pth")
        return self.get_parameters(config={}), len(self.X_train), {}

    def evaluate(self, parameters, config) -> Tuple[float, int, dict]:
        self.set_parameters(parameters)
        test_loss = test(self.model, self.X_test, self.y_test)
        return float(test_loss), len(self.X_test), {"loss": test_loss}

def load_model_for_prediction(model_path: str, input_data: np.ndarray):
    feature_size = input_data.shape[1]
    model = Net(feature_size=feature_size).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    input_tensor = torch.tensor(input_data).float().to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
    return output.cpu().numpy()

def load_last_100_records(filepath: str):
    data = pd.read_csv(filepath)
    
    # Extract the last 100 records
    last_100_records = data.tail(100)
    last_100_records = last_100_records.dropna(subset=['dl_brate', 'ul_brate'])
    # Assuming the relevant features are 'dl_brate' and 'ul_brate'
    dl_brate = last_100_records['dl_brate'].values
   
    ul_brate = last_100_records['ul_brate'].values
 
    print("last 100 records:", dl_brate, ul_brate)
    # Preprocess (e.g., scaling) the features
    features = []
    mean_dl = np.mean(dl_brate)
    std_dl = np.std(dl_brate)
    min_val_dl = np.min(dl_brate)
    max_val_dl = np.max(dl_brate)
    mean_ul = np.mean(ul_brate)
    std_ul = np.std(ul_brate)
    min_val_ul = np.min(ul_brate)
    max_val_ul = np.max(ul_brate)

    features.append([mean_dl, std_dl, min_val_dl, max_val_dl, mean_ul, std_ul, min_val_ul, max_val_ul])
    
    # Convert to DataFrame and scale if necessary
    X = pd.DataFrame(features, columns=['mean_dl', 'std_dl', 'min_val_dl', 'max_val_dl', 'mean_ul', 'std_ul', 'min_val_ul', 'max_val_ul'])
    print("data prediction",X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument(
        "--client-id",
        choices=[0, 1],
        default=0,
        type=int,
        help="Client ID (use 0 or 1 for partitioning)",
    )
    # parser.add_argument(
    #     "--data-path",
    #     type=str,
    #     required=True,
    #     help="Path to the CSV data file.",
    # )

    args = parser.parse_args()

    # Load dataset
    X_train, X_test, y_train, y_test = load_csv_service_dataset(window_size=100)

    # Create the Flower client
    client = FlowerClient(X_train, X_test, y_train, y_test, args.client_id)
    
    # Start the federated learning client
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)


    #prediction
    print("prediction")
   

    X_last_100 = load_last_100_records('Video_ue_metrics.csv')  # Replace with actual CSV path

    # Load the saved model
    prediction = load_model_for_prediction("client_0_model.pth", X_last_100)
    print(f"Prediction for last 100 records: {prediction}")