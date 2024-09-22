import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the neural network model
class Net(nn.Module):
    def __init__(self, feature_size: int=8) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(feature_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# Function to load data and clean NaN records
def load_and_clean_data(filepath: str):
    # Load the data
    data = pd.read_csv(filepath)

    # Drop rows with NaN values
    clean_data = data.dropna()

    # Select the last 100 records
    last_100_records = clean_data.tail(100)

    return last_100_records

# Function to extract features from the last 100 records
def extract_features(last_100_records):
    dl_brate_last_100 = last_100_records['dl_brate'].values
    ul_brate_last_100 = last_100_records['ul_brate'].values

    # Calculate statistics (mean, std, min, max)
    mean_dl = np.mean(dl_brate_last_100)
    std_dl = np.std(dl_brate_last_100)
    min_dl = np.min(dl_brate_last_100)
    max_dl = np.max(dl_brate_last_100)

    mean_ul = np.mean(ul_brate_last_100)
    std_ul = np.std(ul_brate_last_100)
    min_ul = np.min(ul_brate_last_100)
    max_ul = np.max(ul_brate_last_100)

    # Create feature array
    window_features = np.array([[mean_dl, std_dl, min_dl, max_dl, mean_ul, std_ul, min_ul, max_ul]])

    return window_features

# Function to make predictions using the trained model
def make_prediction(model, features):
    # Convert the features to a torch tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).to(DEVICE)

    # Make a prediction
    model.eval()
    with torch.no_grad():
        prediction = model(features_tensor).cpu().numpy()

    # Convert the prediction probability to a binary class (0 or 1)
    predicted_class = (prediction > 0.5).astype(int)

    return predicted_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client Prediction")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the CSV data file."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved PyTorch model file."
    )
    args = parser.parse_args()

    # Load the cleaned data (last 100 records after dropping NaN values)
    last_100_records = load_and_clean_data(args.data_path)

    # Extract features from the last 100 records
    features = extract_features(last_100_records)

    # Load the saved model
    model = Net()
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model = model.to(DEVICE)

    # Make a prediction
    predicted_class = make_prediction(model, features)

    # Output the predicted service type
    service_type = "A" if predicted_class == 0 else "B"
    print(f"The predicted service type for the last 100 records is: {service_type}")
