from centralized import load_csv_dataset, load_model, train, test
import flwr as fl
from collections import OrderedDict
import torch


def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# def get_parameters(net) -> List[np.ndarray]:
#     return [val.cpu().numpy() for _, val in net.state_dict().items()]


# Load the dataset
trainloaders, testloader = load_csv_dataset('src_ue.csv')

# Determine input size from the dataset (number of features)
input_size = next(iter(trainloaders[0]))[0].shape[1]

# Initialize the network
net = load_model(input_size)

# # Train the model
# train(net, trainloaders, epochs=10, verbose=True)

# # Test the model
# test_loss = test(net, testloader)
# print(f"Test Loss: {test_loss}")


# net = load_model()
# trainloaders, valloader, testloader = load_csv_dataset('received_data.csv')
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        #def train(net, trainloaders, epochs: int, verbose=False):
        train(self.net, self.trainloader, epochs=1)
        print("training")
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

fl.client.start_numpy_client(
    server_address ="127.0.0.1:8080",
    client=FlowerClient(net, trainloaders[0], testloader),
    
)