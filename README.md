# Simulation using Federated Learning in 5G


In this project there are 2 folders:
- Centralized learning: We run 1 Server to collect all data from 2 clients using mqtt. Each of them will publish 5000 records. Each of client will take 5000 records from the csv file then sends to server. When the server receives all data, it will save into `recive_data.csv` and starts the centralized training. There is a json config file to configure the mqtt broker, and topics.
```
  +-----------------+       +-----------------+       +-----------------+
  |                 |       |                 |       |                 |
  |    Client 1     +------->    MQTT Broker  +------->    Server       |
  |                 |       |                 |       |                 |
  +-----------------+       +-----------------+       +-----------------+
                                      |
                                      |
                          +-------------------+
                          |      Client 2     |
                          +-------------------+
```
Detailed explanation 
### Clients (Client 1 and Client 2):

- Each client reads 5000 records from its respective CSV file.
- Publishes the records to the MQTT broker using MQTT protocol.
- Uses topics configured in a JSON config file.
### MQTT Broker (Server):

- Receives messages from both clients.
- Subscribes to the topics on the MQTT broker.
- Receives data from both clients through the broker.
- Saves the received data into receive_data.csv after collecting all records.
- Initiates the centralized training process.
### JSON Config File:

### Contains configurations for MQTT broker address, topics, and other necessary settings.


- FL learning: We run 1 server to aggregate weights  and distribute to clients. To run Client we input the id, data path file, and target column to predict. 
```
  +-----------------+       +-----------------+       +-----------------+
  |                 |       |                 |       |                 |
  |    Client 1     +------->    Server       <-------+    Client 2     |
  |                 |       |                 |       |                 |
  +-----------------+       +-----------------+       +-----------------+
        |                           ^                          |
        |                           |                          |
        |                           |                          |
        +------------------------------------------------------+
```
### Clients (Client 1 and Client 2):

- Each client is initialized with an ID, data path file, and target column to predict.
- Clients train their local models on their respective data.
- Clients send their local model weights to the server during the training
- Clients receive updated global model weights from the server.
### Server:

- Aggregates the weights from both clients.
- Updates the global model with the aggregated weights.
- Distributes the updated global model weights back to both clients.

## Requirements
- mqtt, conda, flower, matplotlib, pandas
## Setup
- Setup conda
- Activate environment myenv
- Install libs: flower, matplotlib, paho-mqtt, pandas
## DB
Using 5G data `src_user.csv`
### Centralized learning
### FL learning
Client0

`python3 client.py --partition-id 0 -data-path src_user.csv --target-column targetTput`

Client1

`python3 client.py --partition-id 0 -data-path src_user.csv --target-column targetTput`

Server

`python3 server.py`
