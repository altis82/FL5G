# Simulation using Federated Learning in 5G
In this project there are 2 folders:
- Centralized learning: We run 1 Server to collect all data from 2 clients using mqtt. Each of them will publish 5000 records. When the server receives all data, it starts the centralized training. There is a json config file to configure the mqtt broker, and topics.
- FL learning: We run 1 server to aggregate weights  and distribute to clients. To run Client we input the id, data path file, and target column to predict

## Requirements
- mqtt, conda, flower, matplotlib, pandas
## Setup
- Setup conda
- Activate environment myenv
- Install libs: flower, matplotlib, paho-mqtt, pandas
### Centralized learning
### FL learning
Client0
`python3 client.py --partition-id 0 -data-path src_user.csv --target-column targetTput`
Client1
`python3 client.py --partition-id 0 -data-path src_user.csv --target-column targetTput`
Server
`python3 server.py`
