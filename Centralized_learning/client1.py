import pandas as pd
import paho.mqtt.client as mqtt
import json
import os


current_folder = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (one level up)

config_file_path = os.path.join('config.json')
print(config_file_path)

with open(config_file_path, 'r') as f:
    config = json.load(f)

mqtt_config = config['mqtt']
topics_config = config['topics']['publish'][0] #0 for client1

print (topics_config)
# Load the CSV file
csv_file = config["csv"]["client1"]
data = pd.read_csv(csv_file)

# MQTT Broker details
broker = config["mqtt"]["broker"]  # Replace with your broker address
port = config["mqtt"]["port"] 

# Define MQTT callbacks
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")

def on_publish(client, userdata, mid):
    print(f"Message published with mid: {mid}")

# MQTT Client 1
client1 = mqtt.Client()
client1.on_connect = on_connect
client1.on_publish = on_publish
client1.connect(broker, port, 60)



# Hardcode: Publish data from Client 1 (rows 1-5000) 
# This should be replaced when we have csv file for according to each UE

for index, row in data.iloc[0:5000].iterrows():
    message = row.to_json()  # Or any other serialization format    
    client1.publish(topics_config, message)
# Loop to keep the clients connected
client1.loop_start()
# Stop the clients after publishing is done
client1.loop_stop()
client1.disconnect()

