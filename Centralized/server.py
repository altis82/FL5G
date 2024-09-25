import pandas as pd
import paho.mqtt.client as mqtt
import json
import time
import centralized as centralize

# MQTT Broker details
broker = 'localhost'  # Replace with your broker address
port = 1883
topic1 = 'topic/client1'
topic2 = 'topic/client2'

# Data storage
data = []

# Timing
start_time = None
end_time = None

# Define the total number of messages expected (if known)
expected_message_count = 10000  # Update this to match your expected message count

# Define MQTT callbacks
def on_connect(client, userdata, flags, rc):
    global start_time
    print(f"Connected with result code {rc}")
    
    client.subscribe([(topic1, 0), (topic2, 0)])

def on_message(client, userdata, msg):
    global end_time
    global start_time
    if start_time is None:
        start_time = time.time()  # Start timing when connected

    print(f"Message received on topic {msg.topic}")
    try:
        message = json.loads(msg.payload)
        data.append(message)
        # Check if we have received all the expected messages
        if len(data) >= expected_message_count:
            end_time = time.time()  # End timing when the expected count is reached
            client.disconnect()  # Trigger disconnection
    except json.JSONDecodeError:
        print("Failed to decode message")

def on_disconnect(client, userdata, rc):
    print(f"Disconnected with result code {rc}")
    if end_time is not None:
        print(f"Total time to receive all data: {end_time - start_time:.2f} seconds")
    save_to_csv()
    print("Data saved to received_data.csv")

def save_to_csv():
    df = pd.DataFrame(data)
    df.to_csv('received_data.csv', index=False)
    print("Data saved to received_data.csv")

# Create and configure MQTT Client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect

client.connect(broker, port, 60)

# Start the MQTT client loop
client.loop_start()

# Keep the script running until manually stopped
try:
    while client.is_connected():
        time.sleep(1)  # Sleep briefly to avoid high CPU usage
except KeyboardInterrupt:
    print("Interrupted by user")
    client.disconnect()

#Centrallized training
centralize.centrallized_training()
if end_time is not None:
        print(f"Total time to train all data: {end_time - start_time:.2f} seconds")