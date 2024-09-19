# Use the official Eclipse Mosquitto image from the Docker Hub
FROM eclipse-mosquitto:2.0

# Copy your Mosquitto configuration file (if you have one) into the container
# This line is optional and only needed if you have custom configuration
# COPY mosquitto.conf /mosquitto/config/mosquitto.conf

# Expose the MQTT port (default is 1883)
EXPOSE 1883

# Expose the WebSockets port (default is 9001, uncomment if you use it)
# EXPOSE 9001

# Set the default command to run Mosquitto
CMD ["mosquitto", "-c", "/mosquitto/config/mosquitto.conf"]
