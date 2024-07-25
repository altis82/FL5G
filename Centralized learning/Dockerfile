# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install Mosquitto and dependencies
RUN apt-get update && \
    apt-get install -y mosquitto mosquitto-clients && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /usr/src/app
COPY . .
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -q flwr[simulation] flwr_datasets[vision] torch torchvision matplotlib
# Set environment variable to prevent matplotlib from trying to use a GUI backend
ENV MPLBACKEND=agg

