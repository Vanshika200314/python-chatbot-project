# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install Git and Git LFS so we can download the large model files
RUN apt-get update && apt-get install -y git git-lfs && git-lfs install

# Copy the file that lists the requirements
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# The command to run the app will be taken from the Procfile / Render Start Command
# No CMD or EXPOSE needed here for Render