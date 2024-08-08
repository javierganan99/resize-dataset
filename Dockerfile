FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive


# Create a working directory
WORKDIR /app

# Copy your application code to the Docker image
COPY . /app