# Dockerfile to launch Python code with TensorFlow and Keras on GPU
# To launch the container, run the following command:
# docker run -dit --gpus all -v /home/user:/app --name container_name image_name
# Where:
# -dit: Runs the container in interactive mode and in the background
# --gpus all: Uses the GPU
# -v /home/user:/app: Mounts the /home/user directory in the container's /app directory
# --name container_name: Assigns a name to the container
# image_name: Name of the image to be used

FROM tensorflow/tensorflow:latest-gpu

# Install additional packages
RUN apt-get update && apt-get install -y \
    git \
    vim \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*


# Install additional python packages
RUN pip install --upgrade pip

COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Set up a working directory
WORKDIR /app

# Expose ports (if necessary, adjust according to your application)
EXPOSE 8895

# Run the application
# Run interactive shell
CMD ["/bin/bash"]


