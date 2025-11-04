FROM nvcr.io/nvidia/pytorch:24.12-py3

# system config
RUN apt-get update && apt-get install -y \
    linux-headers-$(uname -r) \
    dkms \
    build-essential \
    pv \
    tmux \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy the rest of the project files/
WORKDIR /n_body_approx
COPY . .

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_VERSION=12.6

CMD ["/bin/bash"]
