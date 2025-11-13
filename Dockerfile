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
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# Create a non-root user whose UID/GID can match the host
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} -o app && \
    useradd -m -u ${UID} -g ${GID} -o -s /bin/bash app

# Copy the rest of the project files/
WORKDIR /n_body_approx

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_VERSION=12.6

USER app

CMD ["/bin/bash"]
