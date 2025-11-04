#!/bin/bash

# Check if IP address is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <instance-ip>"
    exit 1
fi

INSTANCE_IP=$1
SSH_KEY_PATH="$HOME/.ssh/lambda_deploy_key"

# Check if lambda_deploy_key exists
if [ ! -f "$SSH_KEY_PATH" ]; then
    echo "Error: lambda_deploy_key not found at $SSH_KEY_PATH"
    echo "Please generate it first using:"
    echo "ssh-keygen -t ed25519 -f ~/.ssh/lambda_deploy_key -N \"\""
    exit 1
fi

echo "Setting up SSH keys for Lambda instance at $INSTANCE_IP..."

# Copy the SSH key
echo "Copying SSH key..."
scp "$SSH_KEY_PATH" "ubuntu@$INSTANCE_IP:~/.ssh/"

# Set correct permissions and create SSH config
echo "Configuring SSH..."
ssh -o StrictHostKeyChecking=no "ubuntu@$INSTANCE_IP" "chmod 600 ~/.ssh/lambda_deploy_key && \
    echo 'Host github.com
    IdentityFile ~/.ssh/lambda_deploy_key' >> ~/.ssh/config && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts"

# Add user to docker group and create necessary directories
echo "Setting up permissions and directories..."
ssh "ubuntu@$INSTANCE_IP" "sudo usermod -aG docker ubuntu"

# Need to reconnect for docker group to take effect
echo "Reconnecting to apply docker group changes..."
ssh "ubuntu@$INSTANCE_IP" "exit"

# Clone repository
echo "Cloning repository..."
ssh "ubuntu@$INSTANCE_IP" "git clone git@github.com:Simona-Biosystems/n_body_approx.git || true"

echo "Setup complete!"
