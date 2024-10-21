#!/bin/bash

# Check if the script is run as root (user ID 0)
if [ "$(id -u)" -ne 0 ]; then
  echo "This script must be run with sudo or as root" >&2
  exit 1
fi
echo "GPU reset script is running as root."

# Restart ollama service
echo "Restarting ollama.service..."
if ! systemctl restart ollama.service; then
  echo "Error: Failed to restart ollama.service" >&2
  exit 1
fi
echo "ollama.service restarted successfully."

# Notify the start of the GPU reset process
echo "Starting GPU reset..."

# Unload the nvidia_uvm module
if ! rmmod nvidia_uvm; then
  echo "Error: Failed to unload nvidia_uvm module" >&2
  exit 1
fi
echo "nvidia_uvm unloaded."

# Reload the nvidia_uvm module
if ! modprobe nvidia_uvm; then
  echo "Error: Failed to load nvidia_uvm module" >&2
  exit 1
fi
echo "nvidia_uvm reloaded."

# Confirm the module was loaded
if lsmod | grep -q nvidia_uvm; then
  echo "GPU reset done successfully"
else
  echo "Error: nvidia_uvm module not loaded correctly" >&2
  exit 1
fi