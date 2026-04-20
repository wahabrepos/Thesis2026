#!/bin/bash

# =============================================================================
# Self-MedRAG Automated Setup Script for NVIDIA Jetson Orin Nano
# =============================================================================

set -e  # Exit on error

echo "======================================================================="
echo "Self-MedRAG Production Setup for Jetson Orin Nano"
echo "======================================================================="
echo ""

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "WARNING: Not detected as Jetson platform"
    echo "This script is optimized for Jetson. Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# Step 1: System dependencies
echo "Step 1: Installing system dependencies..."
sudo add-apt-repository -y universe
sudo apt-get update
sudo apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    curl

# Install libhdf5-dev if available, otherwise h5py will use its own bundled HDF5
if apt-cache show libhdf5-dev &>/dev/null; then
    sudo apt-get install -y libhdf5-dev
else
    echo "libhdf5-dev not found in repos — h5py will be installed with bundled HDF5 via pip"
fi

# Step 2: Create virtual environment
echo "Step 2: Creating Python virtual environment..."
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
python3 -m virtualenv venv
source venv/bin/activate

# Step 3: Install PyTorch for Jetson
echo "Step 3: Installing PyTorch for Jetson..."
echo "Downloading PyTorch wheel for JetPack 6.0..."
wget -q https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl
pip install torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl
rm torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl

# Step 4: Install dependencies
echo "Step 4: Installing Python dependencies..."
pip install -r requirements-jetson.txt

# Step 5: Setup environment
echo "Step 5: Setting up environment..."
if [ ! -f .env ]; then
    cp .env.template .env
    echo "Created .env file from template"
    echo "Please edit .env to configure paths and API keys"
fi

# Required for Jetson Tegra CUDA allocator compatibility
if ! grep -q "PYTORCH_CUDA_ALLOC_CONF" .env; then
    echo "PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync" >> .env
    echo "Added PYTORCH_CUDA_ALLOC_CONF to .env"
fi

# Step 6: Create directories
echo "Step 6: Creating project directories..."
mkdir -p data/datasets data/corpus models results checkpoints logs cache

# Step 7: Download datasets (optional)
echo "Step 7: Download datasets? (y/n)"
read -r response
if [ "$response" = "y" ]; then
    python scripts/download_datasets.py --dataset all
    python scripts/download_corpus.py --max-docs 50000
fi

# Step 8: Verify installation
echo "Step 8: Verifying installation..."
python scripts/system_check.py

echo ""
echo "======================================================================="
echo "Setup Complete!"
echo "======================================================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Run test: python scripts/run_experiment.py --test-mode"
echo "3. Run experiment: python scripts/run_experiment.py --dataset medqa"
echo ""
echo "To activate environment in the future:"
echo "  source venv/bin/activate"
echo ""
