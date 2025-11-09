# Installation Guide

## Step-by-Step Installation

### Step 1: Install PyTorch First

PyTorch must be installed before torch-geometric dependencies:

```bash
# For CPU-only (faster installation, works everywhere)
pip install torch torchvision torchaudio

# OR for CUDA (if you have NVIDIA GPU)
# Visit https://pytorch.org/get-started/locally/ for the correct command
# Example for CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Install PyTorch Geometric

```bash
# This will automatically install torch-scatter, torch-sparse, etc.
pip install torch-geometric
```

### Step 3: Install Other Dependencies

```bash
# Install remaining packages
pip install numpy pandas scikit-learn matplotlib seaborn plotly jupyter ipykernel tqdm scipy networkx
```

## Alternative: Complete Installation Script

### For Windows (PowerShell)

```powershell
# 1. Install PyTorch
pip install torch torchvision torchaudio

# 2. Install PyTorch Geometric (handles dependencies automatically)
pip install torch-geometric

# 3. Install other dependencies
pip install numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0 matplotlib>=3.7.0 seaborn>=0.12.0 plotly>=5.14.0 jupyter ipykernel tqdm scipy networkx
```

### For Linux/Mac

```bash
# 1. Install PyTorch
pip install torch torchvision torchaudio

# 2. Install PyTorch Geometric
pip install torch-geometric

# 3. Install other dependencies
pip install numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0 matplotlib>=3.7.0 seaborn>=0.12.0 plotly>=5.14.0 jupyter ipykernel tqdm scipy networkx
```

## Troubleshooting

### Issue: torch-scatter/torch-sparse build errors

**Solution**: Install PyTorch first, then let torch-geometric handle these:

```bash
pip install torch
pip install torch-geometric
```

torch-geometric will automatically install compatible versions of torch-scatter and torch-sparse.

### Issue: CUDA version mismatch

**Solution**: Install CPU version first (works for learning):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
```

### Issue: Still having problems

**Solution**: Install packages one by one:

```bash
pip install torch
pip install torch-geometric
pip install numpy pandas scikit-learn
pip install matplotlib seaborn
pip install jupyter ipykernel
```

## Verify Installation

Run this to verify everything is installed:

```python
import torch
import torch_geometric
import numpy as np
import pandas as pd

print(f"PyTorch: {torch.__version__}")
print(f"PyTorch Geometric: {torch_geometric.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print("✓ All packages installed successfully!")
```

## Quick Install (All-in-One)

If you want to install everything at once after PyTorch:

```bash
pip install torch torch-geometric numpy pandas scikit-learn matplotlib seaborn plotly jupyter ipykernel tqdm scipy networkx
```

