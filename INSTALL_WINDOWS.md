# Windows Installation Guide (Long Path Issue Fix)

## Problem
Windows has a 260-character path limit that causes installation errors with some packages (especially Jupyter widgets).

## Solutions (Try in Order)

### Solution 1: Enable Long Path Support (Recommended)

**Run PowerShell as Administrator:**

```powershell
# Check current setting
Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled"

# Enable long paths (requires admin)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Restart your computer for changes to take effect
```

**Or use Registry Editor:**
1. Press `Win + R`, type `regedit`
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Set `LongPathsEnabled` to `1`
4. Restart computer

### Solution 2: Install Without Jupyter Lab (Simpler)

Skip Jupyter Lab extensions that cause the issue:

```powershell
# Install core packages without Jupyter Lab
pip install torch torch-geometric
pip install numpy pandas scikit-learn matplotlib seaborn
pip install jupyter notebook  # Use notebook instead of lab
pip install ipykernel tqdm scipy networkx
```

### Solution 3: Use Shorter Path

Move project to a shorter path:

```powershell
# Current path is very long:
# C:\PERSONAL\BMS\SEM 5\MLG\Project\wildfire-spread-prediction

# Move to shorter path like:
# C:\Projects\wildfire-prediction
```

### Solution 4: Install Packages Individually

Skip problematic packages:

```powershell
# Core packages
pip install torch
pip install torch-geometric
pip install numpy pandas scikit-learn
pip install matplotlib seaborn
pip install jupyter notebook  # Skip jupyterlab
pip install ipykernel tqdm scipy networkx

# Skip plotly if it causes issues (optional for visualization)
# pip install plotly
```

### Solution 5: Use Conda (Alternative)

If pip continues to have issues:

```powershell
# Install Miniconda/Anaconda, then:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
conda install numpy pandas scikit-learn matplotlib seaborn jupyter scipy networkx
```

## Quick Install (Minimal - No Jupyter Lab)

```powershell
# This should work without long path issues
pip install torch torch-geometric numpy pandas scikit-learn matplotlib seaborn jupyter notebook ipykernel tqdm scipy networkx
```

## Verify Installation

```python
import torch
import torch_geometric
print("✓ Installation successful!")
```

## Note

You can use **Jupyter Notebook** instead of Jupyter Lab - it works the same for this project and avoids the long path issue.

