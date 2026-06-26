# Weight Subspace Superposition (WSS)

## Environment Setup

This project uses conda for dependency management. Follow the steps below to set up your environment on local or HPC systems.

### Quick Start

```bash
# Create conda environment
conda create -n wss python=3.11 -y

# Activate environment
conda activate wss

# Install dependencies
pip install -r requirements.txt
```

### HPC System Setup

For HPC clusters (e.g., SLURM), follow this workflow:

#### 1. Load modules (if available on your HPC system)
```bash
# Common HPC module patterns - check with `module avail`
module load python
module load cuda/12.1  # If CUDA is available and you want GPU support
```

#### 2. Create isolated conda environment
```bash
# Create environment in a location with adequate disk space
# (avoid home directory quotas if your HPC has them)
conda create -n wss python=3.11 -y

# Activate the environment
conda activate wss
```

#### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

#### 4. Verify installation
```bash
# Test core imports
python -c "import torch; import jax; import geoopt; print('✓ All packages imported successfully')"

# Check PyTorch GPU availability (if CUDA environment)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Dependency Notes

- **torch 2.3.1**: PyTorch for neural networks
- **jax[cpu]**: JAX with CPU backend (modify to `jax[cuda]` if GPU CUDA support available)
- **geoopt 0.5.1**: Riemannian optimization (note: known M1/MPS compatibility issues; use CPU if you encounter problems)
- **Python 3.11**: Tested with Python 3.11

### CUDA/GPU Configuration

If your HPC system has CUDA-capable GPUs:

```bash
# Install CUDA-enabled PyTorch (example for CUDA 12.1)
conda activate wss
pip uninstall torch torchvision -y
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

For JAX GPU support, replace `jax[cpu]` with `jax[cuda12]` (adjust version as needed).

### Troubleshooting

**Geoopt import errors**: This can occur on Apple Silicon (M1/M2) systems. Use CPU-only mode:
```python
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
```

**Module not found errors on HPC**: Ensure you've activated the environment:
```bash
conda activate wss
```

**Disk space issues**: If conda environments exceed quotas, configure conda to use a different directory:
```bash
mkdir -p /path/to/hpc/scratch/conda
conda config --append envs_dirs /path/to/hpc/scratch/conda
```
