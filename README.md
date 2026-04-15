# JasML environment - initial thoughts

## First test - clone Jaspy and then install extra packages

### Login to the interactive GPU node 

We login to the interactive GPU node so that GPU-aware packages (like `torch`) will pick up the 
existence of `CUDA` and enable GPU mode (which will run a lot faster than CPU mode).

```bash
ssh gpuhost001.jc.rl.ac.uk
```

### Clone a Jaspy env

```bash
/apps/jasmin/jaspy/miniforge_envs/jaspy3.12/mf3-25.3.0-3/bin/conda create --clone jaspy3.12-mf3-25.3.0-3-v20250704 --prefix /gws/smf/j04/cmip6_prep/users/astephen/jaspy-ml-v1
```

### Activate that conda environment

```bash
/apps/jasmin/jaspy/miniforge_envs/jaspy3.12/mf3-25.3.0-3/bin/conda activate /gws/smf/j04/cmip6_prep/users/astephen/jaspy-ml-v1
```

### Pip install extra packages

On a GPU host, start with the `requirements.yml` file below, and convert it to a `requirements.txt` file using this script:

```bash
python convert-reqs.py
```

Pip install, specifiying the PyTorch CUDA index specifically:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple -r requirements.txt
```

### Test the main packages are there and CUDA is available

```bash
python test-environment.py
```

### Output from test script

The output should look like:

```bash
python test-jasml.py 

Python 3.12.11 | packaged by conda-forge | (main, Jun  4 2025, 14:45:31) [GCC 13.3.0]

==================================================
  PyTorch
==================================================
  ✓ Installed       : 2.8.0+cu128
  ✓ CUDA available   : True
  ✓ GPU count       : 4
    [0] NVIDIA A100-SXM4-40GB  (39.5 GB)
    [1] NVIDIA A100-SXM4-40GB  (39.5 GB)
    [2] NVIDIA A100-SXM4-40GB  (39.5 GB)
    [3] NVIDIA A100-SXM4-40GB  (39.5 GB)
  ✓ Tensor on GPU   : cuda:0

==================================================
  TensorFlow
==================================================
  ✓ Installed       : 2.20.0
  ✓ GPU count       : 4
    GPU: /physical_device:GPU:0
    GPU: /physical_device:GPU:1
    GPU: /physical_device:GPU:2
    GPU: /physical_device:GPU:3
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1776248549.933879   24239 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 36997 MB memory:  -> device: 0, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:01:00.0, compute capability: 8.0
I0000 00:00:1776248549.935311   24239 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 38479 MB memory:  -> device: 1, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:41:00.0, compute capability: 8.0
I0000 00:00:1776248549.936566   24239 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:2 with 38479 MB memory:  -> device: 2, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:81:00.0, compute capability: 8.0
I0000 00:00:1776248549.937741   24239 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:3 with 38479 MB memory:  -> device: 3, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:c1:00.0, compute capability: 8.0
  ✓ Tensor on GPU   : /job:localhost/replica:0/task:0/device:GPU:0

==================================================
  JAX
==================================================
  ✓ Installed       : 0.7.2
  ✓ GPU/TPU count   : 4
    cuda:0
    cuda:1
    cuda:2
    cuda:3
  ✓ Default backend : gpu

==================================================
  Summary
==================================================
  Framework      Installed    GPU Visible
  -------------- ----------- -----------
  PyTorch        ✓            ✓
  TensorFlow     ✓            ✓
  JAX            ✓            ✓
```




