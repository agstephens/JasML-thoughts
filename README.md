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
python test-packages.py
```




