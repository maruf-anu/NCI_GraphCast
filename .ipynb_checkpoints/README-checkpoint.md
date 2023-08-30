# NCI_GraphCast
NCI deployment of DeepMind GraphCast

Following are the steps to install and run the DeepMind GraphCast on Gadi.

## Install

Step 0 <br>
Log into Gadi and choose a suitable conda base. <br>
(Alternately, one can also install a coda base.)
```
conda -V
conda 23.1.0
```


Step 1
```bash
conda create -p /path/to/dir/nci_graphcast  \
	-c "nvidia/label/cuda-11.8.0" -c conda-forge \
	python=3.10 cudatoolkit=11.8.0 cuda-nvcc \
	xarray dask netCDF4 bottleneck jupyterlab ipywidgets 
```

Step 2
```bash
conda activate /path/to/dir/nci_graphcast 
pip install --upgrade pip
```

Step 3
```bash
python -m pip install --upgrade \
	nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.13.* tensorrt \
	"jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
	chex -U dm-haiku jraph scipy dm-tree trimesh[all] \
	git+https://github.com/deepmind/jaxline  \
	https://github.com/deepmind/graphcast/archive/master.zip \
	google-cloud-speech google-cloud-storage google-cloud-bigquery \
	google-cloud-texttospeech google-cloud-aiplatform google-cloud-domains \
	--no-cache-dir
```

Step 4
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

```


## Launch on NCI ARE

Step 1 <br>
go to https://are.nci.org.au/

Step 2 <br>
Fill up the Jupyter form
```
Walltime (hours): <As required>
Queue: gpuvolta
Compute Size: 1gpu
Project: <Your Project Code>
Storage: gdata/wb00 + <All other file systems you need to access>

Python or Conda virtual environment base: <base conda installation path>
Conda environment: </path/to/dir/nci_graphcast>
```

Step 3 <br>
Go to your desired directory on Gadi and clone the repo
```bash
git clone https://github.com/maruf-anu/NCI_GraphCast.git
```

Step 4 <br>
Open the ARE JupyterLab. <br>
Load and run the NCI graphcast notebook from download location: <br> 
`/path/to/dir/NCI_GraphCast/graphcast-03.ipynb`

## Run on CMD

One can also develop or run your own code for training and inference using this environment.

Step 0 <br>
Log into Gadi and choose a suitable conda base. <br>

Step 1 <br>
Activate the nci_graphcast environment (path from the above installation).
```bash
conda activate /path/to/dir/nci_graphcast 
```
Now, one can run the graphcast commands/script inside the environment. 

