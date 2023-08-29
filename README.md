# NCI_GraphCast
NCI deployment of DeepMind GraphCast


## Install
Step 1
```bash
conda create -p /scratch/fp0/mah900/env/nci_graphcast \
	-c conda-forge python=3.10 cudatoolkit=11.8.0 \
	xarray dask netCDF4 bottleneck jupyterlab ipywidgets 
``` 
Step 2
```bash
conda activate /scratch/fp0/mah900/env/nci_graphcast
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


