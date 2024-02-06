# Machine Learning for Oxygen (ML4O2)

## Setting up the python environment
  - Install miniconda3 on your cluster account
  - Initialize conda command
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate
```
  - Then create a new environment called ml4o2_v2.  
```
conda create --name ml4o2_v2
```
  - Then activate ml4o2_v2.
```
conda activate ml4o2_v2
```
  - Then install mamba in ml4o2_v2.
```
conda install -c conda-forge mamba
```
  - Install packages manually
```
mamba install -c conda-forge numpy matplotlib pandas netcdf4 dask nc-time-axis cartopy seaborn gsw xarray scikit-learn scipy joblib ipykernel
```
  - Once it is complete, make this environment available to the Jupyter Notebook:
```
python -m ipykernel install --user --name ml4o2_v2 --display-name ML4O2_v2
```
  - At this point the "ML4O2_v2" environment should be ready to use in Jupyterlab/Jupyter Notebook. 

## Xarray tutorial
- This video is a good introduction to X array [link](https://youtu.be/a339Q5F48UQ?si=mcCZE2vuptlOZPuE)

## Project scripts
  - Updated script for [training](https://github.com/takaito1/ML4O2/blob/main/o2_train_202402.ipynb)
  - Updated script for [projection](https://github.com/takaito1/ML4O2/blob/main/o2_project_202402.ipynb)
