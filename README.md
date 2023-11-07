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
  - Then create a new environment called ml4o2_v2. After this, do one of the following option (1) or (2). 
```
conda create --name ml4o2_v2
conda activate ml4o2_v2
conda install -c conda-forge mamba
```
  - Option (1) : Install packages manually
```
mamba install -c conda-forge numpy matplotlib pandas netcdf4 dask nc-time-axis cartopy seaborn gsw xarray scikit-learn scipy loblib ipykernel
```
  - Option (2) : Clone from the package list file (ml4o2-packages.txt)
```
mamba create --name ml4o2_v2 --file ml4o2-packages.txt
```
  - Once it is complete, make this environment available to the Jupyter Notebook:
```
conda activate ml4o2_v2
python -m ipykernel install --user --name ml4o2_v2 --display-name ML4O2_v2
```
  - At this point the "ML4O2_v2" environment should be ready to use in Jupyterlab/Jupyter Notebook. 

## Project scripts
  - develop/ stores scripts for development
  - Example for a data-driven modeling of O2 in the North Atlantic [python](https://github.com/takaito1/ML4O2/blob/main/o2mod_example_CV_v2.ipynb) This is a 3 dimensional version (v2) machine learning model, O2 = O2(T,S,lon,lat,year,month), 80-20 train-test split with K-fold cross validation. The algorithms include Random Forest Regression, shallow and deep Neural Network. 
