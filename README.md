# Machine Learning for Oxygen (ML4O2)

## Python environment
  - Copy the env-ml4o2.yml file to your home directory
  - Initialize conda command
```
source ~/miniconda3/etc/profile.d/conda.sh
conda activate
```
  - Then create a new environment using env-ml4o2.yml file. This may take a few minutes
```
conda env create --file export.yml
```
  - Once it is complete, make this environment available to the Jupyter Notebook:
```
conda activate ml4o2
python -m ipykernel install --user --name ml4o2 --display-name ML4O2
```
  - At this point the "ML4O2" environment should be ready to use in Jupyterlab/Jupyter Notebook. 

## Project scripts
  - develop/ stores scripts for development
  - Example for a data-driven modeling of O2 in the North Atlantic [python](https://github.com/takaito1/ML4O2/blob/main/o2mod_example_CV_v2.ipynb) This is a 3 dimensional version (v2) machine learning model, O2 = O2(T,S,lon,lat,year,month), 80-20 train-test split with K-fold cross validation. The algorithms include Random Forest Regression, shallow and deep Neural Network. 
