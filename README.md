# Machine Learning for Oxygen (ML4O2)

## Python environment
  - Initialize conda command:
    '''
    $ source ~/miniconda3/etc/profile.d/conda.sh
    '''

## Project scripts
  - develop/ stores scripts for development
  - Example for a data-driven modeling of O2 in the North Atlantic [python](https://github.com/takaito1/ML4O2/blob/main/o2mod_example_CV_v2.ipynb) This is a 3 dimensional version (v2) machine learning model, O2 = O2(T,S,lon,lat,year,month), 80-20 train-test split with K-fold cross validation. The algorithms include Random Forest Regression, shallow and deep Neural Network. 
