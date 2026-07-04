# ML4O2 implementation on PACE at Georgia Tech
- prerequisite: a PACE account
- use PACE ondemand with GPU support (A100)

## Setting up the python environment
  - Install miniconda3 on your cluster account
  - Initialize conda command
  - Start the session with GPU support
```
export base='/storage/home/hcoda1/8/takamitsu3/r-takamitsu3-0/'
mkdir -p $base/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $base/miniconda3/miniconda.sh
bash $base/miniconda3/miniconda.sh -b -u -p $base/miniconda3
rm -rf $base/miniconda3/miniconda.sh
source $base/miniconda3/etc/profile.d/conda.sh
conda activate
