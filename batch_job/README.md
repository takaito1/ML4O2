# ML4O2 batch mode
- This set of code is required to perform ML training, evaluation, and project in the batch mode
- Make following changes:
- In train.py, make adjustments to data1, data2, dirout (/glade/derecho/scratch/ito/ML4O2_temp), diro (/glade/derecho/scratch/ito/WOD18_OSDCTD)
- In eval.py, make adjustments to date2, dirout (/glade/derecho/scratch/ito/ML4O2_temp), dirfin (/glade/derecho/scratch/ito/ML4O2_results)
- In project.py, make adjustments to dirout (/glade/derecho/scratch/ito/ML4O2_results), diro (/glade/derecho/scratch/ito/WOD18_OSDCTD/)
- Create disk space to hold results in your scratch
- mkdir -p /glade/derecho/scratch/userid/ML4O2_temp
- mkdir -p /glade/derecho/scratch/userid/WOD18_OSDCTD
- mkdir -p /glade/derecho/scratch/userid/ML4O2_results
- In run.pbs, change email address and source directory (/glade/work/ito/ML4O2/batch_job, /glade/u/home/ito/miniconda3/etc/profile.d/conda.sh)

