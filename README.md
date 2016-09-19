# Recon3D

## Reading data from .edf files to a NumPy array.

The script getdata.py reads data from .edf files and outputs a NumPy array with the dimensions tilt1-steps x tilt2-steps x omega-steps x img-xlen x img-ylen, where tilt1 and tilt2 are the two top tilts in the LAB goniometer at ID06. Omega is the topo-tomo rotation stage. The command to create the array is the following:

```
$ python [datadir] [dataname] [bgdir] [bgname] [poi] [imgsize] [outputpath] [outputdirname]
```

Arguments are the following:

| Argument | Description | Example |
| ------------- | ----------- |
| datadir      | Directory of .edf files. | /data/experiment1 |
| dataname     | Name of data files. | run1_ |
| bgdir     | Directory of background .edf files. | /data/background1 |
| bgname     | Name of background files. | bg1_ |
| poi     | Center point for region of interest. | 512,512 |
| imgsize     | Size of region of interest. | 200,200 |
| outputpath     | Path to put the output directory. | /analysis/output |
| outputdirname     | Name of output dir. | exp1_array |
