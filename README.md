# Recon3D

*** Experimental - still a work in progress ***

White paper that explains the principle can be found [here](https://github.com/acjak/Recon3D/raw/master/dfxm.pdf).

Recon3D is a program to get a 3D reconstruction from topo-tomo data sets from ID06 at ESRF. The program takes data where a sample is rotated around the *diffrx* and at every point, a mosaicity scan is done. This is done through a forward projection algorithm to calculate a given voxel's reflected spot position on the detector.

Future versions will be able to do reconstruction based on strain datasets.

## Reading data from .edf files to a NumPy array.

The script getdata.py reads data from .edf files and outputs a NumPy array with the dimensions *tilt1*-steps x *tilt2*-steps x *omega*-steps x *img_xlen* x *img_ylen*, where *tilt1* and *tilt2* are the two top tilts in the LAB goniometer at ID06. *Omega* is the topo-tomo rotation stage. The command to create the array is the following:

```
$ python getdata.py [datadir] [dataname] [bgdir] [bgname] [poi] [imgsize] [outputpath] [outputdirname]
```

Arguments are the following:

| Argument | Description | Example |
| ------------- | ----------- | ----------- |
| datadir      | Directory of .edf files. | /data/experiment1 |
| dataname     | Name of data files. | run1_ |
| bgdir     | Directory of background .edf files. | /data/background1 |
| bgname     | Name of background files. | bg1_ |
| poi     | Center point for region of interest. | 512,512 |
| imgsize     | Size of region of interest. | 200,200 |
| outputpath     | Path to put the output directory. | /analysis/output |
| outputdirname     | Name of output dir. | exp1_array |

If MPI is available, the following command will run the script in 10 processes at the same time. This will vastly increase the speed.

```
$ mpirun -n 10 python getdata.py [datadir] [dataname] [bgdir] [bgname] [poi] [imgsize] [outputpath] [outputdirname]
```

## Running the reconstruction algorithm on a data set.

```
$ mpirun -n 10 python recon3d.py [initfile]
```
