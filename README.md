# Recon3D

## Reading data from .edf files to a NumPy array.

The script getdata.py reads data from .edf files and outputs a NumPy array with the dimensions tilt1-steps x tilt2-steps x omega-steps x img-xlen x img-ylen, where tilt1 and tilt2 are the two top tilts in the LAB goniometer at ID06. Omega is the topo-tomo rotation stage. The command to create the array is the following:

'''
$ python [datadir] [dataname] [bgdir] [bgname] [poi] [imgsize] [outputpath] [outputdirname]
'''


