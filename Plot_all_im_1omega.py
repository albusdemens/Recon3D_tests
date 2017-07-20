# Derived from Plot_data_1omega.py (code split after memory leak)
# Aim: plot the images collected at a certain projection

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

io_dir = '/u/data/alcer/DFXRM_rec/Rec_test_2/'

A = np.load(os.path.join(io_dir + 'dataarray.npy'))

# Select projection number
omega = 2

# We want to plot, in a grid, all images collected at a certain projection
# Load the datafile with all information, and store to an array the data
# relative to the images in a certain projection
Data = np.loadtxt(os.path.join(io_dir + 'Image_properties.txt'))
Data_angle = np.zeros([A.shape[0]*A.shape[0],2])
idx = 0
for i in range(Data.shape[0]):
    if Data[i,3] == omega:
        idx = idx +1
        Data_angle[idx-1, 0] = Data[i,1]
        Data_angle[idx-1, 1] = Data[i,2]

# Plot the various recorded images

aa = int(Data_angle[0,0])
bb = int(Data_angle[0,1])
AA = np.zeros([A.shape[3], A.shape[4]])
AA[:,:] = A[aa,bb,omega,:,:]

fig, axes = plt.subplots(A.shape[0], A.shape[0])
for i in range(A.shape[0]*A.shape[0]):
    a1 = fig.add_subplot(A.shape[0],A.shape[0],i+1)
    plt.setp(a1.get_xticklabels(), visible=False)
    plt.setp(a1.get_yticklabels(), visible=False)
    aa = Data_angle[i,0]
    bb = Data_angle[i,1]
    plt.imshow(A[int(aa),int(bb),omega,:,:])

plt.show()
