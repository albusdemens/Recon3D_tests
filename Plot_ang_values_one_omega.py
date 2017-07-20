# This script plots the distribution of the angular values for every omega

import numpy as np
import matplotlib.pyplot as plt
import os

# Remember to check the path for the IO directory
io_dir = '/u/data/alcer/DFXRM_rec/Rec_test/'

A = np.load(os.path.join(io_dir + 'dataarray.npy')
Theta = np.load(os.path.join(io_dir + 'theta.npy')

# For each projection, mark the motor position in the pseudomotor, theta space
Counter = np.zeros([A.shape[2], A.shape[0], A.shape[1]])
# Loop over projections
for aa in range(A.shape[2]):
    # Loop over the theta values
    for bb in range(A.shape[0]):
        # Loop over the pseudomotor cvalues
        for cc in range(A.shape[1]):
            if sum(sum(A[bb,cc,aa,:,:])) > 0:
                Counter[aa, bb, cc] = 1

# Plot (pseudmotor, theta) distribution for a single projection
fig = plt.figure()
plt.imshow(Counter[5, :, :])
plt.show()

# Plot (pseudomotor, theta) distribution for all projections
