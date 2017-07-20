# This script plots the distribution of the angular values for every omega

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Remember to check the path for the IO directory
io_dir = '/u/data/alcer/DFXRM_rec/Rec_test_2/'

A = np.load(os.path.join(io_dir + 'dataarray.npy'))
Theta = np.load(os.path.join(io_dir + 'theta.npy'))
print Theta.shape

# For each projection, mark the motor position in the pseudomotor, theta space
Counter = np.zeros([A.shape[2], A.shape[0], A.shape[1]])
num_pos = 0
# Loop over projections
for aa in range(A.shape[2]):
    # Loop over the theta values
    for bb in range(A.shape[0]):
        # Loop over the pseudomotor cvalues
        for cc in range(A.shape[1]):
            if sum(sum(A[bb,cc,aa,:,:])) > 0:
                Counter[aa, bb, cc] = 1
                num_pos = num_pos + 1

# We also want to keep track of the theta motor position in degrees
Counter_deg = np.zeros([num_pos, 4])
num_pos_1 = 0
for aa in range(A.shape[2]):
    # Loop over the theta values
    for bb in range(A.shape[0]):
        # Loop over the pseudomotor cvalues
        for cc in range(A.shape[1]):
            if Counter[aa, bb, cc] > 0:
                num_pos_1 = num_pos_1 + 1
                Counter_deg[num_pos_1 - 1, 0] = aa  # Projection number
                Counter_deg[num_pos_1 - 1, 1] = bb  # Pseudomotor index
                Counter_deg[num_pos_1 - 1, 2] = cc  # Theta index
                Counter_deg[num_pos_1 - 1, 3] = Theta[cc] # Theta in degrees

# Plot (pseudmotor, theta) distribution for a single projection
deg = 5

fig = plt.figure()
plt.imshow(Counter[deg, :, :])
plt.title('Motor positions in the (pseudomotor, theta index) space')
plt.xlabel('Theta (index)')
plt.ylabel('Pseudomotor (index)')
plt.show()

# Plot (pseudomotor, theta) distribution for a single projections
Counter_1 = np.zeros([11*11, 4])
line_num = 0
for i in range(Counter_deg.shape[0]):
    if Counter_deg[i, 0] == deg:
        line_num = line_num + 1
        Counter_1[line_num,:] = Counter_deg[i,:]

fig = plt.figure()
plt.scatter(Counter_1[:,3], Counter_1[:,1])
plt.show()

# Plot (pseudomotor, theta) distribution for all projections
fig = plt.figure()
plt.scatter(Counter_deg[:,3], Counter_deg[:,1])
plt.show()
