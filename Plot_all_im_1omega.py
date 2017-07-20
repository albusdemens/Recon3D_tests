# Derived from Plot_data_1omega.py (code split after memory leak)
# Aim: plot the images collected at a certain projection

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

io_dir = '/u/data/alcer/DFXRM_rec/Rec_test_2/'

A = np.load(os.path.join(io_dir + 'dataarray.npy'))

# Select projection number
omega = 100

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

# Plot the various recorded images (one projection)
fig, axes = plt.subplots(A.shape[0], A.shape[0])
for i in range(A.shape[0]*A.shape[0]):
    a1 = fig.add_subplot(A.shape[0],A.shape[0],i+1)
    plt.setp(a1.get_xticklabels(), visible=False)
    plt.setp(a1.get_yticklabels(), visible=False)
    aa = Data_angle[i,0]
    bb = Data_angle[i,1]
    plt.imshow(A[int(aa),int(bb),omega,:,:])

plt.show()

# Plot the various recorded images (all projections)
fig, axes = plt.subplots(A.shape[0], A.shape[0])
#We also keep track of the integrated intensity
II_matrix = np.zeros([A.shape[0], A.shape[0]])
for i in range(A.shape[0]*A.shape[0]):
    a1 = fig.add_subplot(A.shape[0],A.shape[0],i+1)
    plt.setp(a1.get_xticklabels(), visible=False)
    plt.setp(a1.get_yticklabels(), visible=False)
    aa = int(Data_angle[i,0])
    bb = int(Data_angle[i,1])

    AA = np.zeros([A.shape[3], A.shape[4]])
    for oo in range(A.shape[2]):
        AA[:,:] += A[aa,bb,oo,:,:]
    II_matrix[ int(aa), int(i - (int(aa) * A.shape[0])) ] = sum(sum(AA))
    plt.imshow(AA[:,:])

plt.show()

# Plot how the integrated intensity changes during the rocking scan
fig = plt.figure()
plt.imshow(II_matrix)
plt.title('Distribution of the integrated intensities (all proj)')
plt.xlabel('Theta (index)')
plt.ylabel('Pseudomotor (index)')
plt.show()

# Return the coordinates where we have the lower integrated intensity
II_value = np.zeros([A.shape[0]*A.shape[0], 3])
lin_n = 0
for aa in range(II_matrix.shape[0]):
    for bb in range(II_matrix.shape[1]):
        lin_n = lin_n + 1
        II_value[lin_n - 1, 0] = int(aa)
        II_value[lin_n - 1, 1] = int(bb)
        II_value[lin_n - 1, 2] = II_matrix[aa, bb]

# Sort integrated intensities; for the rolling median consider the images giving
# the two lower values
i = II_value[:,2].argsort()
II_value_sorted = II_value[i,:]

print II_value_sorted

print 'To clean the dataset, we suggest to use the images number %i [%i, %i] and %i [%i, %i]' % ( II_value_sorted[0,1] + (II_value_sorted[0,0] * A.shape[0]), II_value_sorted[0,1], II_value_sorted[0,0], II_value_sorted[1,1] + (II_value_sorted[1,0] * A.shape[0]), II_value_sorted[1,1], II_value_sorted[1,0] )
