import numpy as np
import matplotlib.pyplot as plt
from lib.miniged import GetEdfData	# Read EDF files and plot background
import sys
import time
import os
import fabio
import warnings

im_paths = np.genfromtxt('/home/nexmap/alcer/DFXRM/phiscan/list_phiscan.txt', dtype = str)
im_dir = '/u/data/andcj/hxrm/Al_april_2017/topotomo/monday/Al3/phi_resolution/'

I_stack = np.zeros([im_paths.shape[0], 512, 512])
I_sum = np.zeros([512, 512])
for i in range(im_paths.shape[0]):
    im_path = os.path.join(im_dir + im_paths[i])
    I = fabio.open(im_path).data
    for j in range(I_sum.shape[0]):
        for k in range(I_sum.shape[1]):
            if I[j,k] < 5E+3:
                I_stack[i,j,k] = I[j,k] - 300
                I_sum[j,k] += I[j,k]

fig = plt.figure()
plt.imshow(I_sum, cmap='gray_r')
plt.show()

#for i in range(im_paths.shape[0]):
#   fig = plt.figure()
#   plt.imshow(I_stack[i,:,:])
#   plt.show()

# Plot the various recorded images (one projection)
fig, axes = plt.subplots(8, 8)
for i in range(60):
    a1 = fig.add_subplot(8,8,i+1)
    plt.setp(a1.get_xticklabels(), visible=False)
    plt.setp(a1.get_yticklabels(), visible=False)
    plt.imshow(I_stack[i,:,:], cmap='gray_r')

plt.show()
