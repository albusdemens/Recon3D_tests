import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from itertools import count
import sys

A = np.load('/Users/Alberto/Documents/Data_analysis/DFXRM/Results_sunday/grain_ang.npy')

# Find the max of the mean for the weight
M = np.zeros(A.shape[2])
for i in range(A.shape[2]):
    M[i] = np.mean(A[:,:,i,2])
max_mean = np.max(M)

# Select a z value and plot the respective slice (label by label and combined labels)

z_val = 50
A_z_val = np.empty(shape=[A.shape[0], A.shape[1]])
A_z_val_0 = np.empty(shape=[A.shape[0], A.shape[1]])
A_z_val_1 = np.empty(shape=[A.shape[0], A.shape[1]])
A_z_val_2 = np.empty(shape=[A.shape[0], A.shape[1]])

A_z_val_0[:,:] = A[:,:,z_val,0]#/np.mean(A[:,:,z_val,0])
A_z_val_1[:,:] = A[:,:,z_val,1]#/np.mean(A[:,:,z_val,1])
A_z_val_2[:,:] = A[:,:,z_val,2]/max_mean
A_z_val[:,:] = ( 0.5 * ( A_z_val_0[:,:] + A_z_val_1[:,:]))

plt.subplot(221)
plt.imshow(A_z_val_0)
plt.title('Theta')

plt.subplot(222)
plt.imshow(A_z_val_1)
plt.title('Gamma')

plt.subplot(223)
plt.imshow(A_z_val_2)
plt.title('Weight')

plt.subplot(224)
plt.imshow(A_z_val)
plt.title('Combined value')

plt.show()

plt.imshow(A_z_val_2)
plt.title('Weight')
plt.show()

sys.exit

# Save, layer by layer, the slices of the volume

def gen_filenames(prefix, suffix, places=3):
    """Generate sequential filenames with the format <prefix><index><suffix>

       The index field is padded with leading zeroes to the specified number of places
    """
    pattern = "{}{{:0{}d}}{}".format(prefix, places, suffix)
    for i in count(1):
        yield pattern.format(i)

g = gen_filenames("/Users/Alberto/Documents/Data_analysis/DFXRM/Results_sunday/3D_reconstruction/img_", ".png")
for i in range(A.shape[2]):
    A_2D = np.empty(shape=[A.shape[0], A.shape[1]])
    for j in range(A.shape[0]):
        for k in range(A.shape[1]):
            if ((A[j,k,i,2]/max_mean)<0.03) or (0.4<(A[j,k,i,2]/max_mean)<1):
                A_2D[j,k,i] = ( 0.5 * ( A[j,k,i,0] + A[j,k,i,1]))
            else:
                A_2D[j,k,i] = 0

    plt.imshow(A_2D) #Needs to be in row,col order
    plt.savefig(next(g))
