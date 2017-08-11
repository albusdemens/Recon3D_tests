import numpy as np
import matplotlib.pyplot as plt
import mahotas as mh

A = np.load('/u/data/alcer/DFXRM_rec/Rec_test/dataarray_clean_norm.npy')

B = np.zeros([A.shape[0], A.shape[1], A.shape[2], A.shape[3], A.shape[4],])

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        for k in range(A.shape[2]):
            IM = np.zeros([A.shape[3], A.shape[4]])
            IM[:,:] = A[i,j,k,:,:]
            IM[IM>80] = 1
            B[i,j,k,:,:] = IM[:,:]

fig = plt.figure()
plt.subplot(1,2,1)
plt.imshow(A[3,3,1,:,:])
plt.subplot(1,2,2)
plt.imshow(B[3,3,1,:,:])
plt.show()
