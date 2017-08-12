# Testing script for morphology operations, to be incorporated in getdata.py

import numpy as np
import matplotlib.pyplot as plt
import mahotas as mh
import sys

A = np.load('/u/data/alcer/DFXRM_rec/Rec_test/dataarray_clean.npy')

B = np.zeros([A.shape[0], A.shape[1], A.shape[2], A.shape[3], A.shape[4],])

mean_proj = np.zeros([A.shape[2], 3])

for ii in range(A.shape[2]):
    mean_proj[ii,0] = ii
    sum_img = np.zeros([A.shape[3], A.shape[4]])
    for jj in range(A.shape[3]):
        for kk in range(A.shape[4]):
            sum_img[jj,kk] = np.sum(A[:,:,ii,jj,kk])
    mean_proj[ii,1] = np.mean(sum_img) / (A.shape[0]*A.shape[1])
mean_max = max(mean_proj[:,1])

thr = np.zeros([A.shape[2], 2])
for k in range(0,A.shape[2],10):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            IM = np.zeros([A.shape[3], A.shape[4]])
            IM[:,:] = A[i,j,k,:,:]
            thr[k,0] = k
            # The threshold value adapts to the intensity
            # Recorded for a given projection
	    thr[k,1] = 40 * mean_proj[k,1] / mean_max

	    IM[IM < thr[k,1]] = 0
        IM[IM >= thr[k,1]] = 1
        B[i,j,k,:,:] = IM[:,:]

        print int(thr[k,1]), i, j, k
        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(A[i,j,k,:,:])
        plt.subplot(1,2,2)
        plt.imshow(B[i,j,k,:,:])
        plt.show()

I_int_1 = np.zeros([A.shape[2], 3])
I_int_2 = np.zeros([A.shape[2], 3])
for ii in range(A.shape[2]):
    I_int_1[ii,0] = ii
    I_int_2[ii,0] = ii
    I_int_1[ii,1] = np.sum(A[:,:,ii,:,:])
    I_int_2[ii,1] = np.sum(B[:,:,ii,:,:])
    I_int_1[ii,2] = np.mean(A[:,:,ii,:,:])
    I_int_2[ii,2] = np.mean(B[:,:,ii,:,:])

fig = plt.figure()
plt.subplot(1,2,1)
plt.imshow(A[4,6,138,:,:])
plt.subplot(1,2,2)
plt.imshow(B[4,6,138,:,:])
plt.show()

sys.exit()

fig = plt.figure()
plt.subplot(2,2,1)
plt.scatter(I_int_1[:,0], I_int_1[:,1])
plt.title('Sum before binarization')
plt.subplot(2,2,2)
plt.scatter(I_int_2[:,0], I_int_2[:,1])
plt.title('Sum after binarization')
plt.subplot(2,2,3)
plt.scatter(I_int_1[:,0], I_int_1[:,2])
plt.title('Mean before binarization')
plt.subplot(2,2,4)
plt.scatter(I_int_2[:,0], I_int_2[:,2])
plt.title('Mean after binarization')
plt.show()
