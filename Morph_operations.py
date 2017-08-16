# Testing script for morphology operations, to be incorporated in getdata.py

import numpy as np
import matplotlib.pyplot as plt
import mahotas as mh
import sys

A = np.load('/u/data/alcer/DFXRM_rec/Rec_test/dataarray_clean.npy')

B = np.zeros([A.shape[0], A.shape[1], A.shape[2], A.shape[3], A.shape[4],])

# Size of the frame used to clean the images
sz_fr = 20

# Function to rebin the values in a matrix and assign to each bin the average
# intensity
def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

# Test 1: isolate the diffraction regions and then sum them
# Before that, subtract the image background, calculated usign a frame, where we
# expect no diffraction signal
for ii in range(A.shape[2]):
    for aa in range(A.shape[0]):
        for bb in range(A.shape[1]):
            IM = np.zeros([A.shape[3], A.shape[4]])
            IM_raw = np.zeros([A.shape[3], A.shape[4]])
            IM[:,:] = A[aa,bb,ii,:,:]
            #IM = IM_raw
            # Get rid og hot pixels
            #IM[IM>100] = 0
            # Rebin the considered plot
            IM_reb = np.zeros([A.shape[3]/sz_fr, A.shape[4]/sz_fr])
            IM_reb = rebin(IM, [IM_reb.shape[0],IM_reb.shape[1]])
            # Calculate the expected background distribution, assuming it to
            # be linear
            IM_reb_2 = np.zeros([A.shape[3]/sz_fr, A.shape[4]/sz_fr])
            IM_reb_3 = np.zeros([A.shape[3], A.shape[4]])
            IM_reb_2[0,:] = IM_reb[0,:]
            IM_reb_2[IM_reb.shape[0]-1,:] = IM_reb[IM_reb.shape[0]-1,:]
            IM_reb_2[:,0] = IM_reb[:,0]
            IM_reb_2[:,IM_reb.shape[0]-1] = IM_reb[:,IM_reb.shape[0]-1]
            for jj in range(1,IM_reb.shape[0]-1):
                for kk in range(1,IM_reb.shape[1]-1):
                    I_min_x = min(IM_reb[jj,0], IM_reb[jj,IM_reb.shape[1]-1])
                    I_max_x = max(IM_reb[jj,0], IM_reb[jj,IM_reb.shape[1]-1])
                    I_min_y = min(IM_reb[0,kk], IM_reb[IM_reb.shape[0]-1, kk])
                    I_max_y = max(IM_reb[0,kk], IM_reb[IM_reb.shape[0]-1, kk])
                    I_eval_x = I_min_x + ((I_max_x - I_min_x) / (IM.shape[0] - 2*sz_fr)) * (jj - sz_fr)
                    I_eval_y = I_min_y + ((I_max_y - I_min_y) / (IM.shape[1] - 2*sz_fr)) * (kk - sz_fr)

                    # For the dataset 1, we notice that the crucial component to
                    # take into account is how the background varies along Y
                    IM_reb_2[jj,kk] = np.mean([I_min_x, I_max_x])

            for jj in range(IM_reb.shape[0]):
                for kk in range(IM_reb.shape[1]):
                    IM_reb_3[jj*sz_fr:(jj+1)*sz_fr, kk*sz_fr:(kk+1)*sz_fr] = IM_reb_2[jj,kk]

            # Subtract the calculated background from the initial image
            IM_clean = IM - IM_reb_3
            IM_clean[IM_clean < 0] = 0
            IM_clean_bin = np.zeros([IM_clean.shape[0], IM_clean.shape[1]])
            IM_clean_bin[IM_clean > 20] = 1

            fig = plt.figure()
            plt.subplot(2,3,1)
            # Raw image
            plt.imshow(IM)
            plt.subplot(2,3,2)
            # Binarized image
            plt.imshow(IM_reb)
            plt.subplot(2,3,3)
            # Calculated background
            plt.imshow(IM_reb_2)
            plt.subplot(2,3,4)
            # Cleaned image
            plt.imshow(IM_clean)
            plt.subplot(2,3,5)
            # Cleaned image
            plt.imshow(IM_clean_bin)
            plt.show()

sys.exit()

# Test 2: sum and isolate the diffraction region
for ii in range(A.shape[2]):
    Sum_oo = np.zeros([A.shape[3], A.shape[4]])
    Sum_oo_th = np.zeros([A.shape[3], A.shape[4]])
    for aa in range(A.shape[3]):
        for bb in range(A.shape[4]):
            Sum_oo[aa,bb] += np.sum(A[:,:,ii,aa,bb])
    # Threshold the Image
    Sum_oo_th = Sum_oo
    Sum_oo_th[Sum_oo_th < 1000] = 0

    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(Sum_oo)
    plt.subplot(1,2,2)
    plt.imshow(Sum_oo_th)
    plt.show()

sys.exit()



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
