# Testing script for morphology operations, to be incorporated in getdata.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mahotas as mh
import sys
from  scipy import ndimage
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, disk, dilation, erosion
from skimage.color import label2rgb

A = np.load('/u/data/alcer/DFXRM_rec/Rec_test/dataarray_clean.npy')

B = np.zeros([A.shape[0], A.shape[1], A.shape[2], A.shape[3], A.shape[4],])

# Size of the frame used to clean the images
sz_fr = 20

# Function to rebin the values in a matrix and assign to each bin the average
# intensity
def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

# Isolate the diffraction regions and then sum them
# Before that, subtract the image background, calculated usign a frame, where we
# expect no diffraction signal
for ii in range(0,A.shape[2],20):
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
            IM_clean_bin[IM_clean > 22] = 1
            Cleared = ndimage.binary_fill_holes(IM_clean_bin).astype(int)
            Dilated = erosion(dilation(Cleared, disk(1)), disk(1))
            Dilated_c = ndimage.binary_fill_holes(Dilated).astype(int)

            # Label image regions
            label_image = label(Dilated_c)

            Mask = np.zeros([IM_clean.shape[0], IM_clean.shape[1]])
            IM_clean_masked = np.zeros([IM_clean.shape[0], IM_clean.shape[1]])
            for region in regionprops(label_image):
                #Take regions with large enough areas
                if region.area >= 100:
                    id = region.label
                    Mask[label_image == id] = 1

            IM_clean_masked = IM_clean * Mask

            # Threshold, isolate and recongnize diffraction region
            fig = plt.figure()

            plt.subplot(2,3,1)
            plt.imshow(IM_clean)

            plt.subplot(2,3,2)
            plt.imshow(IM_clean_bin)

            plt.subplot(2,3,3)
            plt.imshow(Dilated_c)

            ax = plt.subplot(2,3,4)
            label_image = label(Dilated)
            plt.imshow(IM_clean)

            for region in regionprops(label_image):
                # take regions with large enough areas
                if region.area >= 100:
                    # draw rectangle around segmented coins
                    minr, minc, maxr, maxc = region.bbox
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)

            plt.subplot(2,3,5)
            plt.imshow(Mask)

            plt.subplot(2,3,6)
            plt.imshow(IM_clean_masked)

            plt.show()

            #fig = plt.figure()
            #plt.subplot(2,3,1)
            # Raw image
            #plt.imshow(IM)
            #plt.subplot(2,3,2)
            # Binarized image
            #plt.imshow(IM_reb)
            #plt.subplot(2,3,3)
            # Calculated background
            #plt.imshow(IM_reb_2)
            #plt.subplot(2,3,4)
            # Cleaned image
            #plt.imshow(IM_clean)
            #plt.subplot(2,3,5)
            # Cleaned image
            #plt.imshow(IM_clean_bin)
            #plt.show()
