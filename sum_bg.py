# This script analyses the background images used in the DFXRM reconstruction

import numpy as np
import matplotlib.pyplot as plt
import lib.EdfFile as EF
import sys

A = [line.rstrip() for line in open('list_bg.txt')]

im_0 = EF.EdfFile(A[0])
sze = list(np.shape(im_0.GetData(0)))
sum_bg = np.zeros(sze)
int_im = np.zeros(len(A))

# Measure the intensity for each image
for ii in range(0,27):#len(A)):
    im = EF.EdfFile(A[ii])
    image = im.GetData(0)
    sum_bg[:,:] = sum_bg[:,:] + image[:,:]
    sum_image = 0
    for jj in range(sze[0]):
        for kk in range(sze[1]):
            sum_image = sum_image + image[jj,kk]
    int_im[ii] = sum_image

# Show the image sum and the intensity histogram
plt.imshow(sum_bg)
plt.show()

plt.bar(range(len(A)), int_im)
plt.show()
