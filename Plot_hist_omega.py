# This script tests various cleaning options, in preparation of the morphological
# operations

import sys
import numpy as np
import matplotlib.pyplot as plt

A = np.load('/u/data/alcer/DFXRM_rec/Rec_test/dataarray_clean_norm.npy')

omega = 0
i = 3
j = 3

img = np.zeros([A.shape[3], A.shape[4]])
img_clean_max = np.zeros([A.shape[3], A.shape[4]])
img_clean_mean = np.zeros([A.shape[3], A.shape[4]])
img[:,:] = A[i,j,omega,:,:]
arr  = np.asarray(img)
flat = arr.reshape(np.prod(arr.shape[:2]),-1)
n, bins, patches = plt.hist(flat, range(max(flat)))
max_value = np.amax(np.array(n)[n != np.amax(n)])
val_max = int(np.where(n == max_value)[0])
val_mean = int(np.mean(img))

img_clean_max[:,:] = A[3,3,omega,:,:] - val_max
img_clean_mean[:,:] = A[3,3,omega,:,:] - val_mean

img_clean_max[img_clean_max < 0] = 0
img_clean_mean[img_clean_mean < 0] = 0
img_clean_max[img_clean_max > 6E4] = 0
img_clean_mean[img_clean_mean > 6E4] = 0

fig = plt.figure()
plt.subplot(1,3,1)
plt.imshow(img)
plt.subplot(1,3,2)
plt.imshow(img_clean_max)
plt.subplot(1,3,3)
plt.imshow(img_clean_mean)

plt.show()

sys.exit()

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        img = np.zeros([A.shape[3], A.shape[4]])
        img[:,:] = A[i,j,omega,:,:]
        arr  = np.asarray(img)
        flat = arr.reshape(np.prod(arr.shape[:2]),-1)
        n, bins, patches = plt.hist(flat, range(max(flat)))
        max_value = np.amax(np.array(n)[n != np.amax(n)])
        print int(np.where(n == max_value)[0]), int(np.mean(img))

#plt.show()
