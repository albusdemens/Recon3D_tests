import numpy as np
import matplotlib.pyplot as plt

A = np.load('/u/data/alcer/DFXRM_rec/Rec_test/dataarray.npy')
B = np.load('/u/data/alcer/DFXRM_rec/Rec_test/dataarray_clean.npy')

fig = plt.figure()
plt.subplot(1,2,1)
plt.imshow(A[3,3,1,:,:])
plt.subplot(1,2,2)
plt.imshow(B[3,3,1,:,:])
plt.show()
