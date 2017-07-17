# this script plots the distribution of the angular values fro one omega

import numpy as np
import matplotlib.pyplot as plt

A = np.load('/u/data/alcer/DFXRM_rec/Rec_test/dataarray.npy')

Counter = np.zeros([A.shape[0], A.shape[1]])
for aa in range(A.shape[0]):
    for bb in range(A.shape[1]):
<<<<<<< HEAD
        if sum(sum(A[aa,bb,5,:,:])) > 0:
=======
        if sum(sum(A[aa,bb,1,:,:])) > 0:
>>>>>>> 077febf5ee14058f5f57ce942f86a24dc78c3637
            Counter[aa, bb] = 1

fig = plt.figure()
plt.imshow(Counter)
plt.show()
