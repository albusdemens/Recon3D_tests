import numpy as np
import matplotlib.pyplot as plt
import sys

# Load data before cleaning
A = np.load('/u/data/alcer/DFXRM_rec/Rec_test/dataarray.npy')
# Load data after cleaning
B = np.load('/u/data/alcer/DFXRM_rec/Rec_test/dataarray_clean.npy')

# Select the gamma, mu combination
aa = 3
bb = 5

# For the higher intensity detector, study how the integrated intensity evolves
# as the sample rotates
fig = plt.figure()
im_count = 0
for oo in range(0, A.shape[2], 20):
    hist_counter = np.zeros([A.shape[4], 3])
    for k in range(A.shape[4]):
        hist_counter[k,0] = k
        hist_counter[k, 1] = sum(A[aa,bb,oo,:,k])
        hist_counter[k, 2] = sum(B[aa,bb,oo,:,k])
    plt.subplot(1,2,1)
    plt.ylim(9E4, 1E5)
    plt.grid(True)
    plt.plot(hist_counter[:,0], hist_counter[:,1], 'o', markersize=1.5)
    plt.title('Raw data')
    plt.xlabel('Y pixel')
    plt.ylabel('Summed intensity')
    plt.subplot(1,2,2)
    plt.ylim(5E3, 8E3)
    plt.grid(True)
    plt.plot(hist_counter[:,0], hist_counter[:,2], 'o', markersize=1.5)
    plt.title('Cleaned data')
    plt.xlabel('Y pixel')
    plt.ylabel('Summed intensity')
plt.show()

fig = plt.figure()
im_count = 0
for oo in range(0, A.shape[2], 20):
    hist_counter = np.zeros([A.shape[4], 3])
    for k in range(A.shape[3]):
        hist_counter[k,0] = k
        hist_counter[k, 1] = sum(A[aa,bb,oo,k,:])
        hist_counter[k, 2] = sum(B[aa,bb,oo,k,:])
    plt.subplot(1,2,1)
    plt.ylim(9E4, 1E5)
    plt.plot(hist_counter[:,0], hist_counter[:,1], 'o', markersize=1.5)
    plt.title('Summed intensity along X')
    plt.xlabel('X pixel')
    plt.ylabel('Summed intensity')
    plt.subplot(1,2,2)
    plt.ylim(5E3, 8E3)
    plt.plot(hist_counter[:,0], hist_counter[:,2], 'o', markersize=1.5)
    plt.title('Cleaned data')
    plt.xlabel('X pixel')
    plt.ylabel('Summed intensity')
plt.show()

### Look at the distribution on the integrated intensity along Y
omega = 5

# Plot intensity sum along Y for the raw data
fig = plt.figure()
plt.title('Summed intensity along Y, raw data, one proj')
im_count = 0
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        hist_counter = np.zeros([A.shape[4], 2])
        im_count = im_count + 1
        for k in range(A.shape[4]):
            hist_counter[k,0] = k
            hist_counter[k, 1] = sum(A[i,j,omega,:,k])

        plt.subplot(A.shape[0], A.shape[1], im_count)
        plt.scatter(hist_counter[:,0], hist_counter[:,1], s=1)

plt.show()

# Plot intensity sum along Y for the cleaned data
fig = plt.figure()
plt.title('Summed intensity along Y, cleaned data, one proj')
im_count = 0
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        hist_counter = np.zeros([A.shape[4], 2])
        im_count = im_count + 1
        for k in range(A.shape[4]):
            hist_counter[k,0] = k
            if sum(B[i,j,omega,:,k]) < 5000:
                hist_counter[k, 1] = sum(B[i,j,omega,:,k])

        plt.subplot(A.shape[0], A.shape[1], im_count)
        plt.scatter(hist_counter[:,0], hist_counter[:,1], s=1)

plt.show()

# Plot intensity sum along Y for the raw data. All projections considered
fig = plt.figure()
plt.title('Summed intensity along Y, raw data, all proj')
im_count = 0
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        hist_counter = np.zeros([A.shape[4], 2])
        im_count = im_count + 1
        for k in range(A.shape[4]):
            hist_counter[k,0] = k
            hist_counter[k, 1] = np.sum(A[i,j,:,:,k])

        plt.subplot(A.shape[0], A.shape[1], im_count)
        plt.scatter(hist_counter[:,0], hist_counter[:,1], s=1)

plt.show()

# Plot intensity sum along Y for the cleaned data. All projections considered
fig = plt.figure()
plt.title('Summed intensity along Y, cleaned data, all proj')
im_count = 0
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        hist_counter = np.zeros([A.shape[4], 2])
        im_count = im_count + 1
        for k in range(A.shape[4]):
            hist_counter[k,0] = k
            hist_counter[k, 1] = np.sum(B[i,j,:,:,k])

        plt.subplot(A.shape[0], A.shape[1], im_count)
        plt.scatter(hist_counter[:,0], hist_counter[:,1], s=1)

plt.show()
