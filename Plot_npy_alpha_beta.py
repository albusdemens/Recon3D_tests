# Script to visualize the files returned by getdata.py
# Command: Plot_results.py path/Input_folder
# Where the Input_folder is the folder with the npy files

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

'''
Input : directory of data
'''

plt.close("all")

dir = sys.argv[1]

alpha = np.load(os.path.join(dir,'alpha.npy'))
beta =  np.load(os.path.join(dir,'beta.npy'))

ar = range(len(alpha))
br = range(len(beta))

# Plot distribution of motor values
plt.figure(1)

plt.subplot(221)
plt.plot(alpha)
plt.xlabel('Acquisition number')
plt.ylabel('Alpha (in degrees)')
plt.title('Distribution of alpha angles')

plt.subplot(222)
plt.plot(beta)
plt.xlabel('Acquisition number')
plt.ylabel('Beta (in degrees)')
plt.title('Distribution of beta angles')

plt.subplot(223)
plt.plot(alpha, ar)
plt.xlabel('Alpha (in degrees)')
plt.ylabel('Acquisition number')

plt.subplot(224)
plt.plot(beta, br)
plt.xlabel('Beta (in degrees)')
plt.ylabel('Acquisition number')

plt.show()
