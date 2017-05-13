# Visualize the sample volume reconstructed by recon3D.py

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import mayavi.mlab as mlab

'''
Input : directory of data
'''

plt.close("all")

dir = sys.argv[1]
Data = np.load(os.path.join(dir,'grain_ang.npy'))

print Data
