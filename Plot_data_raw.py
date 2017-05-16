# Plot, for a given omega, angular values before binning

import numpy as np
import matplotlib.pyplot as plt

Data = np.load('/Users/Alberto/Documents/Data_analysis/DFXRM/Results/All_data.npy')

# Plot distribution of motor values
alpha = Data[:,0]
beta = Data[:,1]

plt.figure(1)

plt.subplot(121)
plt.plot(alpha)
plt.xlabel('Acquisition number')
plt.ylabel('Alpha (in degrees)')
plt.title('Distribution of alpha angles')

plt.subplot(122)
plt.plot(beta)
plt.xlabel('Acquisition number')
plt.ylabel('Beta (in degrees)')
plt.title('Distribution of beta angles')

plt.show()
