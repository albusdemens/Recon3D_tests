# Simple script to visualize an EDF file

import numpy as np
import matplotlib.pyplot as plt
import lib.EdfFile as EF

im = EF.EdfFile('/Users/Alberto/Documents/Data_analysis/DFXRM/topotomo_frelon_far_0022_0000_0002.edf')
image = im.GetData(0)
plt.imshow(image)
plt.show()
