# Simple script to visualize an EDF file

import numpy as np
import matplotlib.pyplot as plt
import lib.EdfFile as EF

im = EF.EdfFile('/Users/Alberto/Documents/Data_analysis/DFXRM/topotomo_frelon_far_0022_0000_0002.edf')
image = im.GetData(0)
#im_flip = np.fliplr(image)

#fig = plt.figure()
#a=fig.add_subplot(1,2,1)
#imgplot = plt.imshow(image)
#a.set_title('Image')
#a=fig.add_subplot(1,2,2)
#imgplot = plt.imshow(im_flip)
#a.set_title('Flipped image')

plt.imshow(image)
plt.show()
#plt.imshow(im_flip)
#plt.show()
