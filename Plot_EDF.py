# Simple script to visualize an EDF file

import numpy as np
import matplotlib.pyplot as plt
import lib.EdfFile as EF
import sys

im = EF.EdfFile(sys.argv[1])
<<<<<<< HEAD
print im.GetHeader(0)

sys.exit()

image = np.array(im.GetData(0))
image_data = im.GetHeader(0)

im_flip = np.fliplr(image)

fig = plt.figure()
=======
image = im.GetData(0)
#im_flip = np.fliplr(image)

#fig = plt.figure()
>>>>>>> 077febf5ee14058f5f57ce942f86a24dc78c3637
#a=fig.add_subplot(1,2,1)
#imgplot = plt.imshow(image)
#a.set_title('Image')
#a=fig.add_subplot(1,2,2)
#imgplot = plt.imshow(im_flip)
#a.set_title('Flipped image')

<<<<<<< HEAD
#plt.imshow(image[100:400, 100:400])
#plt.show()

img = EF.EdfFile(sys.argv[1])
im = img.GetData(0).astype(float)
plt.imshow(im[106:406, 106:406])
plt.show()

sys.exit()

#plt.imshow(im_flip)
#plt.show()

# Consider layer from stored array
A = np.load('/u/data/alcer/DFXRM_rec/Rec_test/dataarray.npy')
B = A[5, 59, 0, :, :]
C = B.reshape(200,200)

# Load the background from getdata.py and play with it
bckg = np.load('bckg_roi.npy')

raw_img_cropped=image[256-100:256+100, 256-100:256+100]
clean_1=raw_img_cropped-bckg
clean_2=raw_img_cropped/bckg

fig = plt.figure()
# Raw image
a=fig.add_subplot(2,4,1)
plt.imshow(bckg)
a.set_title('ROI background')
b=fig.add_subplot(2,4,2)
plt.imshow(raw_img_cropped)
b.set_title('Raw image')
c=fig.add_subplot(2,4,3)
plt.imshow(clean_1)
c.set_title('Raw - background')
d=fig.add_subplot(2,4,4)
plt.imshow(clean_2)
d.set_title('Raw / background')
# Image from npy
a=fig.add_subplot(2,4,5)
plt.imshow(bckg)
a.set_title('ROI background')
b=fig.add_subplot(2,4,6)
plt.imshow(C)
b.set_title('Npy image')
c=fig.add_subplot(2,4,7)
plt.imshow(C/bckg)
c.set_title('Clean npy')
plt.show()

sys.exit()

fig = plt.figure()
a=fig.add_subplot(1,2,1)
=======
>>>>>>> 077febf5ee14058f5f57ce942f86a24dc78c3637
plt.imshow(image)
a.set_title('Raw image')
b=fig.add_subplot(1,2,2)
plt.imshow(B)
b.set_title('Image from npy array')
plt.show()
<<<<<<< HEAD

int_interval = np.empty(shape=[A.shape[0], A.shape[1]])
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        int_interval[i,j] = sum(sum(A[i,j,0,:,:]))

# Plot intensity for theta values
c = plt.figure()
plt.imshow(int_interval)
plt.xlabel('Theta')
plt.ylabel('Gamma')
plt.show()



=======
#plt.imshow(im_flip)
#plt.show()
>>>>>>> 077febf5ee14058f5f57ce942f86a24dc78c3637
