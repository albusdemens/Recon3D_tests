import numpy as np
import matplotlib.pyplot as plt
import lib.EdfFile as EF
import sys
import fabio

A = np.load('/u/data/alcer/DFXRM_rec/Rec_test/Fabio_array.npy')

Im_stack = A[4,14,1,:,:]
Im_raw = fabio.open("/u/data/andcj/hxrm/Al_april_2017/topotomo/sundaynight/topotomo_frelon_far_0012_0000_0001.edf")

#AA = Im_stack[1:10, 1:10]
#BB = Im_raw[100:120, 100:120]

#plt.subplot(122)
#plt.imshow(BB)

#plt.show()

#print Im_stack[5,3], Im_raw[5+107, 3+107]

Diff = np.zeros([Im_stack.shape[0], Im_stack.shape[1]])
for i in range(Im_stack.shape[0]):
    for j in range(Im_stack.shape[1]):
        diff = int(Im_stack[i,j]) - int(Im_raw.data[i+106, j+106])
        Diff[i,j] = diff
        #print Im_stack[i,j], Im_raw[i+106, j+106], Diff[i,j]

Diff_num = sum(sum(Diff))

fig = plt.figure()
#plt.imshow(Im_stack)
#plt.title('Image from stack')
#plt.show()
#sys.exit()

plt.subplot(131)
plt.imshow(Im_raw.data[106:406, 106:406])
plt.title('Raw image')

plt.subplot(132)
plt.imshow(Im_stack)
plt.title('Image from stack')

plt.subplot(133)
plt.imshow(Diff)
plt.title('Difference')

plt.show()
