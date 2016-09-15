#!/bin/python
"""blah."""

from lib.getedfdata import *

from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpisize = comm.Get_size()

print rank, mpisize

if rank == 0:
	start = time.time()

path = '/u/data/andcj/hxrm/Dislocation_november_2015/diamond/ff_topo_2'
bg_path = '/u/data/andcj/hxrm/Dislocation_november_2015/diamond/bg_ff'

filename = 'ff3_'
# filename2 = 'ff2_'
sampletitle = 'fulltopo'
bg_filename = 'bg_ff_2x2_0p5s_'

datatype = 'topotomo'

poi = [512, 512]
size = [200, 200]

test_switch = True

roi = [poi[0]-size[0]/2, poi[0]+size[0]/2, poi[1]-size[1]/2, poi[1]+size[1]/2]

data = GetEdfData(path, filename, bg_path, bg_filename, roi, datatype, test_switch)
data.setTest(True)
data.adjustOffset(False)

try:
	directory = data.directory
except AttributeError:
	directory = 0

a, b, c = data.getMetaValues()

def allFiles(a, b):
	index_list = range(len(data.meta))

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		imgarray = data.makeImgArray(index_list, 50, 'linetrace')

	if rank == 0:
		np.save('/u/data/andcj/tmp/alpha.npy', a)
		np.save('/u/data/andcj/tmp/beta.npy', b)

		reshapedarray = np.reshape(imgarray,[41, 181, np.shape(imgarray)[1], np.shape(imgarray)[2]])

		np.save('/u/data/andcj/tmp/largetest_200x200.npy', reshapedarray)


if __name__ == "__main__":
	allFiles(a, b)
	if rank == 0:
		end = time.time()
		print "Time:", end-start, "seconds."
