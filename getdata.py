from lib.miniged import GetEdfData
import sys
import time
import os
import warnings
import numpy as np
import pdb 	# For debugging

try:
	from mpi4py import MPI
except ImportError:
	print "No MPI, running on 1 core."

'''
Inputs:
Directory of data
Name of data files
Directory of background files
Name of background files
Point of interest
Image size
Output path
Name of new output directory to make
'''


class makematrix():
	def __init__(
		self, datadir,
		dataname, bgpath, bgfilename,
		poi, imgsize, outputpath, outputdir,
		sim=False):

		try:
			self.comm = MPI.COMM_WORLD
			self.rank = self.comm.Get_rank()
			self.size = self.comm.Get_size()
		except NameError:
			self.rank = 0
			self.size = 1

		imgsize = imgsize.split(',')
		poi = poi.split(',')

		if self.rank == 0:
			start = time.time()
			self.directory = self.makeOutputFolder(outputpath, outputdir)

		roi = [
			int(int(poi[0]) - int(imgsize[0]) / 2),
			int(int(poi[0]) + int(imgsize[0]) / 2),
			int(int(poi[1]) - int(imgsize[1]) / 2),
			int(int(poi[1]) + int(imgsize[1]) / 2)]

		data = GetEdfData(datadir, dataname, bgpath, bgfilename, roi, sim)
		self.alpha, self.beta, self.omega = data.getMetaValues() # Redefine
		min_alpha = min(self.alpha)
		max_alpha = max(self.alpha)
		min_beta = min(self.beta)
		max_beta = max(self.beta)
		print min_alpha, max_alpha, min_beta, max_beta

		pdb.set_trace()

		print self.alpha
		print self.beta
		print self.omega
		print len(self.alpha)
		print len(self.beta)
		print len(self.omega)
		self.allFiles(data, imgsize)

		if self.rank == 0:
			stop = time.time()
			print 'Total time: {0:8.4f} seconds.'.format(stop - start)

	def makeOutputFolder(self, path, dirname):
		directory = path + '/' + dirname

		print directory
		if not os.path.exists(directory):
			os.makedirs(directory)
		return directory

	def allFiles(self, data, imsiz):
		index_list = range(len(data.meta))
		met = data.meta

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			imgarray = data.makeImgArray(index_list, 50, 'linetrace')

		if self.rank == 0:
			lena = 7#len(self.alpha)
			lenb = 7#len(self.beta)
			leno = len(self.omega)
			#lena = len(self.alpha)
			#lenb = len(self.beta)
			#leno = len(self.omega)

			bigarray = np.zeros((lena, lenb, leno, int(imsiz[1]), int(imsiz[0])))

			for i, ind in enumerate(index_list):
				a = np.where(self.alpha == met[ind, 0])
				b = np.where(self.beta == met[ind, 1])
				c = np.where(self.omega == met[ind, 2])
				# d = np.where(self.theta == met[ind, 4])

				bigarray[a, b, c, :, :] = imgarray[ind, :, :]

			np.save(self.directory + '/alpha.npy', self.alpha)
			np.save(self.directory + '/beta.npy', self.beta)
			# np.save(self.directory + '/theta.npy', self.theta)
			np.save(self.directory + '/omega.npy', self.omega)

			np.save(self.directory + '/dataarray.npy', bigarray)


if __name__ == "__main__":
	if len(sys.argv) != 9:
		if len(sys.argv) != 10:
			print "Not enough input parameters. Data input should be:\n\
	Directory of data\n\
	Name of data files\n\
	Directory of background files\n\
	Name of background files\n\
	Point of interest\n\
	Image size\n\
	Output path\n\
	Name of new output directory to make\n\
		"
		else:
			mm = makematrix(
				sys.argv[1],
				sys.argv[2],
				sys.argv[3],
				sys.argv[4],
				sys.argv[5],
				sys.argv[6],
				sys.argv[7],
				sys.argv[8],
				sys.argv[9])
	else:
		mm = makematrix(
			sys.argv[1],
			sys.argv[2],
			sys.argv[3],
			sys.argv[4],
			sys.argv[5],
			sys.argv[6],
			sys.argv[7],
			sys.argv[8])
