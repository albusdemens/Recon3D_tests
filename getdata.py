from lib.miniged import GetEdfData
import sys
import time
import os
import warnings
import numpy as np
import pdb 	# For debugging
from find_nearest import find_nearest, find_nearest_idx

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
Initial phi values
Initial chi value
'''


class makematrix():
	def __init__(
		self, datadir,
		dataname, bgpath, bgfilename,
		poi, imgsize, outputpath, outputdir,
		phi_0, chi_0,
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

		self.alpha, self.beta, self.omega, self.theta = data.getMetaValues()

		self.index_list = range(len(data.meta))
		self.meta = data.meta
		self.phi = phi_0
		self.chi = chi_0

		# self.calcEtaIndexList(data, eta)

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
		# index_list = range(len(data.meta))
		# met = data.meta

		# theta = data.theta0 + np.arange(-3.5 * 0.032, 3.5 * 0.032, 0.032)

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			imgarray = data.makeImgArray(self.index_list, 50, 'linetrace')

		if self.rank == 0:

			# If we don't need to bin the angular values
			lena = len(self.alpha)
			lenb = len(self.beta)
			leno = len(self.omega)
			lent = len(self.theta)

			# Reduce alpha and beta to a single index_list
			index_list = np.zeros(len(self.meta))
			for j in range(len(self.meta)):
				alp = float(self.meta[j,0])
				omg = float(self.meta[j,2])
				phi_0 = float(self.phi)
				index_list[j] = int((alp - phi_0)/(np.cos(np.deg2rad(omg))*0.032))

			num_int = len(set(index_list))	# Number of considered alpha, beta
											# intervals
			idx_values = sorted(set(index_list))	# Values of the indices
			print idx_values

			print 'Check that the number of (phi, chi) steps is', num_int

			bigarray = np.zeros((num_int, leno, lent, int(imsiz[1]), int(imsiz[0])), dtype=np.uint16)

			for i, ind in enumerate(self.index_list):
				a = np.where(self.alpha == self.meta[ind,0])  	# rock
				b = np.where(self.beta == self.meta[ind,1])  	# roll
				c = np.where(self.omega == self.meta[ind,2])  	# omega
				d = np.where(self.theta == self.meta[ind,4])	# theta

				bigarray[idx_values, c, d, :, :] = imgarray[ind, :, :]

			np.save(self.directory + '/alpha.npy', self.alpha)
			np.save(self.directory + '/beta.npy', self.beta)
			np.save(self.directory + '/theta.npy', self.theta)
			np.save(self.directory + '/omega.npy', self.omega)
			np.save(self.directory + '/all_data.npy', self.meta)
			#np.save(self.directory + '/dataarray.npy', bigarray)


if __name__ == "__main__":
	if len(sys.argv) != 10:
		if len(sys.argv) != 11:
			print "Not enough input parameters. Data input should be:\n\
	Directory of data\n\
	Name of data files\n\
	Directory of background files\n\
	Name of background files\n\
	Point of interest\n\
	Image size\n\
	Output path\n\
	Name of new output directory to make\n\
	Initial phi value\n\
	Initial chi value\n\
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
				sys.argv[9],
				sys.argv[10])
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
			sys.argv[10],
			sys.argv[11])
