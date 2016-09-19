# !/bin/python
"""Class for loading DFXM data sets.



The class can be loaded from another Python file. This gives access to all
metadata in the data set as well as direct access to an image by giving either
coordinates or an index.

A number of packages are required:

numpy
scipy
EdfFile
matplotlib
seaborn (For prettier plots)
mpi4py (For parallel tasks)


To use the class in another Python file:

	from lib.getedfdata import *

where getedfdata.py is in a sub directory called 'lib'.


An example of using the class can be found in the test function at the bottom
of this file.
"""

import os
import math
import numpy as np
import EdfFile
import warnings
import time

from os import listdir
from os.path import isfile, join

from time import localtime, strftime

from mpi4py import MPI


class GetEdfData(object):

	"""Initialization of GetEdfData.

	The class is initialized with:
	path: Path to data.
	filename: Beginning of the filename of the datafiles.
	bg_filename: Beginning of the filename of the background files.
	roi: A tuple of [x1, x2, y1, y2], designating region of interest.
	datatype: Either 'strain_tt', 'strain_eta' or 'topotomo'. Decides from
		which motor to get the 2nd value (not theta).

	A folder is created in $WORKING_DIR/output/ with the name of current date
		and time. In that dir a txt file is put with information about datatype
		sampletitle, path and ROI. The file works as a log, so the user can put
		in any information that is necessary.
	"""

	def __init__(
		self,
		path,
		filename,
		bg_path,
		bg_filename,
		roi,
		datatype):
		super(GetEdfData, self).__init__()

		self.comm = MPI.COMM_WORLD
		self.rank = self.comm.Get_rank()
		self.size = self.comm.Get_size()

		self.datatype = datatype
		self.sampletitle = filename
		self.path = path
		self.bg_path = bg_path
		self.roi = roi
		self.ts = test_switch

		self.getFilelists(filename, bg_filename)
		#
		# self.dirhash = hashlib.md5(
		# 	self.path + '/' +
		# 	filename +
		# 	str(len(self.data_files))).hexdigest()
		# print self.dirhash
		# print self.path + '/' + filename

		self.getBGarray()
		self.getMetaData()

	def setTest(self, testcase):
		self.test = testcase

	def adjustOffset(self, case):
		self.adjustoffset = case

	def getFilelists(self, filename, bg_filename):
		onlyfiles = [f for f in listdir(self.path) if isfile(join(self.path, f))]
		onlyfiles_bg = [
			f for f in listdir(self.bg_path) if
			isfile(join(self.bg_path, f))]

		self.data_files = []
		self.bg_files = []

		for k in onlyfiles[:]:
			if k[:len(filename)] == filename:
				self.data_files.append(k)

		for k in onlyfiles_bg[:]:
			if k[:len(bg_filename)] == bg_filename:
				self.bg_files.append(k)

	def getROI(self):
		return self.roi

	def setROI(self, roi):
		self.roi = roi

	def getBGarray(self):
		bg_file_with_path = self.bg_path + '/' + self.bg_files[0]
		bg_class = EdfFile.EdfFile(bg_file_with_path)
		bg_img = bg_class.GetData(0).astype(np.int64)[
			self.roi[2]:self.roi[3], self.roi[0]:self.roi[1]]

		self.bg_combined = np.zeros(np.shape(bg_img))

		if self.rank == 0:
			print "Reading background files (ROI)..."

		for i in range(len(self.bg_files)):
			bg_file_with_path = self.bg_path + '/' + self.bg_files[i]
			bg_class = EdfFile.EdfFile(bg_file_with_path)
			self.bg_combined += bg_class.GetData(0).astype(np.int64)[
				self.roi[2]:self.roi[3], self.roi[0]:self.roi[1]]

		self.bg_combined /= len(self.bg_files)

		bg_img_full = bg_class.GetData(0).astype(np.int64)
		self.bg_combined_full = np.zeros(np.shape(bg_img_full))

		if self.rank == 0:
			print "Reading background files (Full)..."

		for i in range(len(self.bg_files)):
			bg_file_with_path = self.bg_path + '/' + self.bg_files[i]
			bg_class = EdfFile.EdfFile(bg_file_with_path)
			self.bg_combined_full += bg_class.GetData(0).astype(np.int64)

		self.bg_combined_full /= len(self.bg_files)

	def getIndexList(self):
		file_with_path = self.path + '/' + self.data_files[0]
		img = EdfFile.EdfFile(file_with_path)
		header = img.GetHeader(0)

		indexlist = []

		for ind in header.keys():
			if ind != 'motor_pos' and ind != 'motor_mne' and ind != 'counter_pos' and ind != 'counter_mne':
				indexlist.append(ind)
			# if ind == 'motor_mne':
			# if ind == 'counter_mne':
		indexlist.extend(header['motor_mne'].split(' '))
		try:
			indexlist.extend(header['counter_mne'].split(' '))
		except KeyError:
			pass

		self.indexlist = indexlist

	def getHeader(self, filenumber):
		file_with_path = self.path + '/' + self.data_files[filenumber]
		img = EdfFile.EdfFile(file_with_path)
		header = img.GetHeader(0)

		mot_array = header['motor_mne'].split(' ')
		motpos_array = header['motor_pos'].split(' ')

		try:
			det_array = header['counter_mne'].split(' ')
			detpos_array = header['counter_pos'].split(' ')
		except KeyError:
			det_array = []
			detpos_array = []

		try:
			srcur = header['machine current'].split(' ')[-2]
		except KeyError:
			print "KeyError"
			srcur = 0

		return mot_array, motpos_array, det_array, detpos_array, srcur

	def getFullHeader(self, filenumber):
		file_with_path = self.path + '/' + self.data_files[filenumber]
		img = EdfFile.EdfFile(file_with_path)
		header = img.GetHeader(0)

		metalist = []
		# indexlist = []
		#
		# a = 0

		for ind in header.keys():
			if ind != 'motor_pos' and ind != 'motor_mne' and ind != 'counter_pos' and ind != 'counter_mne':
				metalist.append(header[ind])

		metalist.extend(header['motor_pos'].split(' '))
		try:
			metalist.extend(header['counter_pos'].split(' '))
		except:
			pass

			# if filenumber == 0:
			# 	if ind != 'motor_pos' and ind != 'motor_mne' and
			# 	ind != 'counter_pos' and ind != 'counter_mne':
			# 		indexlist.append(ind)
			# 	# if ind == 'motor_mne':
			# 	# if ind == 'counter_mne':
			# 	indexlist.extend(header['motor_mne'].split(' '))
			# 	indexlist.extend(header['counter_mne'].split(' '))
			#
			# 	self.indexlist = indexlist

		# print indexlist

		return metalist

	def readMetaFile(self, metadatafile):
		while True:
			time.sleep(0.5)
			try:
				self.meta = np.loadtxt(metadatafile)
				if len(self.meta) == len(self.data_files):
					break
			except ValueError:
				pass

	def readFullMetaFile(self, fullmetadatafile, metadatafile, indexfile):
		self.indexlist = np.load(indexfile).tolist()
		while True:
			time.sleep(0.5)
			try:
				self.fma = np.load(fullmetadatafile)
				self.meta = np.load(metadatafile)
				if len(self.fma) == len(self.data_files):
					break
			except ValueError:
				pass

	def makeFullMetaArray(self):
		self.meta = np.zeros((len(self.data_files), 4))

		if self.rank == 0:
			print "Reading meta data..."

		self.fma = []
		self.getIndexList()

		for i in range(len(self.data_files)):
			self.fma.append(self.getFullHeader(i))

	# def makeMetaArray(self):
	# 	self.meta = np.zeros((len(self.data_files), 4))
	#
	# 	if self.rank == 0:
	# 		print "Reading meta data..."
	#
	# 	for i in range(len(self.data_files)):
	# 		try:
	# 			mot_array, motpos_array, det_array, detpos_array, srcur = self.getHeader(i)
	# 		except ValueError:
	# 			mot_array, motpos_array, det_array, detpos_array = self.getHeader(i)
	# 			self.meta[i, 3] = round(float(detpos_array[det_array.index('srcur')]), 5)
	#
	# 		if self.datatype == 'topotomo':
	# 			self.meta[i, 0] = round(float(motpos_array[mot_array.index('diffrx')]), 8)
	# 			self.meta[i, 1] = round(float(motpos_array[mot_array.index('diffrz')]), 8)
	# 			self.meta[i, 2] = float(self.data_files[i][-8:-4])
	#
	# 		if self.datatype == 'strain_eta':
	# 			theta = (11.006 - 10.986) / 40
	# 			self.meta[i, 0] = round(float(motpos_array[mot_array.index('obpitch')]), 8)
	# 			self.meta[i, 1] = round(
	# 				10.986 + theta *
	# 				(float(self.data_files[i][-8:-4])) +
	# 				theta / 2, 8)
	# 			self.meta[i, 2] = float(self.data_files[i][-8:-4])
	#
	# 		if self.datatype == 'strain_tt':
	# 			self.meta[i, 0] = round(float(motpos_array[mot_array.index('obyaw')]), 8)
	# 			self.meta[i, 1] = round(float(motpos_array[mot_array.index('diffrz')]), 8)
	# 			self.meta[i, 2] = round(float(motpos_array[mot_array.index('diffrx')]), 8)
	#
	# 		if self.datatype == 'mosaicity':
	# 			self.meta[i, 0] = round(float(motpos_array[mot_array.index('samry')]), 8)
	# 			self.meta[i, 1] = round(float(motpos_array[mot_array.index('samrz')]), 8)
	# 			self.meta[i, 2] = round(float(motpos_array[mot_array.index('diffrx')]), 8)

	def makeMetaArrayNew(self):
		self.meta = np.zeros((len(self.data_files), 4))

		if self.rank == 0:
			print "Making meta array."

		for i in range(len(self.fma)):

			mot_array, motpos_array, det_array, detpos_array, srcur = self.getHeader(i)

			self.meta[i, 0] = round(float(motpos_array[mot_array.index('samrz')]), 8)
			self.meta[i, 1] = round(float(motpos_array[mot_array.index('samry')]), 8)
			self.meta[i, 2] = round(float(motpos_array[mot_array.index('diffrx')]), 8)
			self.meta[i, 2] = srcur

		self.meta = np.around(self.meta, decimals=8)

	def getMetaData(self):
		# fullmetadatafile = 'tmp/datafullmeta_%s.npy' % self.dirhash
		# metadatafile = 'tmp/datameta_%s.txt' % self.dirhash
		# indexfile = 'tmp/dataindex_%s.npy' % self.dirhash

		print "Starting meta data collection."
		self.makeMetaArrayNew()
		#
		# if os.path.isfile(metadatafile):
		# 	print "Reading meta data from file."
		# 	self.readMetaFile(metadatafile)
		#
		# else:
		# 	print "Making meta data file."
		# 	self.makeFullMetaArray()
		# 	self.makeMetaArrayNew()
		# 	np.savetxt(metadatafile, self.meta)
		# 	np.save(indexfile, self.indexlist)
		# 	np.save(fullmetadatafile, self.fma)

		alphavals = sorted(list(set(self.meta[:, 0])))
		betavals = sorted(list(set(self.meta[:, 1])))
		gammavals = sorted(list(set(self.meta[:, 2])))
		self.alphavals = np.zeros((len(alphavals)))
		self.betavals = np.zeros((len(betavals)))
		self.gammavals = np.zeros((len(gammavals)))
		for i in range(len(alphavals)):
			self.alphavals[i] = float(alphavals[i])
		for i in range(len(betavals)):
			self.betavals[i] = float(betavals[i])
		for i in range(len(gammavals)):
			self.gammavals[i] = float(gammavals[i])

		self.alpha0 = self.alphavals[len(self.alphavals) / 2]
		self.beta0 = self.betavals[len(self.betavals) / 2]
		self.gamma0 = self.gammavals[len(self.gammavals) / 2]

		if self.rank == 0:
			print "Meta data from %s files read." % str(len(self.data_files))

	def rebin(self, a, bs):
		shape = (a.shape[0] / bs, a.shape[1] / bs)
		sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
		return a.reshape(sh).sum(-1).sum(1)

	def line(self, x, a, b):
		return a * x + b

	def fitLine(self, x, y):
		from scipy.optimize import curve_fit

		try:
			popt, pcov = curve_fit(self.line, x, y, p0=[0, 30], maxfev=10000)
			return popt, pcov
		except RuntimeError:
			pass

	def getIndex(self, alpha, beta, gamma):
		if alpha != -10000 and beta == -10000:
			index = np.where(self.meta[:, 0] == alpha)
		if alpha == -10000 and beta != -10000:
			index = np.where(self.meta[:, 1] == beta)
		if alpha != -10000 and beta != -10000 and gamma == -10000:
			i1 = np.where(self.meta[:, 0] == alpha)
			i2 = np.where(self.meta[:, 1] == beta)
			index = list(set(i1[0]).intersection(i2[0]))
		if alpha != -10000 and beta != -10000 and gamma != -10000:
			i1 = np.where(self.meta[:, 0] == alpha)
			i2 = np.where(self.meta[:, 1] == beta)
			index_ab = list(set(i1[0]).intersection(i2[0]))
			i3 = np.where(self.meta[:, 2] == gamma)
			index = list(set(index_ab).intersection(i3[0]))
		return index

	def getImage(self, index, full):
		file_with_path = self.path + '/' + self.data_files[index]
		if self.rank == 0:
			print file_with_path

		if True:
			img = EdfFile.EdfFile(file_with_path)
			if self.adjustoffset:
				alpha = self.meta[index, 0]
				a_index = np.where(self.alphavals == alpha)
				roi = self.adj_array[a_index[0]][0]
			else:
				roi = self.roi

			if full:
				im = img.GetData(0).astype(np.int64) - self.bg_combined_full
			else:
				im = img.GetData(0).astype(np.int64)[
					roi[2]:roi[3],
					roi[0]:roi[1]] - self.bg_combined

			im = self.cleanImage(im)

		return im

	def cleanImage(self, img):
		# img = self.rfilter(img, 18, 3)
		img[img < 0] = 0

		return img

	def smooth(self, a, n):
		"""Do a moving average."""
		ret = np.cumsum(a, dtype=float)
		ret[n:] = ret[n:] - ret[:-n]
		return ret / n

	def rfilter(self, img, nstp, slen):
		start = time.time()
		mask = np.ones(np.shape(img))
		stp = 180. / nstp

		stack = np.zeros((len(img[:, 0]), len(img[0, :]), nstp))

		def smooth(a, n):
			"""Do a moving average."""
			ret = np.cumsum(a, dtype=float)
			ret[n:] = ret[n:] - ret[:-n]
			return ret / n

		for n in xrange(nstp):
			rot = n * stp
			imgr = scipy.ndimage.interpolation.rotate(img, rot)
			mskr = scipy.ndimage.interpolation.rotate(mask, rot)

			for j, xvals in enumerate(mskr[:, 0]):  # range(len(mskr[:, 0])):
				ids = np.nonzero(mskr[j, :])
				if ids[0].any():
					imgr[j, ids[0]] = smooth(imgr[j, ids[0]], slen)

			imgb = scipy.ndimage.interpolation.rotate(imgr, -rot)

			idx0 = [
				np.floor((len(imgb[:, 0]) - len(img[:, 0])) / 2),
				np.floor((len(imgb[0, :]) - len(img[0, :])) / 2)]
			idx1 = [
				np.floor((len(imgb[:, 0]) + len(img[:, 0])) / 2),
				np.floor((len(imgb[0, :]) + len(img[0, :])) / 2)]

			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				imgb = imgb[idx0[0]:idx1[0], idx0[1]:idx1[1]]

			stack[:, :, n] = imgb
		end = time.time()
		print "Rfilter took:", end - start, "seconds for an array."
		return np.amin(stack, 2)

	def getMetaValues(self):
		return self.alphavals, self.betavals, self.gammavals

	def getMetaArray(self):
		return self.meta

	def getBG(self):
		return self.bg_combined

	def makeImgArray(self, index, xpos, savefilename):
		img = self.getImage(index[0], False)
		imgarray = np.zeros((len(index), len(img[:, 0]), len(img[0, :])))

		def addToArray(index_part):
			imgarray_part = np.zeros((len(index_part), len(img[:, 0]), len(img[0, :])))
			for i in range(len(index_part)):
				print "Adding image {} to array. (Rank {})".format(i, self.rank)

				img0 = self.getImage(index_part[i], False)
				imgsum = np.sum(img0, 1) / len(img0[0, :])

				# return imgarray_part[i, :, :], imgsum
				ran = np.array(range(len(imgsum)))
				popt, pcov = self.fitLine(ran, imgsum)
				fittedline = ran * popt[0] + popt[1]
				fittedline = fittedline - fittedline[len(fittedline) / 2]
				gradient = np.tile(fittedline, (len(img0[:, 0]), 1)).transpose()
				imgarray_part[i, :, :] = img0 - gradient

			imgarray_part[0, 0, 0] = self.rank
			return imgarray_part

		# Chose part of data set for a specific CPU (rank).
		local_n = len(index) / self.size
		istart = self.rank * local_n
		istop = (self.rank + 1) * local_n
		index_part = index[istart:istop]

		# Calculate strain on part of data set.
		imgarray_part = addToArray(index_part)

		# CPU 0 (rank 0) combines data parts from other CPUs.
		if self.rank == 0:
			# Make empty arrays to fill in data from other cores.
			recv_buffer = np.zeros((np.shape(imgarray_part)))
			# strainpic = np.zeros((np.shape(img)))

			datarank = imgarray_part[0, 0, 0]
			imgarray_part[0, 0, 0] = 0
			imgarray[istart:istop, :, :] = imgarray_part
			for i in range(1, self.size):
				self.comm.Recv(recv_buffer, MPI.ANY_SOURCE)
				datarank = int(recv_buffer[0, 0, 0])
				recv_buffer[0, 0, 0] = 0
				imgarray[
					datarank * local_n:(datarank + 1) * local_n, :, :] = recv_buffer

			return imgarray

		else:
			# all other process send their result
			self.comm.Send(imgarray_part, dest=0)

if __name__ == '__main__':
	pass
