from lib.miniged import GetEdfData
import sys

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
		poi, imgsize, outputpath, outputdir):

		initminiged()
		print self.alpha
		print self.beta
		print self.omega

	def initminiged(self):
		data = GetEdfData(path, filename, bg_path, bg_filename, roi, datatype)
		self.alpha, self.beta, self.omega = data.getMetaValues()

	def allFiles(self):
		index_list = range(len(data.meta))

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			imgarray = data.makeImgArray(index_list, 50, 'linetrace')

		if rank == 0:
			np.save('/u/data/andcj/tmp/alpha.npy', self.alpha)
			np.save('/u/data/andcj/tmp/beta.npy', self.beta)
			np.save('/u/data/andcj/tmp/omega.npy', self.omega)

			reshapedarray = np.reshape(imgarray,[41, 181, np.shape(imgarray)[1], np.shape(imgarray)[2]])

			np.save('/u/data/andcj/tmp/largetest_200x200.npy', reshapedarray)


if __name__ == "__main__":
	print sys.argv
	print len(sys.argv)
	if len(sys.argv) != 9:
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
			sys.argv[8],)
