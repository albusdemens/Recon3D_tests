from check_input import read as ini
import sys
import time
import numpy as np

try:
	from mpi4py import MPI
except ImportError:
	print "No MPI, running on 1 core."


class main():
	def __init__(self, inifile):
		self.par = self.getparameters(inifile)
		print self.par['M']

		try:
			self.comm = MPI.COMM_WORLD
			self.rank = self.comm.Get_rank()
			self.size = self.comm.Get_size()
		except NameError:
			self.rank = 0
			self.size = 1

		if self.rank == 0:
			start = time.time()

		self.readarrays()
		print np.shape(self.fullarray)
		print self.alpha

	def getparameters(self, inifile):
		checkinput = ini(inifile)
		return checkinput.par

	def readarrays(self):
		self.fullarray = np.load(self.par['path'] + '/dataarray.npy')
		self.alpha = np.load(self.par['path'] + '/alpha.npy')
		self.beta = np.load(self.par['path'] + '/beta.npy')
		self.omega = np.load(self.par['path'] + '/omega.npy')

	def reconstruct_mpi(self):
		ypix = grain_steps[2]

		# Chose part of data set for a specific CPU (rank).
		local_n = ypix / mpisize
		istart = rank * local_n
		istop = (rank + 1) * local_n

		local_grain_ang = self.reconstruct_part(
			grain_dim, grain_steps, data,
			slow, med, fast,
			theta, M, t_x, t_y, t_z,
			mode="horizontal", ista=istart, isto=istop)

		if rank == 0:
			# Make empty arrays to fill in data from other cores.
			recv_buffer = n.zeros(n.shape(local_grain_ang), dtype='float64')
			grain_ang = n.zeros(n.shape(local_grain_ang), dtype='float64')
			datarank = local_grain_ang[0, 0, 0, 0]
			local_grain_ang[0, 0, 0, 0] = n.mean(
				local_grain_ang[1:5, 1:5, istart:istop, 0])
			# print local_grain_ang[1:5, 1:5, istart:istop, 0]
			grain_ang[:, :, istart:istop, :] = local_grain_ang[:, :, istart:istop, :]
			for i in range(1, mpisize):
				try:
					comm.Recv(recv_buffer, MPI.ANY_SOURCE)
					datarank = int(recv_buffer[0, 0, 0, 0])
					rstart = datarank * local_n
					rstop = (datarank + 1) * local_n
					recv_buffer[0, 0, 0, 0] = n.mean(recv_buffer[1:5, 1:5, rstart:rstop, 0])
					grain_ang[:, :, rstart:rstop, :] =\
						recv_buffer[:, :, rstart:rstop, :]
				except Exception:
					print "MPI error."

		else:
			# all other process send their result
			comm.Send(local_grain_ang, dest=0)

		# root process prints results
		if self.rank == 0:
			return grain_ang

		# n.save('/u/data/andcj/tmp/grain_ang.npy', grain_ang)

	def reconstruct_part(self):
		"""
	Loop through virtual sample voxel-by-voxel and assign orientations based on
	forward projections onto read image stack. Done by finding the max intensity
	in a probability map prop[slow,med] summed over the fast coordinate.
	NB AS OF PRESENT THETA IS A NUMBER, NOT AN ARRAY. TO ALLOW FOR AN ARRAY
	NEED TO THINK ABOUT THE LOOPING AND THE DIMENSIONS OF prop.
	"""
	if self.rank == 0:
		t0 = time.clock()

	grain_xyz = n.zeros(grain_steps + [3])
	grain_ang = n.zeros(grain_steps + [2])
	grain_dimstep = n.array(grain_dim) / n.array(grain_steps)
	# grain_prop = n.zeros(grain_steps)

	detx_size = n.shape(data)[3]
	detz_size = n.shape(data)[4]
	detx_center = (detx_size - 0.) / 2  # should probably be -1 in stead of -0...
	detz_center = (detz_size - 0.) / 2.  # also here... but simulations used 0
	lens = len(slow)
	lenm = len(med)
	lenf = len(fast)
	mas = max(slow)
	mis = min(slow)
	mam = max(med)
	mim = min(med)
	cosinecurve = n.ones((lenf)) * 11.81
	cosineampl = n.ones((lenf))
	prop = n.zeros((lens, lenm, lenf))
	cospopt = [1., 1., 1.]

	t_x = "None"
	if self.rank == 0:
		print "Making forward projection..."
	# T_s2d = forward_projection.build_rotation_lookup(slow,med,fast,n.array([theta]),M,t_x,t_y,t_z,mode)
	T_s2d = forward_projection.build_rotation_lookup(n.array([0]), n.array([0]), fast, slow, M, t_x, t_y, t_z, mode)
	if self.rank == 0:
		print "Forward projection done."

	for iz in range(ista, isto):  # range(grain_steps[2]):

		# if self.rank == 0:
		done = 100 * (float(iz - ista) / (isto - ista))
		print "Calculation is %g perc. complete on core %g." % (done, self.rank)

		for ix in range(grain_steps[0]):
			timelist = []
			timedata = []

			for iy in range(grain_steps[1]):
				if self.rank == 0:
					t_0 = time.clock()
				grain_xyz[ix, iy, iz] = n.array(grain_pos) + grain_dimstep * (n.array([ix, iy, iz]) - 0.5 * (n.array(grain_steps) - 1))  # try n.meshgrid :)

				xyz_d_f = n.matmul(T_s2d[0, :, 0, 0], grain_xyz[ix, iy, iz])
				detx_f = n.rint(xyz_d_f[:, 0] + detx_center).astype(int)
				detz_f = n.rint(xyz_d_f[:, 2] + detz_center).astype(int)
				# projections outside detector frame hit the outmost row or column
				# should be OK assuming that the signal doesn't reach the very borders
				detx_f[detx_f < 0] = 0
				detx_f[detx_f >= detx_size] = detx_size - 1
				detz_f[detz_f < 0] = 0
				detz_f[detz_f >= detz_size] = detz_size - 1

				prop = data[:, :, range(lenf), detx_f, detz_f]

				cos = list(ndimage.measurements.center_of_mass(n.sum(prop, 2)))
				cos[0] = cos[0] * (mas - mis) / lens + mis
				cos[1] = cos[1] * (mam - mim) / lenm + mim

				grain_ang[ix, iy, iz, :] = cos

				if self.rank == 0:
					t_8 = time.clock()
					timelist.append(t_8 - t_0)
		if self.rank == 0:
			print "Avg. voxel time: {0:8.4f} seconds.".format(sum(timelist) / len(timelist))
			# print "Avg. data retrieval time: {0:8.4f} seconds.".format(sum(timedata)/len(timedata))
	if self.rank == 0:
		t1 = time.clock()
		print "time spent", t1 - t0
	grain_ang[0, 0, 0, 0] = rank
	return grain_ang  # grain_xyz,grain_ang,grain_prop

	def build_rotation_lookup(self):
		pass

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print "No .ini file specified."
	rec = main(sys.argv[1])
