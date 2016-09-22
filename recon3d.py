from check_input import read as ini
import sys
import time
import numpy as np
from scipy import ndimage

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
		grain_ang = self.reconstruct_mpi()

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
		# self.theta = np.load(self.par['path'] + '/theta.npy')

	def reconstruct_mpi(self):
		ypix = np.array(self.par['grain_steps'])[2]

		# Chose part of data set for a specific CPU (rank).
		local_n = ypix / self.size
		istart = self.rank * local_n
		istop = (self.rank + 1) * local_n

		local_grain_ang = self.reconstruct_part(ista=istart, isto=istop)

		if self.rank == 0:
			# Make empty arrays to fill in data from other cores.
			recv_buffer = np.zeros(np.shape(local_grain_ang), dtype='float64')
			grain_ang = np.zeros(np.shape(local_grain_ang), dtype='float64')
			datarank = local_grain_ang[0, 0, 0, 0]
			local_grain_ang[0, 0, 0, 0] = np.mean(
				local_grain_ang[1:5, 1:5, istart:istop, 0])
			# print local_grain_ang[1:5, 1:5, istart:istop, 0]
			grain_ang[:, :, istart:istop, :] = local_grain_ang[:, :, istart:istop, :]
			for i in range(1, self.size):
				try:
					comm.Recv(recv_buffer, MPI.ANY_SOURCE)
					datarank = int(recv_buffer[0, 0, 0, 0])
					rstart = datarank * local_n
					rstop = (datarank + 1) * local_n
					recv_buffer[0, 0, 0, 0] = np.mean(recv_buffer[1:5, 1:5, rstart:rstop, 0])
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

		# np.save('/u/data/andcj/tmp/grain_ang.npy', grain_ang)

	def reconstruct_part(self, ista, isto):
		"""
		Loop through virtual sample voxel-by-voxel and assign orientations based on
		forward projections onto read image stack. Done by finding the max intensity
		in a probability map prop[slow,med] summed over the fast coordinate.
		NB AS OF PRESENT THETA IS A NUMBER, NOT AN ARRAY. TO ALLOW FOR AN ARRAY
		NEED TO THINK ABOUT THE LOOPING AND THE DIMENSIONS OF prop.
		"""
		if self.rank == 0:
			t0 = time.clock()

		slow = self.alpha
		med = self.beta
		fast = self.omega

		grain_steps = self.par['grain_steps']
		grain_dim = np.array(self.par['grain_dim'])
		grain_pos = np.array(self.par['grain_pos'])

		grain_xyz = np.zeros(grain_steps + [3])
		grain_ang = np.zeros(grain_steps + [2])
		grain_dimstep = np.array(grain_dim) / np.array(grain_steps)
		# grain_prop = np.zeros(grain_steps)

		detx_size = np.shape(self.fullarray)[3]
		detz_size = np.shape(self.fullarray)[4]
		detx_center = (detx_size - 0.) / 2  # should probably be -1 in stead of -0...
		detz_center = (detz_size - 0.) / 2.  # also here... but simulations used 0
		lens = len(slow)
		lenm = len(med)
		lenf = len(fast)
		mas = max(slow)
		mis = min(slow)
		mam = max(med)
		mim = min(med)
		prop = np.zeros((lens, lenm, lenf))

		t_x = "None"
		if self.rank == 0:
			print "Making forward projection..."
		# T_s2d = forward_projection.build_rotation_lookup
		# (slow,med,fast,np.array([theta]),M,t_x,t_y,t_z,mode)
		# T_s2d = forward_projection.build_rotation_lookup
		# (np.array([0]), n.array([0]), fast, slow, M, t_x, t_y, t_z, mode)
		T_s2d = self.build_rotation_lookup()
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

					grain_xyz[ix, iy, iz] =\
						grain_pos + grain_dimstep *\
						(np.array([ix, iy, iz]) - 0.5 *
							(np.array(grain_steps) - 1))

					xyz_d_f = np.matmul(T_s2d[0, :, 0, 0], grain_xyz[ix, iy, iz])
					detx_f = np.rint(xyz_d_f[:, 0] + detx_center).astype(int)
					detz_f = np.rint(xyz_d_f[:, 2] + detz_center).astype(int)
					# projections outside detector frame hit the outmost row or column
					# should be OK assuming that the signal doesn't reach the very borders
					detx_f[detx_f < 0] = 0
					detx_f[detx_f >= detx_size] = detx_size - 1
					detz_f[detz_f < 0] = 0
					detz_f[detz_f >= detz_size] = detz_size - 1

					prop = self.fullarray[:, :, range(lenf), detx_f, detz_f]

					cos = list(ndimage.measurements.center_of_mass(np.sum(prop, 2)))
					cos[0] = cos[0] * (mas - mis) / lens + mis
					cos[1] = cos[1] * (mam - mim) / lenm + mim

					grain_ang[ix, iy, iz, :] = cos

					if self.rank == 0:
						t_8 = time.clock()
						timelist.append(t_8 - t_0)
			if self.rank == 0:
				print "Avg. voxel time: {0:8.4f} seconds.".format(
					sum(timelist) / len(timelist))
				# print "Avg. data retrieval time: {0:8.4f} seconds.".format(
				# 	sum(timedata)/len(timedata))
		if self.rank == 0:
			t1 = time.clock()
			print "time spent", t1 - t0
		grain_ang[0, 0, 0, 0] = self.rank
		return grain_ang  # grain_xyz,grain_ang,grain_prop

	def build_rotation_lookup(self):
		# def build_rotation_lookup(phi_up,phi_lo,omega,\
		# theta,M,t_x="None",t_y="None",t_z="None",mode="horizontal"):
		"""
		Set up the rotation_lookup[theta,omega,phi_lo,phi_up] lookup table
		of rotation matrices for each value in the theta, omega, phi_lo and
		phi_up arrays.

		NB! NEED TO THINK ABOUT THE IMPLICATIONS OF
		theta BEING MORE THAN A SINGLE VALUE!!!
		"""

		# up = np.pi * phi_up / 180.
		# lo = np.pi * phi_lo / 180.
		# om = np.pi * omega / 180.
		# th = np.pi * theta / 180.

		up = np.pi * self.alpha / 180.
		lo = np.pi * self.beta / 180.
		om = np.pi * self.omega / 180.
		# th = np.pi * self.par['theta'] / 180.
		th = np.pi * np.array([0]) / 180.

		try:
			t_xx = np.pi * self.par['t_x'] / 180.
			t_yy = np.pi * self.par['t_y'] / 180.
			t_zz = np.pi * self.par['t_z'] / 180.
		except:
			print "No detector tilt"
			self.par['t_x'] = "None"
			self.par['t_z'] = "None"

		th_mat, om_mat, lo_mat, up_mat = np.meshgrid(th, om, lo, up, indexing='ij')

		T_up = np.zeros((len(th), len(om), len(lo), len(up), 3, 3))
		T_lo = np.zeros((len(th), len(om), len(lo), len(up), 3, 3))
		Omega = np.zeros((len(th), len(om), len(lo), len(up), 3, 3))
		Theta = np.zeros((len(th), len(om), len(lo), len(up), 3, 3))
		T_det = np.zeros((len(th), len(om), len(lo), len(up), 3, 3))

		# For now the detector tilt is the unit matrix, i.e. an ideal detector
		# positioned perpendicular to the diffracted beam (t_x=t=y=t_z=None).
		# This can be changed by supplying tilts t_x (vertical) or t_z (horizontal).
		T_det[:, :, :, :, 0, 0] = -1.
		# T_det[:,:,:,:,1,1] = 1. #leaving T_det[:,:,:,:,1,1]=0 gives the
		# projection onto the detector plane
		T_det[:, :, :, :, 2, 2] = -1.

		if self.par['mode'] == "horizontal":
			Theta[:, :, :, :, 0, 0] = np.cos(th_mat)
			Theta[:, :, :, :, 0, 1] = -np.sin(th_mat)
			Theta[:, :, :, :, 1, 0] = np.sin(th_mat)
			Theta[:, :, :, :, 1, 1] = np.cos(th_mat)
			Theta[:, :, :, :, 2, 2] = 1.
			Omega[:, :, :, :, 0, 0] = 1.
			Omega[:, :, :, :, 1, 1] = np.cos(om_mat)
			Omega[:, :, :, :, 1, 2] = -np.sin(om_mat)
			Omega[:, :, :, :, 2, 1] = np.sin(om_mat)
			Omega[:, :, :, :, 2, 2] = np.cos(om_mat)
			T_lo[:, :, :, :, 0, 0] = np.cos(lo_mat)
			T_lo[:, :, :, :, 0, 2] = np.sin(lo_mat)
			T_lo[:, :, :, :, 1, 1] = 1.
			T_lo[:, :, :, :, 2, 0] = -np.sin(lo_mat)
			T_lo[:, :, :, :, 2, 2] = np.cos(lo_mat)
			T_up[:, :, :, :, 0, 0] = np.cos(up_mat)
			T_up[:, :, :, :, 0, 1] = -np.sin(up_mat)
			T_up[:, :, :, :, 1, 0] = np.sin(up_mat)
			T_up[:, :, :, :, 1, 1] = np.cos(up_mat)
			T_up[:, :, :, :, 2, 2] = 1.
			if self.par['t_z'] != "None":
				T_det[:, :, :, :, 0, 0] = -1. / np.cos(t_zz - 2 * np.mean(th))
		elif self.par['mode'] == "vertical":
			Theta[:, :, :, :, 0, 0] = 1.
			Theta[:, :, :, :, 1, 1] = np.cos(th_mat)
			Theta[:, :, :, :, 1, 2] = -np.sin(th_mat)
			Theta[:, :, :, :, 2, 1] = np.sin(th_mat)
			Theta[:, :, :, :, 2, 2] = np.cos(th_mat)
			Omega[:, :, :, :, 0, 0] = np.cos(om_mat)
			Omega[:, :, :, :, 0, 1] = -np.sin(om_mat)
			Omega[:, :, :, :, 1, 0] = np.sin(om_mat)
			Omega[:, :, :, :, 1, 1] = np.cos(om_mat)
			Omega[:, :, :, :, 2, 2] = 1.
			# NB Should define around which axes the upper and lower rotation belong
			T_lo[:, :, :, :, 0, 0] = 1.
			T_lo[:, :, :, :, 1, 1] = 1.
			T_lo[:, :, :, :, 2, 2] = 1.
			T_up[:, :, :, :, 0, 0] = 1.
			T_up[:, :, :, :, 1, 1] = 1.
			T_up[:, :, :, :, 2, 2] = 1.
			if self.par['t_x'] != "None":
				T_det[:, :, :, :, 2, 2] = -1. / np.cos(t_xx - 2 * np.mean(th))
		else:
			print "ERROR: scattering geometry not defined"

		T_s2d = float(self.par['M']) * np.matmul(
			T_det, np.matmul(Theta, np.matmul(Omega, np.matmul(T_lo, T_up))))
		return T_s2d

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print "No .ini file specified."
	rec = main(sys.argv[1])
