import numpy as np
from consts import * 

class Dummy(object):

	def __init__(self, sim):
		## Initialise , taking an instance of SimulationManager such that energy grids etc are the same
		self.sim = sim
		self._aterms = np.zeros(sim.BIN_X)
		self._sourceterms = np.zeros(sim.BIN_X)
		self._escapeterms = np.zeros(sim.BIN_X)

		self._BIN_X = sim.BIN_X
		self._D_X = sim.D_X
		self._energygrid = sim.energygrid

	def clear_internal_arrays(self):
		self._aterms = np.zeros(self._BIN_X)
		self._sourceterms = np.zeros(self._BIN_X)
		self._escapeterms = np.zeros(self._BIN_X)


	def initialise_kernels(self):
		pass

	def get_temperature(self):
		## Get electron temperature from simulation class ## 
		self._T = getattr(self.sim, 'T')

	def get_density(self):
		## Get matter density from simulation class ## 
		self._rho = getattr(self.sim, 'rho')

	def get_current_photonarray(self):
		## Get current photon array from simulation class ## 
		self._photonarray = getattr(self.sim, 'photonarray')


	def calculate_and_pass_coefficents(self):
	## Calculate all escape and sink terms ##
		self.get_temperature()
		self.get_density()
		self.get_current_photonarray()

		#for i in range(len(self._BIN_X)):
		#	x = self._energygrid[i]
		#	self._escapeterms[i] = self.alpha_freefree_Vurm2011(x)
	#		self._sourceterms[i] = self.j_freefree_Vurm2011(x)

		self.sim.add_to_escapeterms(self._escapeterms)
		self.sim.add_to_sourceterms(self._sourceterms)

	def get_injectionrate(self):
		return self._sourceterms

	def get_coolingrate(self):
		return self._escapeterms


	