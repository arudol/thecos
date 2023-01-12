import numpy as np
from consts import * 
from scipy.integrate import simps

class Cyclotron(object):

	def __init__(self, sim):
		## Initialise , taking an instance of SimulationManager such that energy grids etc are the same
		self.sim = sim
		self._aterms = np.zeros(sim.BIN_X)
		self._sourceterms = np.zeros(sim.BIN_X)
		self._escapeterms = np.zeros(sim.BIN_X)

		self._BIN_X = sim.BIN_X
		self._X_I = sim.X_I
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
		self._Theta = getattr(self.sim, 'T')
		self._T = self._Theta * m_e * c0**2 / k_B_erg

	def get_density(self):
		## Get matter density from simulation class ## 
		self._n_e = getattr(self.sim, 'n_e')

	def get_bprime(self):
		## Get matter density from simulation class ## 
		self._bprime = getattr(self.sim, 'bprime')

	def get_current_photonarray(self):
		## Get current photon array from simulation class ## 
		self._photonarray = getattr(self.sim, 'photonarray')

	def calculate_n_photons(self):
		array_to_integrate = self._energygrid**2 * self._photonarray
		n_integral = simps(array_to_integrate, self._energygrid)
		res = 8 * np.pi * (m_e* c0**2)**3 /(h *c0)**3 * n_integral
		self._N = res

	def get_current_photonnumber(self):
		## Get current photon N from simulation class ## 
		self._N = getattr(self.sim, 'N')		


	def calculate_and_pass_coefficents(self):
	## Calculate all escape and sink terms ##
		self.get_temperature()
		self.get_density()
		self.get_current_photonarray()
		self.get_current_photonnumber()
		self.get_bprime()
		self.calculate_terms()

		self.sim.add_to_escapeterms(self._escapeterms)
		self.sim.add_to_sourceterms(self._sourceterms)


	def calculate_terms(self):
		for k in range(self._BIN_X):
			x = self._energygrid[k]
			self._escapeterms[k] = self.alpha_cy_Vurm2011(x)
			self._sourceterms[k] = self.j_cy_Vurm2011(x)


	def alpha_cy_Vurm2011(self, x):
	## photon absorption as in Vurm 2011  A 19 ##
		a = 1.e10
		q = 4
		s = 10
		E = x * m_e*c0**2
		E_B = h * eStatC *self._bprime /(2 *np.pi * m_e *c0)
		prefactor = sigma_t *c0/alpha_f * (E_B /(m_e*c0**2))**(s-1)
		res = a * prefactor * self._Theta**(q) * x **(-s) *self._n_e
		return res

	def j_cy_Vurm2011(self, x):
	## photon emission from Vurm 2011 alpha and black body assumption ##
		E = x * m_e*c0**2
		alpha = self.alpha_cy_Vurm2011(x)

		lgr = x/self._Theta
		#if np.abs(lgr) < 1.e-80: lgr = 1.e-80

		B_E = 2 * E**3 /(h*c0)**2 * (np.exp(lgr) -1 )**(-1)
		B_E =  (np.exp(lgr) -1 )**(-1)
		res = alpha * B_E
		return res

	def get_injectionrate(self):
		return self._sourceterms

	def get_coolingrate(self):
		return self._escapeterms
	