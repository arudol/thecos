import numpy as np
from consts import * 
from scipy.integrate import simps

class Cyclotron(object):

	def __init__(self, sim):
		## Initialise , taking an instance of SimulationManager such that energy grids etc are the same
		self.sim = sim
		self._BIN_X = sim.grid_parameters['BIN_X']
		self._X_I = sim.grid_parameters['X_I']
		self._D_X = sim.grid_parameters['D_X']
		self._aterms = np.zeros(self._BIN_X-1)
		self._sourceterms = np.zeros(self._BIN_X)
		self._escapeterms = np.zeros(self._BIN_X)

		self._energygrid = sim.energygrid

		if not 'T' in sim.source_parameters:
			raise Exception('No electron temperature provided, necessary for cyclotron')
		if not 'n_e' in sim.source_parameters:
			raise Exception('No electron density provided, necessary for cyclotron')
		if not 'bprime' in sim.source_parameters:
			raise Exception('No magnetic field strength provided, necessary for cyclotron')

		self._source_parameters = sim.source_parameters

	def clear_arrays(self):
		self._aterms = np.zeros(self._BIN_X)
		self._sourceterms = np.zeros(self._BIN_X)
		self._escapeterms = np.zeros(self._BIN_X)


	def initialise_kernels(self):
		pass

	@property
	def _Theta(self):
		"""dimensionless electron temperature """
		return self._source_parameters['T']

	@property
	def _T(self):
		""" electron temperature in Kelvin """
		return self._source_parameters['T'] * m_e * c0**2 / k_B_erg

	@property
	def _bprime(self):
		""" Comoving magnetic field """
		return self._source_parameters['bprime']

	@property
	def _n_e(self):
		"""
		electron number density
		"""
		return self._source_parameters["n_e"]


	def get_source_parameters(self):
		""" Get current source parameters """
		self._source_parameters = getattr(self.sim, 'source_parameters')

	def get_current_photonarray(self):
		## Get current photon array from simulation class ## 
		self._photonarray = getattr(self.sim, 'photonarray')

	def get_current_photonnumber(self):
		## Get current photon N from simulation class ## 
		self._N = getattr(self.sim, 'N')		


	def calculate_and_pass_coefficents(self):
	## Calculate all escape and sink terms ##

		self.get_current_photonarray()
		self.get_current_photonnumber()
		self.get_source_parameters()
		self.calculate_terms()

		self.sim.add_to_escape_term(self._escapeterms)
		self.sim.add_to_source_term(self._sourceterms)


	def calculate_terms(self):
		self._escapeterms = np.array(list(map(self.alpha_cy_Vurm2011, self._energygrid)))
		self._sourceterms = np.array(list(map(self.j_cy_Vurm2011, self._energygrid)))


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
	