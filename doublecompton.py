import numpy as np
from consts import * 
from scipy.integrate import simps

class DoubleCompton(object):

	def __init__(self, sim):
		## Initialise , taking an instance of SimulationManager such that energy grids etc are the same
		self.sim = sim
		self._BIN_X = sim.grid_parameters['BIN_X']
		self._X_I = sim.grid_parameters['X_I']
		self._D_X = sim.grid_parameters['D_X']
		self._aterms = np.zeros(self._BIN_X-1)
		self._sourceterms = np.zeros(self._BIN_X)
		self._escapeterms = np.zeros(self._BIN_X)


		if not 'T' in sim.source_parameters:
			raise Exception('No electron temperature provided, necessary for double compton')
		if not 'n_e' in sim.source_parameters:
			raise Exception('No electron density provided, necessary for double compton')

		self._source_parameters = sim.source_parameters

		self._energygrid = sim.energygrid

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
	def _n_e(self):
		"""
		electron number density
		"""
		return self._source_parameters["n_e"]

	def get_current_photonarray(self):
		## Get current photon array from simulation class ## 
		self._photonarray = getattr(self.sim, 'photonarray')

	def get_current_photonnumber(self):
		## Get current photon N from simulation class ## 
		self._N = getattr(self.sim, 'N')		

	def get_source_parameters(self):
		""" Get current source parameters """
		self._source_parameters = getattr(self.sim, 'source_parameters')

	def calculate_and_pass_coefficents(self):
	## Calculate all escape and sink terms ##
		self.get_source_parameters()
		self.get_current_photonarray()
		self.get_current_photonnumber()
		self.calculate_terms()

		self.sim.add_to_escape_term(self._escapeterms)
		self.sim.add_to_source_term(self._sourceterms)


	def calculate_terms(self):
		#for k in range(self._BIN_X):
		#	x = self._energygrid[k]
		#	self._escapeterms[k] = self.alpha_dc_Vurm2011(x)
		#	self._sourceterms[k] = self.j_dc_Vurm2011(x)

		self._escapeterms = np.array(list(map(self.alpha_dc_Vurm2011, self._energygrid)))
		self._sourceterms = np.array(list(map(self.j_dc_Vurm2011, self._energygrid)))

	def gaunt_theta(self, theta):
		res = (1+ 13.91*theta + 11.05 * theta**2 + 19.92 * theta**3 )**(-1.)
		return res

	def alpha_dc_Vurm2011_planck(self, x):
	## photon absorption as in Vurm 2011 A 16##
		E = x * m_e*c0**2
		prefactor = 38.4 * alpha_f / np.pi 
		res = prefactor *(self._T * k_B_erg/E)**(2) * self._Theta**2 * self.gaunt_theta(self._Theta) * sigma_t * c0 * self._n_e
		return res

	def alpha_dc_Vurm2011(self, x):
	## photon absorption as in Vurm 2011  A 14 ##
		E = x * m_e*c0**2
		prefactor = 2 * alpha_f / np.pi**2 * lambda_C**3 *sigma_t *c0
		res = prefactor *x**(-2) * self._Theta * self.gaunt_theta(self._Theta) * self._n_e * self._N
		return res

	def j_dc_Vurm2011(self, x):
	## photon emission from alpha as in Vurm 2011 A 14 and using Kirchhoffs law to link alpha and j ##
		E = x * m_e*c0**2
		alpha = self.alpha_dc_Vurm2011(x)

		lgr = x/self._Theta
		#if np.abs(lgr) < 1.e-80: lgr = 1.e-80

	#	B_E = 2 * E**3 /(h*c0)**2 * (np.exp(lgr) -1 )**(-1) # this is not in photon momentum space!
		B_E =  (np.exp(lgr) -1 )**(-1)
		res = alpha * B_E
		return res

	def get_injectionrate(self):
		return self._sourceterms

	def get_coolingrate(self):
		return self._escapeterms
	