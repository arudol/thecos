import numpy as np
from consts import * 

class Bremsstrahlung(object):
	""" Bremsstrahlung class to calculculate the terms entering the PDE deu to Free-Free emission/absorption
		of a thermal electron population
	"""
	def __init__(self, sim):
		""" Initialize the class with an instance of SimulationManager sim. 
		From this instance we draw (1) the grid parameters at the beginning and 
		(2) the current physical conditions at computation step."""

		self.sim = sim
		self._BIN_X = sim.grid_parameters['BIN_X']
		self._X_I = sim.grid_parameters['X_I']
		self._D_X = sim.grid_parameters['D_X']

		self.clear_arrays()

		if not 'T' in sim.source_parameters:
			raise Exception('No electron temperature provided, necessary for double compton')
		if not 'n_e' in sim.source_parameters:
			raise Exception('No electron density provided, necessary for double compton')

		self._source_parameters = sim.source_parameters

		self._energygrid = sim.energygrid


	def clear_arrays(self):
		""" Set all internal arrays to zero """

		self._aterms = np.zeros(self._BIN_X-1)
		self._sourceterms = np.zeros(self._BIN_X)
		self._escapeterms = np.zeros(self._BIN_X)

	def initialise_kernels(self):
		""" 
		Pre-calculate kernals at beginning of a calculation.
		Note: nothing done here, not neccessary for this process. 
		"""
		pass

	@property
	def _theta(self):
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

	def get_source_parameters(self):
		""" Get current source parameters from self._sim """
		self._source_parameters = getattr(self.sim, 'source_parameters')

	def get_current_photonarray(self):
		""" Get current photon array from self._sim """
		self._photonarray = getattr(self.sim, 'photonarray')


	def get_current_photonnumber(self):
		""" Get current photon N from self._sim """
		self._N = getattr(self.sim, 'N')


	def calculate_and_pass_coefficents(self):
		"""
		Calculate and pass terms to the PDE:
		(1) get the current state from self._sim
		(2) calculate the escape and source terms
		(3) add them to the corresponding arrays of self._sim
		"""
		self.get_source_parameters()
		self.get_current_photonarray()
		self.calculate_terms()

		self.sim.add_to_escape_term(self._escapeterms)
		self.sim.add_to_source_term(self._sourceterms)

	def calculate_terms(self):
		"""Calculate the source and escape terms, store them in internal arrays"""

		self._escapeterms = np.array(list(map(self.alpha_freefree_Vurm2011, self._energygrid)))
		self._sourceterms = np.array(list(map(self.j_freefree_Vurm2011, self._energygrid)))


	def gff(self, x, theta):
		"""
		Gaunt factor for Free-Free absorption.

		Args:
			x (float): dimensionless photon energy
			theta (float): dimensionless electron temperature

		Returns:
			float: gaunt factor
		"""
		res = np.sqrt(3)/np.pi* np.log(2.25 * theta/x)
		return res

	def alpha_freefree_Vurm2011(self, x):
		"""Calculate photon absorption as in Vurm 2011 

		Arg:
			x (float): dimensionless photon energy
		
		Returns:
			float: alpha(x) absorption term
		"""		
		E = x * m_e*c0**2
		lgr = E/(k_B_erg*self._T)

		#if lgr > 1.e10: lgr = 1.e10
		expo_factor = (1- np.exp(-lgr))

		prefactor = alpha_f  * lambda_C**3 *sigma_t * c0/(np.sqrt(3 *8* np.pi**3))
		res = prefactor *self._theta**(-1/2.)*(self._n_e)**2*x**(-3) *self.gff(x, self._theta) * expo_factor
		if res < 0.0: res = 0.0
		return res

	def j_freefree_Vurm2011(self, x):
		"""Calculate photon emission from absorption + Kirchhoffs law

		Arg:
			x (float): dimensionless photon energy
		
		Returns:
			float: j(x) injection term
		"""

		E = x * m_e*c0**2
		alpha = self.alpha_freefree_Vurm2011(x)
		lgr = x/self._theta
		B_E = (np.exp(lgr) -1 )**(-1)
		res = alpha * B_E

		return res

	def get_injectionrate(self):
		""" Get internally stored injection terms
		
		Returns:
			array of floats: injection terms due to photon emission"""
		return self._sourceterms

	def get_coolingrate(self):
		""" Get internally stored cooling terms
		
		Returns:
			array of floats: escape/sink terms due to absorption"""
		return self._escapeterms
	