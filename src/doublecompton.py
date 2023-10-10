import numpy as np
from consts import * 
from scipy.integrate import simps
 ######################################################################################
 # This file is part of THECOS (https://github.com/thecos).
 # Copyright (c) 2023 Annika Rudolph.
 # 
 # This program is free software: you can redistribute it and/or modify  
 # it under the terms of the GNU General Public License as published by  
 # the Free Software Foundation, version 3.
 # 
 # This program is distributed in the hope that it will be useful, but 
 # WITHOUT ANY WARRANTY; without even the implied warranty of 
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 # General Public License for more details.
 # You should have received a copy of the GNU General Public License 
 # along with this program. If not, see <http://www.gnu.org/licenses/>.
 ######################################################################################

class DoubleCompton(object):
	""" Double Compton class to calculculate the terms entering the PDE deu to cyclotron emission/absorption
		of a thermal electron population. Only attribute is sim, an object of type SimulationManager.
		Everything else should be handled through interface functions, do not access attributes directly!
	"""
	def __init__(self, sim):
		""" Initialize the class with an instance of SimulationManager sim. 
		From this instance we draw (1) the grid parameters at the beginning and 
		(2) the current physical conditions at computation step."""

		self.sim = sim
		self._BIN_X = sim.grid_parameters['BIN_X']
		self._energygrid = sim.energygrid

		self.clear_arrays()

		if not 'T' in sim.source_parameters: ## DC requires electron temperature 
			raise Exception('No electron temperature provided, necessary for double compton')
		if not 'n_e' in sim.source_parameters: ## DC requires electron number density
			raise Exception('No electron density provided, necessary for double compton')
		self._source_parameters = sim.source_parameters


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

	def get_current_photonarray(self):
		""" Get current photon array from self._sim """
		self._photonarray = getattr(self.sim, 'photonarray')

	def get_current_photonnumber(self):
		""" Get current photon N from self._sim """
		self._N = getattr(self.sim, 'N')		

	def get_source_parameters(self):
		""" Get current source parameters from self._sim """
		self._source_parameters = getattr(self.sim, 'source_parameters')

	def calculate_and_pass_coefficents(self):
		"""
		Calculate and pass terms to the PDE:
		(1) get the current state from self._sim
		(2) calculate the escape and source terms
		(3) add them to the corresponding arrays of self._sim
		"""	
		self.get_source_parameters()
		self.get_current_photonarray()
		self.get_current_photonnumber()
		self.calculate_terms()

		self.sim.add_to_escape_term(self._escapeterms)
		self.sim.add_to_source_term(self._sourceterms)


	def calculate_terms(self):
		"""Calculate the source and escape terms, store them in internal arrays"""

		self._escapeterms = np.array(list(map(self.alpha_dc_Vurm2011, self._energygrid)))
		self._sourceterms = np.array(list(map(self.j_dc_Vurm2011, self._energygrid)))

	def gaunt_theta(self, theta):
		""" Helper function to calculate gaunt (theta)
		Arg:
			theta (float)
		Returns:
			float
		"""
		res = (1+ 13.91*theta + 11.05 * theta**2 + 19.92 * theta**3 )**(-1.)
		return res

	def alpha_dc_Vurm2011_planck(self, x):
		"""Calculate photon absorption as in Vurm 2011  A 16, for a planck distribution

		Arg:
			x (float): dimensionless photon energy
		
		Returns:
			float: alpha(x) absorption term
		"""		
		E = x * m_e*c0**2
		prefactor = 38.4 * alpha_f / np.pi 
		res = prefactor *(self._T * k_B_erg/E)**(2) * self._theta**2 * self.gaunt_theta(self._theta) * sigma_t * c0 * self._n_e
		return res

	def alpha_dc_Vurm2011(self, x):
		"""Calculate photon absorption as in Vurm 2011  A 14

		Arg:
			x (float): dimensionless photon energy
		
		Returns:
			float: alpha(x) absorption term
		"""
		E = x * m_e*c0**2
		prefactor = 2 * alpha_f / np.pi**2 * lambda_C**3 *sigma_t *c0
		res = prefactor *x**(-2) * self._theta * self.gaunt_theta(self._theta) * self._n_e * self._N
		return res

	def j_dc_Vurm2011(self, x):
		"""Calculate photon emission from absorption + Kirchhoffs law

		Arg:
			x (float): dimensionless photon energy
		
		Returns:
			float: j(x) injection term
		"""

		E = x * m_e*c0**2
		alpha = self.alpha_dc_Vurm2011(x)

		lgr = x/self._theta
		#if np.abs(lgr) < 1.e-80: lgr = 1.e-80
		B_E =  (np.exp(lgr) -1 )**(-1)
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
	