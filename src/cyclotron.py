 ######################################################################################
 # This file is part of THECOS (https://github.com/arudol/thecos).
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


import numpy as np
from consts import * 
from scipy.integrate import simps

class Cyclotron(object):
	""" Cyclotron class to calculculate the terms entering the PDE deu to cyclotron emission/absorption
		of a thermal electron population. Only attribute is sim, an object of type SimulationManager.
		Everything else should be handled through interface functions, do not access attributes directly!
	"""

	def __init__(self, sim):
		""" Initialize the class with an instance of SimulationManager sim. 
		From this instance we draw (1) the grid parameters at the beginning and 
		(2) the current physical conditions at computation step."""
		self._sim = sim
		self._BIN_X = sim.grid_parameters['BIN_X']
		self._energygrid = sim.energygrid

		self.clear_arrays()

		# Check if the electron temperature, electron number density and magnetic field are provided by sim, 
		# Those are all neccessary for calulation of cyclotron emission. 
		if not 'T' in sim.source_parameters: 
			raise Exception('No electron temperature provided, necessary for cyclotron')
		if not 'n_e' in sim.source_parameters:
			raise Exception('No electron density provided, necessary for cyclotron')
		if not 'bprime' in sim.source_parameters:
			raise Exception('No magnetic field strength provided, necessary for cyclotron')

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

	## The next ones are properties such that they are automatically updated when ._source_parameters is updated
	@property
	def _theta(self):
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
		""" Get current source parameters from self._sim """
		self._source_parameters = getattr(self._sim, 'source_parameters')

	def get_current_photonarray(self):
		""" Get current photon array from self._sim """
		self._photonarray = getattr(self._sim, 'photonarray')

	def get_current_photonnumber(self):
		""" Get current photon N from self._sim """
		self._N = getattr(self._sim, 'N')		


	def calculate_and_pass_coefficents(self):
		"""
		Calculate and pass terms to the PDE:
		(1) get the current state from self._sim
		(2) calculate the escape and source terms
		(3) add them to the corresponding arrays of self._sim
		"""

		self.get_current_photonarray()
		self.get_current_photonnumber()
		self.get_source_parameters()
		self.calculate_terms()

		self._sim.add_to_escape_term(self._escapeterms)
		self._sim.add_to_source_term(self._sourceterms)


	def calculate_terms(self):
		"""Calculate the source and escape terms, store them in internal arrays"""
		self._escapeterms = np.array(list(map(self.alpha_cy_Vurm2011, self._energygrid)))
		self._sourceterms = np.array(list(map(self.j_cy_Vurm2011, self._energygrid)))


	def alpha_cy_Vurm2011(self, x):
		"""Calculate photon absorption as in Vurm 2011  A 19

		Arg:
			x (float): dimensionless photon energy
		
		Returns:
			float: alpha(x) absorption term
		"""
		a = 1.e10
		q = 4
		s = 10
		E = x * m_e*c0**2
		E_B = h * eStatC *self._bprime /(2 *np.pi * m_e *c0)
		prefactor = sigma_t *c0/alpha_f * (E_B /(m_e*c0**2))**(s-1)
		res = a * prefactor * self._theta**(q) * x **(-s) *self._n_e
		return res

	def j_cy_Vurm2011(self, x):
		"""Calculate photon emission from absorption + Kirchhoffs law

		Arg:
			x (float): dimensionless photon energy
		
		Returns:
			float: j(x) injection term
		"""

		E = x * m_e*c0**2
		alpha = self.alpha_cy_Vurm2011(x)

		lgr = x/self._theta
		#if np.abs(lgr) < 1.e-80: lgr = 1.e-80

		B_E = 2 * E**3 /(h*c0)**2 * (np.exp(lgr) -1 )**(-1)
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
	