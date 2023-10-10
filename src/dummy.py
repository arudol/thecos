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


import numpy as np
from consts import * 

class Dummy(object):

	def __init__(self, sim):
		## Initialise , taking an instance of SimulationManager such that energy grids etc are the same
		self.sim = sim
		self.clear_internal_arrays()

		self._BIN_X = sim.BIN_X
		self._energygrid = sim.energygrid

	def clear_internal_arrays(self):
		""" Set all internal arrays to zero 

		THIS IS A PRE-DEFINED HOOK FOR THE SIMULATION MANAGER
		"""

		self._aterms = np.zeros(self._BIN_X-1)
		self._sourceterms = np.zeros(self._BIN_X)
		self._escapeterms = np.zeros(self._BIN_X)


	def initialise_kernels(self):
		""" 
		Pre-calculate kernals at beginning of a calculation.
		Note: nothing done here, fill in something if needed

		THIS IS A PRE-DEFINED HOOK FOR THE SIMULATION MANAGER
		"""

		pass

	## properties such that it is automatically updated when ._source_parameters is updated
	@property
	def _theta(self):
		"""dimensionless electron temperature """
		return self._source_parameters['T']

	def get_source_parameters(self):
		""" Get current source parameters from self._sim """
		self._source_parameters = getattr(self._sim, 'source_parameters')

	def get_current_photonarray(self):
		""" Get current photon array from self._sim """
		self._photonarray = getattr(self._sim, 'photonarray')


	def calculate_and_pass_coefficents(self):
		"""
		Calculate and pass terms to the PDE:
		(1) get the current state from self._sim
		(2) calculate the escape and source terms
		(3) add them to the corresponding arrays of self._sim

		THIS IS A PRE-DEFINED HOOK FOR THE SIMULATION MANAGER
		"""
		self.get_current_photonarray()
		self.get_source_parameters()
		self.calculate_terms()

		self.sim.add_to_escapeterms(self._escapeterms)
		self.sim.add_to_sourceterms(self._sourceterms)

	def calculate_terms(self):
		"""Calculate the source and escape terms, store them in internal arrays.
		
		Note: Set to zero here, fill differently if needed
		"""
		self._escapeterms = np.zeros(self._BIN_X)
		self._sourceterms = np.zeros(self._BIN_X)

	def get_injectionrate(self):
		"""
		Get the source terms of this module. 

		THIS IS A PRE-DEFINED HOOK FOR THE SIMULATION MANAGER
		"""
		return self._sourceterms

	def get_coolingrate(self):
		"""
		Get the cooling time of this module. 

		THIS IS A PRE-DEFINED HOOK FOR THE SIMULATION MANAGER
		"""
		return self._escapeterms


	