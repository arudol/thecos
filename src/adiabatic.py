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

class Adiabatic(object):

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

		if not 'lorentz' in sim.source_parameters:
			raise Exception('No Lorentz factor provided, necessary for adiabatic cooling')
		if not 'radius' in sim.source_parameters:
			raise Exception('No radius provided, necessary for adiabatic cooling')
		if not 'pl_decay' in sim.source_parameters:
			raise Exception('No scaling of volume provided, necessary for adiabatic cooling')

		self._source_parameters = sim.source_parameters

	def clear_arrays(self):
		""" Clear all internal arrays"""
		self._aterms = np.zeros(self._BIN_X-1)
		self._sourceterms = np.zeros(self._BIN_X)
		self._escapeterms = np.zeros(self._BIN_X)


	def initialise_kernels(self):
		pass

	@property
	def _radius(self):
		""" current radius"""
		return self._source_parameters['radius']

	@property
	def _lorentz(self):
		"""Lorentz factor of the plasma """
		return self._source_parameters['lorentz']

	@property
	def _PL(self):
		"""
		power-law index for the density decay
		"""
		return self._source_parameters["pl_decay"]

	def get_source_parameters(self):
		""" Get current source parameters """
		self._source_parameters = getattr(self.sim, 'source_parameters')

	
	def get_halfgrid(self):
		"""
		Get the half grid of the energies
		"""
		self._halfgrid = getattr(self.sim, 'half_grid')


	def calculate_and_pass_coefficents(self):
		"""
		Fetch current simulation parameters from the parent simulation manager,
		fill the internal cooling/escape arrays and pass them to the parent simulation manager.
		"""
		self.get_source_parameters()
		self.get_halfgrid()

		self.calculate_terms()

		self.sim.add_to_escape_term(self._escapeterms)
		self.sim.add_to_heating_term(-self._aterms_momentumspace)
		#self.sim.add_to_heating_term(self._aterms)


	def calculate_terms(self):
		"""
		Fill the arrays with the escape and cooling terms
		"""
		self._escapeterms.fill(self.adiabatic_escape())
		self._aterms_momentumspace = np.array(list(map(self.adiabatic_cooling_momentumspace, self._halfgrid)))

		#for k, x in enumerate(self._halfgrid):
	#		self._aterms[k] = self.adiabatic_cooling(x)

	def t_ad(self):
		"""
		Calculate the standard adiabatic timescale of a plasma expanding with velocity :math:'\beta c', at radius :math:'r'

		;returns: t_ad , the adiabatic cooling timescale [s]
		"""
		t_ad = self._radius/ self._lorentz / (c0 * beta(self._lorentz))/ self._PL
		return t_ad

	def adiabatic_escape(self):
		""" 
		Timescale to mimic plasma dilution for a plasma expanding with velocity :math:'\beta c', at radius :math:'r' and with 
		Volume evolving as :math:'V \propto r^{s_n}'. THe correpsonding fictional escape timescale is defined as

		..math::
			\tau_\mathrm{AD}^{-1} = \frac{s_n \beta c}{r \Gamma}

		:returns: Escape timescale :math:'\tau_\mathrm{AD}^{-1}'' [1/s]
		"""
		return 1/self.t_ad()


	def adiabatic_cooling_momentumspace(self, x):
		""" 
		Adiabatic cooling timescale for a plasma expanding with velocity :math:'\beta c', at radius :math:'r' and with 
		Volume evolving as :math:'V \propto r^{s_n}'. 
		The cooling timescale in photon momentum space (which requires extra factor of :math:'8 \pi') is then 

		..math::
			- a_\mathrm{cool} = - \frac{s_n \beta c }{3 \cdot 8 \pi \Gamma r } x 

		:param: x ,photon dimensionless energy
		:returns: cooling timescale :math:'a_\mathrm{cool}' [1/s]
		"""
		return  x**3/self.t_ad()/3


	def get_injectionrate(self):
		"""
		Get the array holding source terms to the PDE.
		"""
		return self._sourceterms

	def get_coolingrate(self):
		"""
		Get the array holding escape terms to the PDE.
		"""
		return self._escapeterms

	def get_aterm(self):
		"""
		Get the array holding cooling terms to the PDE.
		"""
		return self._aterms
	