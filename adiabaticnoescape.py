import numpy as np
from consts import * 

class AdiabaticNoEscape(object):

	def __init__(self, sim):
		## Initialise , taking an instance of SimulationManager such that energy grids etc are the same
		self.sim = sim
		self._aterms = np.zeros(sim.BIN_X-1)
		self._sourceterms = np.zeros(sim.BIN_X)
		self._escapeterms = np.zeros(sim.BIN_X)

		self._BIN_X = sim.BIN_X
		self._X_I = sim.X_I
		self._D_X = sim.D_X
		self._energygrid = sim.energygrid

	def clear_arrays(self):
		""" Clear all internal arrays"""
		self._aterms = np.zeros(self._BIN_X-1)
		self._sourceterms = np.zeros(self._BIN_X)
		self._escapeterms = np.zeros(self._BIN_X)


	def initialise_kernels(self):
		pass

	def get_radius(self):
		""" Get the current radius"""
		self._radius = getattr(self.sim, 'radius')

	def get_lorentz(self):
		""" Get the Lorentz factor of the plasma """
		self._lorentz = getattr(self.sim, 'lorentz')

	def get_powerlaw_densitydecay(self):
		"""
		Get power-law index for the density decay
		"""
		self._PL= getattr(self.sim, 'pl_decay')

	def get_halfgrid(self):
		"""
		Get the half grid of the energies
		"""
		self._halfgrid = getattr(self.sim, 'half_grid')


	def calculate_and_pass_coefficents(self):
		"""
		Fetch current simulation parameters from the parent simulation manager, f
		ill the internal arrays and pass them to the parent simulation manager.
		"""
		self.get_radius()
		self.get_lorentz()
		self.get_powerlaw_densitydecay()
		self.get_halfgrid()

		self.calculate_terms()

		self.sim.add_to_escapeterms(self._escapeterms)
		self.sim.add_to_heatingterms(self._aterms)


	def calculate_terms(self):
		"""
		Fill the arrays with the escape and cooling terms
		"""
		for k, x in enumerate(self._halfgrid):
			self._aterms[k] = self.adiabatic_cooling(x)

	def t_ad(self):
		"""
		Calculate the standard adiabatic timescale of a plasma expanding with velocity :math:'\beta c', at radius :math:'r'

		;returns: t_ad , the adiabatic cooling timescale [s]
		"""
		t_ad = self._radius* self._lorentz / (c0 * beta(self._lorentz))/ self._PL
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


	def adiabatic_cooling(self, x):
		""" 
		Adiabatic cooling timescale for a plasma expanding with velocity :math:'\beta c', at radius :math:'r' and with 
		Volume evolving as :math:'V \propto r^{s_n}'. 
		The cooling timescale in photon momentum space (which requires extra factor of :math:'8 \pi') is then 

		..math::
			- a_\mathrm{cool} = - \frac{s_n \beta c }{3 \cdot 8 \pi \Gamma r } x 

		:param: x ,photon dimensionless energy
		:returns: cooling timescale :math:'a_\mathrm{cool}' [1/s]
		"""
		return 1/self.t_ad()/3 * x


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
	