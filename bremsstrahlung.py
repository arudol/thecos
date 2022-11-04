import numpy as np
from consts import * 

class Bremsstrahlung(object):

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

		for i in range(self._BIN_X):
			x = self._energygrid[i]
			self._escapeterms[i] = self.alpha_freefree_Vurm2011(x)
			self._sourceterms[i] = self.j_freefree_Vurm2011(x)

		self.sim.add_to_escapeterms(self._escapeterms)
		self.sim.add_to_sourceterms(self._sourceterms)



	def Theta(self, T):
	## Return dimensionless photon energy ##
		return k_B_erg*self._T/(m_e*c0**2)

	def gff(self, E, T):
	## gaunt factor ##
		res = np.sqrt(3)/np.pi* np.log(2.35 * k_B_erg *self._T/E)
		return res

	def alpha_freefree_Vurm2011(self, x):
	## photon absorption as in Vurm 2011 ##
		E = x * m_e*c0**2
		expo_factor = (1- np.exp(-E/(k_B_erg*self._T)))
		prefactor = 4 *e**6 *h**2 /(3 * m_e * c0)* (2/np.pi/(3*k_B_erg*m_e))**(1/2)
		res = prefactor *self._T**(-1/2.)*(self._rho/m_p)**2*E**(-3) *self.gff(E, self._T) * expo_factor
		return res

	def j_freefree_Vurm2011(self, x):
	## photon emission as in Vurm 2011 ##
		E = x * m_e*c0**2
		prefactor =8 *e**6 *c0 /(3 * m_e)* (2/np.pi/(3*k_B_erg*m_e))**(1/2)
		res = prefactor*self._T**(-1/2.)*(self._rho/m_p)**2 *self.gff(E, self._T)
		return res

	def get_injectionrate(self):
		return self._sourceterms

	def get_coolingrate(self):
		return self._escapeterms
	