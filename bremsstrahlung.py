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

	def clear_arrays(self):
		self._aterms = np.zeros(self._BIN_X)
		self._sourceterms = np.zeros(self._BIN_X)
		self._escapeterms = np.zeros(self._BIN_X)

	def initialise_kernels(self):
		pass

	def get_temperature(self):
		## Get electron temperature from simulation class ## 
		self._Theta = getattr(self.sim, 'T') # dimensionless
		self._T = self._Theta * m_e * c0**2 / k_B_erg # Kelvin

	def get_density(self):
		## Get matter density from simulation class ## 
		self._n_e = getattr(self.sim, 'n_e')

	def get_current_photonarray(self):
		## Get current photon array from simulation class ## 
		self._photonarray = getattr(self.sim, 'photonarray')


	def get_current_photonnumber(self):
		## Get current photon N from simulation class ## 
		self._N = getattr(self.sim, 'N')


	def calculate_and_pass_coefficents(self):
	## Calculate all escape and sink terms ##
		self.get_temperature()
		self.get_density()
		self.get_current_photonarray()
		self.calculate_terms()

		self.sim.add_to_escapeterms(self._escapeterms)
		self.sim.add_to_sourceterms(self._sourceterms)

	def calculate_terms(self):
		for k in range(self._BIN_X):
			x = self._energygrid[k]
			self._escapeterms[k] = self.alpha_freefree_Vurm2011(x)
			self._sourceterms[k] = self.j_freefree_Vurm2011(x)

	def Theta(self, T):
	## Return dimensionless photon energy ##
		return k_B_erg*self._T/(m_e*c0**2)

	def gff(self, x, Theta):
	## gaunt factor ##
		res = np.sqrt(3)/np.pi* np.log(2.25 * Theta/x)
		return res

	def alpha_freefree_Vurm2011(self, x):
	## photon absorption as in Vurm 2011 ##
		E = x * m_e*c0**2
		lgr = E/(k_B_erg*self._T)

		#if lgr > 1.e10: lgr = 1.e10
		expo_factor = (1- np.exp(-lgr))

		#prefactor = 4 *e**6 *h**2 /(3 * m_e * c0)* (2/np.pi/(3*k_B_erg*m_e))**(1/2)
		prefactor = alpha_f  * lambda_C**3 *sigma_t * c0/(np.sqrt(3 *8* np.pi**3))
		#res = prefactor *self._T**(-1/2.)*(self._n_e/m_p)**2*E**(-3) *self.gff(E, self._T) * expo_factor
		res = prefactor *self._Theta**(-1/2.)*(self._n_e)**2*x**(-3) *self.gff(x, self._Theta) * expo_factor
		if res < 0.0: res = 0.0
		return res

	def j_freefree_Vurm2011(self, x):
	## photon emission as in Vurm 2011 ##
		#E = x * m_e*c0**2
		#prefactor =8 *e**6 *c0 /(3 * m_e)* (2/np.pi/(3*k_B_erg*m_e))**(1/2)
		#res = prefactor*self._T**(-1/2.)*(self._n_e/m_p)**2 *self.gff(E, self._T)
		#res = prefactor*self._T**(-1/2.)*(self._n_e)**2 *self.gff(E, self._T)

		E = x * m_e*c0**2
		alpha = self.alpha_freefree_Vurm2011(x)

		lgr = x/self._Theta

		#B_E = 2 * E**3 /(h*c0)**2 * (np.exp(lgr) -1 )**(-1)
		B_E = (np.exp(lgr) -1 )**(-1)
		res = alpha * B_E

		return res

	def get_injectionrate(self):
		return self._sourceterms

	def get_coolingrate(self):
		return self._escapeterms
	