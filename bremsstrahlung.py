import numpy as np
from consts import * 

class Bremsstrahlung(SimulationManager):

	def __init__():
		# Initialise the parent class to inherit the grids etc
		super().__init__()

		self.aterms = np.zeros(BIN_X)
		self.sourceterms = np.zeros(BIN_X)
		self.escapeterms = np.zeros(BIN_X)

	def clear_internal_arrays():
		self.aterms = np.zeros(BIN_X)
		self.sourceterms = np.zeros(BIN_X)
		self.escapeterms = np.zeros(BIN_X)

	def initialise_kernels():
		pass

	def Theta(T):
    	return k_B_erg*T/(m_e*c0**2)

	def gff(E, T):
    	res = np.sqrt(3)/np.pi* np.log(2.35 * k_B_erg *T/E)
    	return res

	def alpha_freefree_Vurm2011(self, x, T, rho):
	    E = x * m_e*c0**2
	    exp = (1- np.exp(E/k_B_erg*T))
	    res = 4 *e**6 *h**2 /(3 * m_e * c0)* (2/np.pi/(3*k_B_erg*m_e))**(1/2)*T**(-1/2.)*(rho/m_p)**2*E**(-3) *gff(E, T) * exp
	    return res

	def j_freefree_Vurm2011(x, T, rho):
	    E = x * m_e*c0**2
	    res = 8 *e**6 *c0 /(3 * m_e)* (2/np.pi/(3*k_B_erg*m_e))**(1/2)*T**(-1/2.)*(rho/m_p)**2 *gff(E, T)
	    return res


	