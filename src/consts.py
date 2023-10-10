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


import scipy.constants as const
import numpy as np
# Define useful constants here
# ALL IN CGS !!!

sigma_t = const.physical_constants['Thomson cross section'][0]*1.e4 #cm2
c0 = const.c*1.e2 #cm/s
m_e = const.m_e * 1.e3 #g
m_p = const.m_p * 1.e3 #g
kappa_pp = 0.5
sigma_pp = 5.e-26 # cm2
k_B_erg = 1.380649 * 1.e-16 # erg/K
k_B_eV = 8.617333262*1.e-5 # eV/K
a = 7.56e-15  # erg/cm3/K4
h =6.6261* 1.e-27 #cm2 g s-1 , planck constant
hbar = h/(2* np.pi)
eStatC = 4.8e-10 # statC
alpha_f = eStatC**2/(hbar *c0) # fine structure constant
lambda_C = h /(m_e *c0) # Compton wavelength	

def beta(gamma):
	return np.sqrt(1- 1/gamma**2)

def gamma(beta):
	return np.sqrt(1/(1-beta**2))