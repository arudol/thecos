import numpy as np
import config as config
from consts import *
from scipy.integrate import trapz, simps
from scipy.interpolate import *
from chang_cooper_kompaneets import ChangCooper


# dynamic import 
def dynamic_imp(name, class_name):
	import imp
	fp, path, desc = imp.find_module(name)
	example_package = imp.load_module(name, fp,path, desc)
	myclass = imp.load_module("% s.% s" % (name,class_name), fp, path, desc)
	print(example_package, myclass)
	return example_package, myclass

def dynamic_imp_2(name, class_name):
	import importlib
	module = importlib.import_module(name)
	my_class = getattr(module, class_name)
	return my_class

def chisquare(energygrid, array, interpolation):
	chi2_array = (array - interpolation(energygrid))/array 
	chi2 = np.sum(chi2_array)
	return chi2



class SimulationManager(object):
## This class handles the simulation run: ## 
## It reads the config, loads the modules specified in the config and initialises/evolves the run ##
## Important: It also holds the data and grid arrays! All modules need to be initialised with it## 

	def __init__(self, BIN_X, X_I, D_X, delta_t, type_grid = 'log', CN_solver = False, include_kompaneets = True, 
		kompaneets_extended_by = 'none', compute_delta_j = 'kompaneets'):

		self.BIN_X = BIN_X
		self.X_I = X_I
		self.D_X = D_X
		self.delta_t = delta_t
		if type_grid == 'log':
			self.energygrid = np.asarray([np.exp((self.X_I + i) * self.D_X) for i in range(self.BIN_X)])
		else: 
			self.energygrid = np.asarray([(self.X_I + i) * self.D_X for i in range(self.BIN_X)])
		#self.energygrid = np.asarray([np.exp( i * self.D_X) for i in range(self.BIN_X)])
		
		# not elegant right now: properties of the plasma belong to the sim manager
		self.type_grid = type_grid
		self.T = 0 # electron temperature
		self.lorentz = 0 #plasma lorentz factor
		self.radius = 0 # radius from central engine
		self.bprime = 0 # comoving magnetic field
		self.N = 0 # total photon number
		self.n_e = 0 # electron number density 
		self.CN_solver = CN_solver # Crank Nicolson solver: bool for on (True) and off (False)
		self.pl_decay = 2.

		# counters for the solver
		self.time = 0
		self.n_iterations = 0
		self.include_kompaneets = include_kompaneets # Use the Kompaneets Kernel. Bool
		self.kompaneets_extended_by = kompaneets_extended_by
		self.compute_delta_j = compute_delta_j


	def initialise_modules(self):
	## Reads through all the modules specified in the config and adds them to the run ##
		self.modules = []
		for i in range(len(config.modules)):
			single_class = dynamic_imp_2(config.modules[i][0], config.modules[i][1])
#			from single_module import single_class
			self.modules.append(single_class(self))

	def reset_modules(self, new_modules_list):
	## Reads through all the modules specified in the new_modules_list and sets them to the simulation ##
		self.modules = []
		for i in range(len(new_modules_list)):
			single_class = dynamic_imp_2(new_modules_list[i][0], new_modules_list[i][1])
#			from single_module import single_class
			self.modules.append(single_class(self))


	def initialise_arrays(self):
	## Sets the photon array and the terms of the arrays holding the terms of the equation to zero##

		self.photonarray = np.zeros(self.BIN_X)
		self._heatingterms = np.zeros(self.BIN_X-1)
		self._dispersionterms = np.zeros(self.BIN_X-1)
		self._sourceterms = np.zeros(self.BIN_X)
		self._escapeterms = np.zeros(self.BIN_X)

	def initialise_run(self, input_array = []): 
	## Set arrays to zero, add all modules to run, initialise kernels of the modules.##
	## If an array is passed, it is taken as the initial photon distribution ##

		self.initialise_arrays()
		self.initialise_modules()
		self.initialise_kernels()

		if input_array != []: 
			if len(input_array) == self.BIN_X:
				self.photonarray = input_array
			else:
				print("Initial photon array has incorrect length, setting zero")
				self.photonarray = np.zeros(self.BIN_X)

		self.ccsolver = ChangCooper(self.energygrid, self.delta_t, self.photonarray, 
			Theta_e = self.T, n_e = self.n_e, N = self.N, type_grid = self.type_grid, CN_solver = self.CN_solver)

		self.half_grid = self.ccsolver.half_grid

	def compute_E_total(self):
	## Compute the total energy in the photon field ##
		prefactor = 8* np.pi /(c0*h)**3*(m_e*c0**2)**4
		#self.E = prefactor * trapz(array_to_integrate[::10], self.energygrid[::10])
		array_to_integrate = self.energygrid *self.energygrid *self.energygrid * self.photonarray
		cspline = CubicSpline(self.energygrid, array_to_integrate)
		self.E = prefactor *cspline.integrate(min(self.energygrid), max(self.energygrid))
		#chi2 = chisquare(self.energygrid, array_to_integrate, cspline)
		#print(chi2)

	def compute_N_total(self):
	## Compute the total number of photons ##
		prefactor = 8* np.pi /(c0*h)**3*(m_e*c0**2)**3
		#self.N = prefactor * trapz(array_to_integrate[::10], self.energygrid[::10])
		array_to_integrate = self.energygrid *self.energygrid * self.photonarray
		cspline = CubicSpline(self.energygrid, array_to_integrate)
		self.N = prefactor *cspline.integrate(min(self.energygrid), max(self.energygrid))
		#chi2 = chisquare(self.energygrid, array_to_integrate, cspline)
		#print(chi2)

	def evolve_one_timestep(self):
	## Evolve the photon distribution for one timestep ## 


		#Compute total energy and photon number
		self.compute_E_total()
		self.compute_N_total()

		#Clean all solver internal heating, dispersion, source and escape terms
		self.ccsolver.clean_terms()

		# Iterate through all modules and add make them add terms to the source/escape/heating/dispersion arrays
		for mod in self.modules:
			mod.calculate_and_pass_coefficents()

		# Update the quantities in the solver
		self.ccsolver.Theta_e = self.T
		self.ccsolver.n_e = self.n_e
		self.ccsolver.delta_t = self.delta_t
		self.ccsolver.N = self.N

		if self.include_kompaneets: 
			if self.kompaneets_extended_by == 'none':
				self.ccsolver._construct_terms_kompaneets()
			elif self.kompaneets_extended_by == 'frequency':
				self.ccsolver._construct_terms_kompaneets_extended_by_nu()
			elif self.kompaneets_extended_by == 'momentum':
				self.ccsolver._construct_terms_kompaneets_extended_by_p()

			if self.compute_delta_j == 'kompaneets':
				self.ccsolver._compute_delta_j_kompaneets()
			elif self.compute_delta_j == 'classic':
				self.ccsolver._compute_delta_j_mix()

		else: 
			self.ccsolver._compute_delta_j()

		# pass the cooling etc terms to the solver
		self.ccsolver.add_source_terms(self._sourceterms)
		self.ccsolver.add_escape_terms(self._escapeterms)
		self.ccsolver.add_heating_terms(self._heatingterms)
		#self.ccsolver.pass_diffusion_terms(self._dispersionterms)
		
		# let the solver evolve a timestep
		self.ccsolver.solve_time_step()

		# update the core-internal photon array, and export other terms from the solver for easy readout
		self.photonarray = self.ccsolver.n
		self.delta_j = self.ccsolver.delta_j
		self.time  = self.ccsolver.current_time
		self.n_iterations = self.ccsolver.n_iterations

		self.heating_term = self.ccsolver._heating_term
		self.heating_term_kompaneets = self.ccsolver._heating_term_kompaneets
		self.dispersion_term_kompaneets = self.ccsolver._dispersion_term_kompaneets
		self.dispersion_term = self.ccsolver._dispersion_term

	## the next four are the interface functions for external modules/ passing arrays at runtime

	def add_to_sourceterms(self, array):
		if len(array == self.BIN_X):
			self._sourceterms += array
		else:
			pass

	def add_to_escapeterms(self, array):
		if len(array == self.BIN_X):
			self._escapeterms += array
		else:
			pass

	def add_to_heatingterms(self, array):
		if len(array == self.BIN_X -1):
			self._heatingterms += array
		else:
			pass

	def add_to_dispersionterms(self, array):
		if len(array == self.BIN_X):
			self._dispersionterms += array
		else:
			pass

	def clear_internal_arrays(self):
	## Clear internal arrays of all modules ##
		for mod in self.modules:
			mod.clear_arrays()

	def initialise_kernels(self):
	## Initialise kernels of all modules ##
		for mod in self.modules:
			mod.initialise_kernels()


	# Next two are interfaces to access cooling and injection rates of modules by their name
	def get_coolingrate(self, name_of_class):
		res = np.zeros(self.BIN_X)
		for i in range(len(self.modules)):
			if self.modules[i].__class__.__name__ == name_of_class:
				res = self.modules[i].get_coolingrate()
		return res

	def get_aterm(self, name_of_class):
		res = np.zeros(self.BIN_X-1)
		for i in range(len(self.modules)):
			if self.modules[i].__class__.__name__ == name_of_class:
				res = self.modules[i].get_aterm()
		return res
	
	def get_injectionrate(self, name_of_class):
		res = np.zeros(self.BIN_X)
		for i in range(len(self.modules)):
			if self.modules[i].__class__.__name__ == name_of_class:
				res = self.modules[i].get_injectionrate()
		return res