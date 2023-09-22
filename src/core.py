import numpy as np
import config as config
from consts import *
from scipy.integrate import trapz, simps
from scipy.interpolate import *
from chang_cooper_kompaneets_w_seminanalytical import ChangCooper
from copy import deepcopy

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

	def __init__(self, grid_parameters, delta_t, solver_settings, source_parameters = {}, module_list = []):
		"""
		Initialisation of the run. 

		:param: grid_paramters - dict containing the number of grid points BIN_X, the minimum grid point X_I, grid spacing D_X
						and type fo grid ('lin' or 'log', default is log)
		:param: delta_t - timestep in seconds
		:param: solver settings - dict containing solver settings like whether to include kompaneets kernel, if extended by frequency or momentum, 
						and if Cranck Nicolson sovler
		:param: source parameters - dict holding source parameters like electron number density, dimensionless temperature, etc. default is empty.
		:param: module list - radiation modules to be used. Have to specified by touple (filename, classname)
		"""

		self.module_list = module_list

		self.grid_parameters = dict()
		self.grid_parameters['BIN_X'] = grid_parameters['BIN_X']
		self.grid_parameters['X_I'] = grid_parameters['X_I']
		self.grid_parameters['D_X'] = grid_parameters['D_X']

		if "type_grid" in grid_parameters:
			self.grid_parameters['type_grid'] = grid_parameters['type_grid']
		else:
			self.grid_parameters['type_grid'] = 'log'

		if self.grid_parameters['type_grid'] == 'log':
			self.energygrid = np.asarray([np.exp((self.grid_parameters['X_I']  + i) * grid_parameters['D_X']) for i in range(self.grid_parameters['BIN_X'])])
		elif self.grid_parameters['type_grid'] == 'lin':
			self.energygrid = np.asarray([(self.grid_parameters['X_I'] + i) * grid_parameters['D_X'] for i in range(self.grid_parameters['BIN_X'])])
		else : raise TypeError("No valid grid type selected")

		
		self.delta_t = delta_t

		self.source_parameters = dict()

		# Overwrite with initialisation settings
		for key in source_parameters:
			self.source_parameters[key] = source_parameters[key]

		self.N = 0 # total photon number
		self.E = 0 # total energy in photons

		for key in source_parameters:
			self.source_parameters[key] = source_parameters[key]


		# settings for the solver
		self.time = 0
		self.n_iterations = 0

		self.solver_settings = dict()

		# Pre-define settings
		self.solver_settings['include_kompaneets'] = True # Use the Kompaneets Kernel. Bool
		self.solver_settings['kompaneets_extended_by'] = 'none'
		self.solver_settings['compute_delta_j'] = 'classic'
		self.solver_settings['CN_solver'] = False
		self.solver_settings['solver_type'] = 'matrix'
		self.solver_settings['phase_space'] = True


		# Overwrite with initialisation settings
		for key in solver_settings:
			self.solver_settings[key] = solver_settings[key]


	def initialise_modules(self):
	## Reads through all the modules specified in the config and adds them to the run ##
		self.modules = []
		for i in range(len(self.module_list)):
			single_class = dynamic_imp_2(self.module_list[i][0], self.module_list[i][1])
#			from single_module import single_class
			self.modules.append(single_class(self))

	def reset_modules(self, new_modules_list):
	## Reads through all the modules specified in the new_modules_list and sets them to the simulation ##
		self.module_list = new_modules_list
		self.modules = []
		for i in range(len(new_modules_list)):
			single_class = dynamic_imp_2(new_modules_list[i][0], new_modules_list[i][1])
#			from single_module import single_class
			self.modules.append(single_class(self))


	def initialise_arrays(self):
	## Sets the photon array and the terms of the arrays holding the terms of the equation to zero##

		self._photonarray = np.zeros(self.BIN_X)
		self._heating_term = np.zeros(self.BIN_X-1)
		self._dispersion_term = np.zeros(self.BIN_X-1)
		self._source_term = np.zeros(self.BIN_X)
		self._escape_term = np.zeros(self.BIN_X)

	def initialise_run(self, input_array = []): 
	## Set arrays to zero, add all modules to run, initialise kernels of the modules.##
	## If an array is passed, it is taken as the initial photon distribution ##

		self.initialise_arrays()
		self.initialise_modules()
		self.initialise_kernels()

		if input_array != []: 
			if len(input_array) == self.BIN_X:
				self._photonarray = deepcopy(input_array)
			else:
				print("Initial photon array has incorrect length, setting zero")
				self._photonarray = np.zeros(self.BIN_X)

		self._ccsolver = ChangCooper(self.energygrid, self.source_parameters, self.delta_t, self._photonarray, 
			N = self.N, type_grid = self.grid_parameters['type_grid'], CN_solver = self.solver_settings['CN_solver'])

		self.half_grid = self._ccsolver.half_grid

	def compute_E_total(self):
	## Compute the total energy in the photon field ##
		prefactor = 8* np.pi /(c0*h)**3*(m_e*c0**2)**4
		#self.E = prefactor * trapz(array_to_integrate[::10], self.energygrid[::10])
		array_to_integrate = self.energygrid *self.energygrid *self.energygrid * self._photonarray
		cspline = CubicSpline(self.energygrid, array_to_integrate)
		self.E = prefactor *cspline.integrate(min(self.energygrid), max(self.energygrid))
		#chi2 = chisquare(self.energygrid, array_to_integrate, cspline)
		#print(chi2)

	def compute_N_total(self):
	## Compute the total number of photons ##
		prefactor = 8* np.pi /(c0*h)**3*(m_e*c0**2)**3
		#self.N = prefactor * trapz(array_to_integrate[::10], self.energygrid[::10])
		array_to_integrate = self.energygrid *self.energygrid * self._photonarray
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
		self._ccsolver.clear_arrays()
		self.clear_arrays_for_PDE()
		self.clear_arrays_modules()

		# Iterate through all modules and add make them add terms to the source/escape/heating/dispersion arrays
		for mod in self.modules:
			mod.calculate_and_pass_coefficents()

		# Update the quantities in the solver
		self._ccsolver.pass_source_parameters(self.source_parameters)
		self._ccsolver.N = self.N
		self._ccsolver.phase_space = self.solver_settings['phase_space']

		if self.solver_settings['include_kompaneets'] and not self.solver_settings['phase_space']:
			raise Exception('Kompaneets kernel can only be used if calculations are carried out in phase space!')

		# pass the cooling etc terms to the solver
		self._ccsolver.add_source_terms(self._source_term)
		self._ccsolver.add_escape_terms(self._escape_term)
		self._ccsolver.add_heating_term(self._heating_term)
		self._ccsolver.set_internal_photonarray(self.photonarray)
		self._ccsolver.update_timestep(self.delta_t)
		#self._ccsolver.pass_diffusion_terms(self._dispersion_term)

		if self.solver_settings['include_kompaneets']: 
			if self.n_iterations == 0:
					if not 'T' in self.source_parameters:
						raise Exception('No electron temperature provided, necessary for Kompaneets Compton scattering')
					if not 'n_e' in self.source_parameters:
						raise Exception('No electron number density provided, necessary for Kompaneets Compton scattering')

			if self.solver_settings['kompaneets_extended_by'] == 'none':
				self._ccsolver.construct_terms_kompaneets()
			elif self.solver_settings['kompaneets_extended_by'] == 'frequency':
				self._ccsolver.construct_terms_kompaneets_extended_by_nu()
			elif self.solver_settings['kompaneets_extended_by'] == 'momentum':
				self._ccsolver.construct_terms_kompaneets_extended_by_p()


			if self.solver_settings['compute_delta_j'] == 'kompaneets':
				self._ccsolver.compute_delta_j_kompaneets()
			elif self.solver_settings['compute_delta_j'] == 'classic':
				self._ccsolver.compute_delta_j()


		else: 
			self._ccsolver.set_kompaneets_terms_zero()
			self._ccsolver.compute_delta_j()
			#self._ccsolver.delta_j_onehalf()

		self._ccsolver._compute_boundary()
		# let the solver evolve a timestep
		self._ccsolver.solve_time_step(solver = self.solver_settings['solver_type'])

		# update the core-internal photon array, and export other terms from the solver for easy readout
		self._photonarray = self._ccsolver.n
		self.delta_j = self._ccsolver.delta_j
		self.time  = self._ccsolver.current_time
		self.n_iterations = self._ccsolver.n_iterations

		#For external readout: store arrays from solver
		self.heating_term = self._ccsolver.heating_term
		self.heating_term_kompaneets = self._ccsolver.heating_term_kompaneets
		self.dispersion_term_kompaneets = self._ccsolver.dispersion_term_kompaneets
		self.dispersion_term = self._ccsolver.dispersion_term
		self.pre_factor_term = self._ccsolver.pre_factor_term
		self.escape_term = self._ccsolver.escape_term
		self.source_term = self._ccsolver.source_term

	def clear_arrays_for_PDE(self):
		self._heating_term = np.zeros(self.BIN_X-1)
		self._dispersion_term = np.zeros(self.BIN_X-1)
		self._source_term = np.zeros(self.BIN_X)
		self._escape_term = np.zeros(self.BIN_X)

	def set_photonarray(self, array):
		self._photonarray = array

	## the next four are the interface functions for external modules/ passing arrays at runtime

	def add_to_source_term(self, array):
		if len(array == self.BIN_X):
			self._source_term += array
		else:
			pass

	def clear_arrays_modules(self):
	## Clear internal arrays of all modules ##
		for mod in self.modules:
			mod.clear_arrays()

	def initialise_kernels(self):
	## Initialise kernels of all modules ##
		for mod in self.modules:
			mod.initialise_kernels()

	def add_to_escape_term(self, array):
		if len(array == self.BIN_X):
			self._escape_term += array
		else:
			pass

	def add_to_heating_term(self, array):
		if len(array == self.BIN_X -1):
			self._heating_term += array
		else:
			pass

	def add_to_dispersion_term(self, array):
		if len(array == self.BIN_X-1):
			self._dispersion_term += array
		else:
			pass


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

	@property
	def BIN_X(self):
		return self.grid_parameters['BIN_X']


	@property
	def photonarray(self):
		return self._photonarray
	