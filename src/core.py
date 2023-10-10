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
from scipy.integrate import trapz, simps
from scipy.interpolate import *
from solver import ChangCooper
from copy import deepcopy


class SimulationManager(object):
	"""
		Class handling a single simulation run; 
		reads the config, loads the modules specified in the config and initialises/evolves the run
		Important: It also holds the data and grid arrays! All modules need to be initialised with it

		Attributes:
			grid_parameters (dict): Specifics of the grid. Containing number of 
											grid points ('BIN_X', int), 
											the minimum grid point ('X_I', int), 
											grid spacing ('D_X', int)
											and type of grid ('type_grid', 'lin'/'log')
			solver_settings (dict): Solver settings. Containing
											whether to include kompaneets kernel ('include_kompaneets', Bool), 
											if extended by something ('kompaneets_extended_by', 'energy'/'momentum'/'none'),
											if Crank-Nicolson to be applied ('CN_solver', Bool),
											if running in phase space ('phase_space', Bool) 
			source_parameters (dict):  Holding source parameters like electron number density, dimensionless temperature
			module_list (list): list of all radiation modules to be used, each specified by touple (filename, classname)
			
			time (float) : Simulation time in seconds
			delta_t (float): Timestep in seconds
			n_iterations (int) : Number of iterations
			
			N (float) : Total number of photons
			E (float) : Total energy of photons
			energy_change_rate (float): Change of contained energy since last step, calculated as E(last time step) - E(current step)/ delta_t
			photonarray (array, floats) : Current photonarray

			heating_term (array, floats) : Total heating/Cooling term of the PDE (from radiation modules).
			heating_term_kompaneets (array, floats) : Heating/Cooling term from the Kompaneets Kernel.
			dispersion_term_kompaneets (array, floats) : Dispersion term from the Kompaneets Kernel.
			dispersion_term(array, floats) : Total dispersion term of the PDE (from radiation modules).
			pre_factor_term  : Pre-factor term of the Dispersion + Cooling terms. Equals to 1 if treatment is in energy space, 1/energy^2 for treatment in momentum space.
			escape_term : Total escape/sink term of the PDE (from radiation modules).
			source_term : Total source/Injection term of the PDE (from radiation modules).

	"""


	def __init__(self, grid_parameters, delta_t, solver_settings, source_parameters = None, module_list = None):
		"""
		Initialize the SimulationManager

		Args: 

			grid_parameters (dict):	Specifics of the grid. Containing number of 
											grid points ('BIN_X', int), 
											the minimum grid point ('X_I', int), 
											grid spacing ('D_X', int)
											and type of grid ('type_grid', 'lin'/'log')
			delta_t (float): Timestep in seconds
			solver_settings (dict): Solver settings. Containing
											whether to include kompaneets kernel ('include_kompaneets', Bool), 
											if extended by something ('kompaneets_extended_by', 'energy'/'momentum'/'none'),
											if Crank-Nicolson to be applied ('CN_solver', Bool),
											if running in phase space ('phase_space', Bool) 
			source_parameters (dict): Holding source parameters like electron number density, dimensionless temperature. Optional, pre-set to empty. Can be updated later.
			module_list (list): List of all radiation modules to be used, each specified by touple (filename, classname). Optional, pre-set to empty list. Can be updated later.

		"""
		if module_list is None:
			self.module_list = []
		else:
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
		if source_parameters is not None:
			for key in source_parameters:
				self.source_parameters[key] = source_parameters[key]

		self.N = 0 # total photon number
		self.E = 0 # total energy in photons

		# settings for the solver
		self.time = 0
		self.n_iterations = 0

		self.solver_settings = dict()

		# Pre-define settings
		self.solver_settings['include_kompaneets'] = True # Use the Kompaneets Kernel. Bool
		self.solver_settings['kompaneets_extended_by'] = 'none'
		self.solver_settings['compute_delta_j'] = 'classic'
		self.solver_settings['CN_solver'] = False
		self.solver_settings['phase_space'] = True


		# Overwrite with initialisation settings
		if solver_settings is not None : 
			for key in solver_settings:
				self.solver_settings[key] = solver_settings[key]

		self.initialise_arrays()


	def initialise_modules(self):
		"""
		Initialise all modules specified in self.module_list and add them to the current run
		"""
		self.modules = []
		for i in range(len(self.module_list)):
			single_class = dynamic_imp_2(self.module_list[i][0], self.module_list[i][1])
#			from single_module import single_class
			self.modules.append(single_class(self))

	def reset_modules(self, new_modules_list):
		"""
		Set the modules to a new specified list

		Args:
			new_modules_list (list): list of the new radiation modules to be used, each specified by touple (filename, classname)
		"""		
		self.module_list = new_modules_list
		self.modules = []
		for i in range(len(new_modules_list)):
			single_class = dynamic_imp_2(new_modules_list[i][0], new_modules_list[i][1])
#			from single_module import single_class
			self.modules.append(single_class(self))


	def initialise_arrays(self):
		""" Initialize the internal arrays for the computation, i.e. the photonarray and the arrays for the PDE """

		self._photonarray = np.zeros(self.BIN_X)
		self._heating_term = np.zeros(self.BIN_X-1)
		self._dispersion_term = np.zeros(self.BIN_X-1)
		self._source_term = np.zeros(self.BIN_X)
		self._escape_term = np.zeros(self.BIN_X)

		self.heating_term = np.zeros(self.BIN_X-1)
		self.heating_term_kompaneets = np.zeros(self.BIN_X-1)
		self.dispersion_term_kompaneets = np.zeros(self.BIN_X-1)
		self.dispersion_term = np.zeros(self.BIN_X-1)
		self.pre_factor_term = np.zeros(self.BIN_X)
		self.escape_term = np.zeros(self.BIN_X)
		self.source_term = np.zeros(self.BIN_X)

		self.compute_E_total()
		self.compute_N_total()

	def initialise_run(self, input_array = None): 
		"""
		Initialise the run, i.e. the arrays, the modules, and the kernels for the modules.

		Args:
			input_array (array of length BIN_X): Initial photon distribution, optional. 
					If no value is given, initial values are zero.

		Note:
			If input_array has wrong length, photons are initialised as zeros.
		"""

		self.initialise_arrays()
		self.initialise_modules()
		self.initialise_kernels()

		if input_array is not None: 
			if len(input_array) == self.BIN_X:
				self._photonarray = deepcopy(input_array)
			else:
				print("Initial photon array has incorrect length, setting zero")
				self._photonarray = np.zeros(self.BIN_X)

		self._ccsolver = ChangCooper(self.energygrid, self.source_parameters, self.delta_t, self._photonarray, 
			N = self.N, type_grid = self.grid_parameters['type_grid'], CN_solver = self.solver_settings['CN_solver'])

		self.half_grid = self._ccsolver.half_grid
		self.compute_E_total()
		self.compute_N_total()

	def compute_E_total(self):
		"""
		Compute the total energy of photons.
		"""
		prefactor = 8* np.pi /(c0*h)**3*(m_e*c0**2)**4
		#self.E = prefactor * trapz(array_to_integrate[::10], self.energygrid[::10])
		array_to_integrate = self.energygrid *self.energygrid *self.energygrid * self._photonarray
		cspline = CubicSpline(self.energygrid, array_to_integrate)
		self.E = prefactor *cspline.integrate(min(self.energygrid), max(self.energygrid))

	def compute_N_total(self):
		"""
		Compute the total number of photons.
		"""
		prefactor = 8* np.pi /(c0*h)**3*(m_e*c0**2)**3
		#self.N = prefactor * trapz(array_to_integrate[::10], self.energygrid[::10])
		array_to_integrate = self.energygrid *self.energygrid * self._photonarray
		cspline = CubicSpline(self.energygrid, array_to_integrate)
		self.N = prefactor *cspline.integrate(min(self.energygrid), max(self.energygrid))


	def evolve_one_timestep(self):
		"""	
		Evolve the system by one timestep.
		"""
		#Compute total energy and photon number
		self.compute_E_total()
		self.compute_N_total()
		E_last_step = deepcopy(self.E)

		#Clean all solver internal heating, dispersion, source and escape terms
		self._ccsolver.clear_arrays()
		#self.clear_arrays_for_PDE()
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
					self._ccsolver.compute_delta_j_kompaneets()


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

			self.energy_transfer_kompaneets = self._ccsolver.compute_energy_transfer_kompaneets()


		else: 
			self._ccsolver.set_kompaneets_terms_zero()
			self._ccsolver.compute_delta_j()
			#self._ccsolver.delta_j_onehalf()

		#self._ccsolver._compute_boundary()
		# let the solver evolve a timestep
		self._ccsolver.solve_time_step()

		# update the core-internal photon array, and export other terms from the solver for easy readout
		self._photonarray = self._ccsolver.n
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

		self.compute_E_total()
		self.compute_N_total()
		E_current_step = deepcopy(self.E)

		self.energy_change_rate = (E_current_step - E_last_step)/deepcopy(self.delta_t)

	def clear_arrays_for_PDE(self):
		""" Clear internal arrays of the PDE """
		self._heating_term = np.zeros(self.BIN_X-1)
		self._dispersion_term = np.zeros(self.BIN_X-1)
		self._source_term = np.zeros(self.BIN_X)
		self._escape_term = np.zeros(self.BIN_X)

	def clear_arrays_modules(self):
		""" Clear internal arrays of all modules """
		for mod in self.modules:
			mod.clear_arrays()

	def initialise_kernels(self):
		"""Initialise kernels of all modules"""
		for mod in self.modules:
			mod.initialise_kernels()

	## the next four are the interface functions for external modules/ passing arrays at runtime

	def add_to_source_term(self, array):
		""" Add array to the source terms.
		Args:
			array (array, BIN_X): source terms to add

		Note:
			If array is of wrong length, nothing will be done
		"""
		if len(array == self.BIN_X):
			self._source_term += array
		else:
			pass

	def add_to_escape_term(self, array):
		""" Add array to the escape terms.
		Args:
			array (array, BIN_X): escape terms to add

		Note:
			If array is of wrong length, nothing will be done
		"""
		if len(array == self.BIN_X):
			self._escape_term += array
		else:
			pass

	def add_to_heating_term(self, array):
		""" Add array to the heating terms.
		Args:
			array (array, BIN_X): heating terms to add

		Note:
			If array is of wrong length, nothing will be done
		"""
		if len(array == self.BIN_X -1):
			self._heating_term += array
		else:
			pass

	def add_to_dispersion_term(self, array):
		""" Add array to the dispersion terms.
		Args:
			array (array, BIN_X): dispersion terms to add

		Note:
			If array is of wrong length, nothing will be done
		"""
		if len(array == self.BIN_X-1):
			self._dispersion_term += array
		else:
			pass


	# Next two are interfaces to access cooling and injection rates of modules by their name
	def get_coolingrate(self, name_of_class):
		""" get cooling rate for a specific module
		Args:
			name_of_class (str): Name of the module to fetch the cooling rates of

		Returns:
			array: Cooling rate from the module"""
		res = np.zeros(self.BIN_X)
		for i in range(len(self.modules)):
			if self.modules[i].__class__.__name__ == name_of_class:
				res = self.modules[i].get_coolingrate()
		return res

	def get_aterm(self, name_of_class):
		""" get a-term for a specific module
		Args:
			name_of_class (str): Name of the module to fetch the a-term of

		Returns:
			array: a-term from the module"""
		res = np.zeros(self.BIN_X)
		res = np.zeros(self.BIN_X-1)
		for i in range(len(self.modules)):
			if self.modules[i].__class__.__name__ == name_of_class:
				res = self.modules[i].get_aterm()
		return res
	
	def get_injectionrate(self, name_of_class):
		""" get injection rate for a specific module
		Args:
			name_of_class (str): Name of the module to fetch the injection rates of

		Returns:
			array: Injection rates from the module"""
		res = np.zeros(self.BIN_X)
		res = np.zeros(self.BIN_X)
		for i in range(len(self.modules)):
			if self.modules[i].__class__.__name__ == name_of_class:
				res = self.modules[i].get_injectionrate()
		return res

	@property
	def BIN_X(self):
		""" number of grid points"""
		return self.grid_parameters['BIN_X']


	@property
	def photonarray(self):
		""" Photon array (current)"""
		return self._photonarray

	@photonarray.setter
	def photonarray(self, array):
		""" Set photon array.
		Args:
			array (array, BIN_X): new values for the photon array
		"""
		self._photonarray = array


####### Helper functions not part of the class ##############

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