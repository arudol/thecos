import numpy as np
import config as config

import imp
from chang_cooper import ChangCooper


# dynamic import 
def dynamic_imp(name, class_name):
      
    # find_module() method is used
    # to find the module and return
    # its description and path

    fp, path, desc = imp.find_module(name)
            
    try:
    # load_modules loads the module 
    # dynamically ans takes the filepath
    # module and description as parameter
        example_package = imp.load_module(name, fp,
                                          path, desc)
          
    except Exception as e:
        print(e)
          
    try:
        myclass = imp.load_module("% s.% s" % (name,
                                               class_name), 
                                  fp, path, desc)
          
    except Exception as e:
        print(e)
          
    return example_package, myclass


class SimulationManager(object):
## This class handles the simulation run: ## 
## It reads the config, loads the modules specified in the config and initialises/evolves the run ##
## Important: It also holds the data and grid arrays! All modules need to be initialised with it## 

	def __init__(self, config):
		self.BIN_X = config.BIN_X
		self.XI = config.XI
		self.D_X = config.D_X
		self.delta_t = config.delta_t
		self.energygrid = np.asarray([np.exp((self.XI + i) * self.D_X) for i in range(self.BIN_X)])

		# not elegant right now: properties of the plasma belong to the sim manager
		self.T = 0 # electron temperature
		self.rho = 0 # electron density
		self.lorentz = 0 #plasma lorentz factor
		self.radius = 0 # radius from central engine
		self.bprime = 0 # comoving magnetic field


	def initialise_modules(self):
	## Reads through all the modules specified in the config and adds them to the run ##
		self.modules = []
		for i in range(len(config.modules)):
			single_module, single_class = dynamic_imp(config.modules[i][0], config.modules[i][1])
			from single_module import single_class
			self.modules.append(single_class(self))

	def initialise_arrays(self):
	## Sets the photon array and the terms of the arrays holding the terms of the equation to zero##

		self.photonarray = np.zeros(self.BIN_X)
		self._heatingterms = np.zeros(self.BIN_X)
		self._dispersionterms = np.zeros(self.BIN_X)
		self._sourceterms = np.zeros(self.BIN_X)
		self._escapeterms = np.zeros(self.BIN_X)

	def initialise_run(self, input_array= np.empty(self.BIN_X)): 
	## Set arrays to zero, add all modules to run, initialise kernels of the modules.##
	## If an array is passed, it is taken as the initial photon distribution ##

		self.initialise_arrays()
		self.initialise_modules()
		self.initialise_kernels()

		if input_array: 
			if len(input_array) == self.BIN_X:
				self.photonarray = input_array
			else:
				print("Initial photon array has incorrect length, setting zero")
				self.photonarray = np.zeros(self.BIN_X)

		ccsolver = ChangCooper(self.grid, self.delta_t, self.photonarray)

	def evolve_one_timestep(self):
	## Evolve the photon distribution for one timestep ## 

		# Iterate through all modules and add make them add terms to the source/escape/heating/dispersion arrays
		for mod in self.modules:
			mod.calculate_and_pass_coefficents()

		# pass the cooling etc terms to the solver
		ccsolver.pass_source_terms(self._sourceterms)
		ccsolver.pass_escape_terms(self._escapeterms)
		ccsolver.pass_heating_terms(self._heatingterms)
		ccsolver.pass_diffusion_terms(self._dispersionterms)

		# let the solver evolve a timestep
		ccsolver.solve_time_step()

		# update the core-internal photon array
		self.photonarray = ccsolver._n_current


	## the next four are the interface functions for external modules. 

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
		if len(array == self.BIN_X):
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