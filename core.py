import numpy as np
import .config as config

import imp

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
          
    return myclass


class SimulationManager(object):
## This class handles the simulation run: ## 
## It reads the config, loads the modules specified in the config and initialises/evolves the run ##
## Important: It also holds the data and grid arrays! All modules need to have it as a parent class ## 

	def __init__(self, config):
		self.BIN_X = config.BIN_X
		self.XI = config.XI
		self.D_X = config.D_X
		self.energygrid = np.asarray([np.exp((self.XI + i) * self.D_X) for i in self.BIN_X])

		

	def initialise_modules(self):
	## Reads through all the modules specified in the config and adds them to the run ##
		self.modules = []
		for i in range(len(config.modules)):
			self.modules.append(dynamic_imp(config.modules[i][0], config.modules[i][1]))

	def initialise_arrays(self):
	## Sets the photon array and the terms of the arrays holding the terms of the equation to zero##

		self.photonarray = np.zeros(self.BIN_X)
		self.aterms = np.zeros(self.BIN_X)
		self.sourceterms = np.zeros(self.BIN_X)
		self.escapeterms = np.zeros(self.BIN_X)

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
				print("Initial photon array has incorrect length")

	def evolve_one_timestep():
	## Evolve the photon distribution for one timestep ## 
		for mod in self.modules:
			mod.pass_coefficients(self.photonarray)

	def clear_internal_arrays():
	## Clear internal arrays of all modules ##
		for mod in self.modules:
			mod.clear_arrays()

	def initialise_kernels():
	## Initialise kernels of all modules ##
		for mod in self.modules:
			mod.initialise_kernels()