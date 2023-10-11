 ######################################################################################
 # This file is part of THECOS (https://github.com/arudol/thecos).
 # Copyright (c) 2023 Annika Rudolph.
 # It includes some of the functionality of pychangcooper (https://github.com/grburgess/pychangcooper), although significantly altered.
 # Copyright (c) 2020 J. Michael Burgess.
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
import math
from tridiagonal_solver import TridiagonalSolver
from mpmath import *
from consts import *
from copy import deepcopy
from scipy.integrate import trapz, simps
from scipy.interpolate import CubicSpline

class ChangCooper(object):
    """
    Class holding the Chang \& Cooper solver methods

    Attributes:
        N(float): Total number of photons
        type_grid (str): linear ("lin") or log-space ("log") grid.

    Properties:
        heating_term (array, floats) : Total heating/Cooling term of the PDE (from radiation modules).
        heating_term_kompaneets (array, floats) : Heating/Cooling term from the Kompaneets Kernel.
        dispersion_term_kompaneets (array, floats) : Dispersion term from the Kompaneets Kernel.
        dispersion_term(array, floats) : Total dispersion term of the PDE (from radiation modules).
        pre_factor_term  : Pre-factor term of the Dispersion + Cooling terms. Equals to 1 if treatment is in energy space, 1/energy^2 for treatment in momentum space.
        escape_term : Total escape/sink term of the PDE (from radiation modules).
        source_term : Total source/Injection term of the PDE (from radiation modules).

    """
    def __init__(
        self,
        grid,
        source_parameters, 
        delta_t=1.0,
        initial_distribution=None, 
        N = 0, 
        type_grid = 'log',
        CN_solver = False, 
    ):
        """
        Initialialise the Chang Cooper solver

        Args:
            grid (array): grid to be used
            source_parameters (dict): Source parameters containing the electron number density and dimensionless temperature,
            delta_t (float): The time step in the equation, default is 1
            initial distribution (array): Initial photon distribution. Optional, default is none.
            N (float): Total number of photons. Optional, default is 0.
            grid_type (str): set to logarithmic ('log') or linear ('lin'). Optional, default is 'log'.
            CN_solver (Bool): Switch to use Crank-Nicolson solver. Optional, default is False.
        """
        
        self._source_parameters = source_parameters
        self.N = N
        self._grid = grid
        self._n_grid_points = len(grid)
        self.phase_space = True
        self.CN_solver = CN_solver
        self._delta_t = delta_t
        self._iterations = 0
        self._current_time = 0.0
        self.type_grid = type_grid
        # first build the grid which is independent of the scheme
        self._build_grid()

        #set up and clear arrays
        self.clear_arrays()
        if initial_distribution is None:
            # initalize the grid of electrons
            self._n_current = np.zeros(self._n_grid_points)
            self._initial_distribution = np.zeros(n_grid_points)

        else:
            assert len(initial_distribution) == self._n_grid_points
            self._n_current = np.array(initial_distribution)
            self._initial_distribution = initial_distribution
        # Pre-set the delta_j to 1/2
        self.delta_j_onehalf()

    @property
    def _N_internal_units(self):
        """
        Set the solver-internal total number N from the real N (includes correction for pre-factors)

        """
        return self.N/  (8* np.pi /(c0*h)**3*(m_e*c0**2)**3)

    @property
    def _Theta_e(self):
        """dimensionless electron temperature """
        return self._source_parameters['T']


    @property
    def _n_e(self):
        """
        electron number density
        """
        return self._source_parameters["n_e"]

    def pass_source_parameters(self, source_params):
        """
        Pass source parameters to internal ones. This is an interface to use from outside. 

        Arg:
            source_params: New source parameters to be stored internally
        """
        self._source_parameters = source_params

    def _build_grid(self):
        """
        setup the grid for the calculations and initialize the
        solution
        """

        # generate the other grids from the input grid

        self._grid, self._half_grid, self._delta_grid, self._delta_half_grid = generate_subgrids_from_nonuniformgrid(self._grid, self.type_grid)
        self._grid2 = self._grid ** 2
        self._half_grid2 = self._half_grid ** 2


    def _find_C_equilibrium_solution(self):
        """
        Calculate the normalisation constant C of the equilibrium function f_e of Chang & Cooper 1970. 

        Returns:
            float: normalisation constant
        """
        prefactor = 2

        last_guess = self._Theta_e**3 *  prefactor/self._N_internal_units

        #def sum_n(x, N):
        #    return sum(x**i/((N+1)-N*i)**3 for i in range(0, N))
        quad_c = -1
        quad_b = -2**3
        quad_a = self._N_internal_units / self._Theta_e**3 /  prefactor

        current_guess =  (-quad_b + np.sqrt(quad_b**2 - 4 * quad_a * quad_c) )/ (2*quad_a)
        mp.dps = 30; mp.pretty = True
        N = 3
        while np.abs((last_guess-current_guess)/current_guess) > 0.01:
            f = lambda x: self._N_internal_units/(prefactor*self._N_internal_units)*x**N - sum(x**(N-i)/(i**3) for i in range(1, N))
            try:
                next_guess = float(findroot(f, current_guess))
            except ValueError:
                return 0.0
            last_guess = current_guess
            current_guess = next_guess
            N +=1 
        C = current_guess
        return C

    def _f_e(self, i, C):
        """
        Calculate the equilibrium function f_e of Chang & Cooper 1970

        Args:
            i (int): Gridpoint
            C (float): Normalisation constant

        Returns:
            float: f_e (i)

        """

        res = 0.0

        lgr = self._grid[i]/self._Theta_e
        if np.abs(lgr) < 1.e-80: lgr = 1.e-80
        
        res = 1 / (C * np.exp(lgr) -1)

        return res

    def compute_delta_j_kompaneets(self):
        """
        Calculate delta_j for the Kompaneets kernel following Chang & Cooper 1970. 

        """

        self._delta_j = np.zeros(self._n_grid_points - 1)

        C = self._find_C_equilibrium_solution()
        if C==0.0:
            for j in range(self._n_grid_points - 1):
                self._delta_j[j] = 1/2.

        else: 
            for j in range(self._n_grid_points - 1):
                ## solve the quadratic equation that corresponds to zero flux condition
                try:
                    fj = self._f_e(j, C)
                    fjplusone = self._f_e(j+1, C)
                    quad_c = self._Theta_e/self._delta_half_grid[j]*(fjplusone - fj) + fjplusone + fjplusone**2
                    quad_b = fj -fjplusone - 2* fjplusone**2 +2*fjplusone*fj
                    quad_a = fjplusone**2 + fj**2
                    if quad_a == 0:
                        self._delta_j[j] = 1/2.
                    else:
                        root_1 = (-quad_b + np.sqrt(quad_b**2 - 4 * quad_a * quad_c) )/ (2*quad_a)
                        if root_1 < 0 or root_1 > 1:
                            root_2 = (-quad_b - np.sqrt(quad_b**2 - 4 * quad_a * quad_c) )/ (2*quad_a)
                            if root_2 < 0 or root_2 > 1:
                                self._delta_j[j] = 1/2.
                            else:
                                self._delta_j[j] = root_2
                        else:
                            self._delta_j[j] = root_1
                except ZeroDivisionError: 
                    self._delta_j[j] = 1/2.

        # precompute 1- delta_j
        self._one_minus_delta_j = 1 - self._delta_j


    def delta_j_onehalf(self):
        """
        Set all delta_j = 1/2

        """

        self._delta_j = np.zeros(self._n_grid_points - 1)

        for j in range(self._n_grid_points - 1):
            self._delta_j[j] = 1/2.

        # precompute 1- delta_j
        self._one_minus_delta_j = 1 - self._delta_j

    def construct_terms_kompaneets(self):
        """
        Constract the dispersion, heating and pre-factor terms of the Kompaneets equation following Chang & Cooper 1970
        """
        self._dispersion_term_kompaneets = np.zeros(len(self._grid)-1)
        self._heating_term_kompaneets = np.zeros(len(self._grid)-1)

        for j in range(self._n_grid_points - 1):
            self._heating_term_kompaneets[j] = 1 + (1 - self._delta_j[j])*self._n_current[j+1] + self._delta_j[j]*self._n_current[j]
            self._heating_term_kompaneets[j] *= self._half_grid[j]**4 * self._n_e * sigma_t *c0

            self._dispersion_term_kompaneets[j] = self._Theta_e
            self._dispersion_term_kompaneets[j] *= self._half_grid[j]**4 * self._n_e * sigma_t *c0

    def construct_terms_kompaneets_extended_by_nu(self):
        """
        Constract the dispersion, heating and pre-factor terms of the Kompaneets equation, extended by Frequency
        """
        self._dispersion_term_kompaneets = np.zeros(len(self._grid)-1)
        self._heating_term_kompaneets = np.zeros(len(self._grid)-1)


        for j in range(self._n_grid_points - 1):
            self._heating_term_kompaneets[j] = 1 + (1 - self._delta_j[j])*self._n_current[j+1] + self._delta_j[j]*self._n_current[j]
            self._heating_term_kompaneets[j] *= self._half_grid[j]**4* (1 + 7/10/self._Theta_e * self._half_grid[j]**2) *self._n_e * sigma_t *c0

            self._dispersion_term_kompaneets[j] = self._Theta_e
            self._dispersion_term_kompaneets[j] *= self._half_grid[j]**4* (1 + 7/10/self._Theta_e*self._half_grid[j]**2) *self._n_e * sigma_t *c0


    def construct_terms_kompaneets_extended_by_p(self):
        """
        Constract the dispersion, heating and pre-factor terms of the Kompaneets equation, extended by momentum
        """
        self._dispersion_term_kompaneets = np.zeros(len(self._grid)-1)
        self._heating_term_kompaneets = np.zeros(len(self._grid)-1)

        for j in range(self._n_grid_points - 1):
            self._heating_term_kompaneets[j] = 1 + (1 - self._delta_j[j])*self._n_current[j+1] + self._delta_j[j]*self._n_current[j]
            self._heating_term_kompaneets[j] *= self._half_grid[j]**4* (1 + 14/5*self._half_grid[j]) *self._n_e * sigma_t *c0

            self._dispersion_term_kompaneets[j] = self._Theta_e
            self._dispersion_term_kompaneets[j] *= self._half_grid[j]**4* (1 + 14/5*self._half_grid[j]) *self._n_e * sigma_t *c0


    def set_kompaneets_terms_zero(self):
        """
        Set the Kompaneets dispersion and advection terms to zeros.
        """
        self._dispersion_term_kompaneets = np.zeros(len(self._grid)-1)
        self._heating_term_kompaneets = np.zeros(len(self._grid)-1)


    def compute_delta_j(self):
        """
        Compute delta_j as specified in CC70. Includes the Kompaneets kernel contributions, that are however zero if not explicitly computed.

        delta_j controls where the differences are computed. If there are no dispersion
        terms, then delta_j is zero
        """

        # set to zero. note delta_j[n] = 0 by default

        self._delta_j = np.zeros(self._n_grid_points - 1)

        dispersion_term_combined  = self._dispersion_term +self._dispersion_term_kompaneets

        heating_term_combined = self._heating_term + self._heating_term_kompaneets

        for j in range(self._n_grid_points - 1):

            # if the dispersion term is 0 => delta_j = 0
            if dispersion_term_combined[j] != 0:

                w = (
                    self._delta_half_grid[1:-1][j] * heating_term_combined[j]
                ) / dispersion_term_combined[j]

                # w asymptotically approaches 1/2, but we need to set it manually
                if w == 0:

                    self._delta_j[j] = 0.5

                # otherwise, we use appropriate bounds
                else:

                    self._delta_j[j] = (1.0 / w) - 1.0 / (np.exp(w) - 1.0)

        # precomoute 1- delta_j
        self._one_minus_delta_j = 1 - self._delta_j

    def compute_energy_transfer_kompaneets(self):
        """ Calculate the energy transfer from the Kompaneets kernel

            Returns:
                float: Energy transfer 
        """
        array_to_integrate = self._grid **3 * self._n_current
        cspline = CubicSpline(self._grid, array_to_integrate)
        first_term = 4 * self._Theta_e*cspline.integrate(min(self._grid), max(self._grid))


        array_to_integrate = self._grid **4* self._n_current * (self._n_current+1)
        cspline = CubicSpline(self._grid, array_to_integrate)
        second_term = - cspline.integrate(min(self._grid), max(self._grid))
        
        lambda_c = h/(m_e *c0)
        res = sigma_t * self._n_e * 8 * np.pi * m_e * c0**3 /lambda_c**3*(first_term +second_term)

        return res

    def _setup_vectors(self):
        """
        From the heating, dispersion, escape/sink and source/injection terms, set up the tri-diagonal matrix.
        """

        # initialize everything to zero

        a = np.zeros(self._n_grid_points)
        b = np.zeros(self._n_grid_points)
        c = np.zeros(self._n_grid_points)

        # If we are in phase space, the advection term and dispersion term are in a 1/x² parenthesis
        if self.phase_space:
            self._pre_factor_term = 1 / self._grid2
        else:
            self._pre_factor_term = 1

        # walk backwards in j starting from the second to last index
        # then set the end points
        for k in range(self._n_grid_points - 2, 1, -1):
            # pre compute one over the delta of the grid
            # this is the 1/delta_grid in front of the F_j +/- 1/2.

            one_over_delta_grid_forward = 1.0 / self._delta_half_grid[k + 1]
            one_over_delta_grid_backward = 1.0 / self._delta_half_grid[k]

            # this is the delta grid in front of the full equation

            one_over_delta_grid_bar = 1.0 / self._delta_grid[k]

            # The B_j +/- 1/2 from CC
            B_forward = self._heating_term[k] + self._heating_term_kompaneets[k] 
            B_backward =self._heating_term[k - 1] + self._heating_term_kompaneets[k - 1]

            # The C_j +/- 1/2 from CC
            C_forward = self._dispersion_term[k] + self._dispersion_term_kompaneets[k]
            C_backward = self._dispersion_term[k - 1]+ self._dispersion_term_kompaneets[k - 1]

            A_k = self._pre_factor_term[k]

            # in order to keep math errors at a minimum, the tridiagonal terms
            # are computed in separate functions so that boundary conditions are
            # set consistently.

            # First we solve (N - N) = F
            # then we will move the terms to form a tridiagonal equation

            # n_j-1 term
            a[k] = _compute_n_j_minus_one_term(
                one_over_delta_grid=one_over_delta_grid_bar,
                one_over_delta_grid_bar_backward=one_over_delta_grid_backward,
                C_backward=C_backward,
                B_backward=B_backward,
                A = A_k, 
                delta_j_minus_one=self._delta_j[k - 1],
            )

            # n_j term
            b[k] = _compute_n_j(
                one_over_delta_grid=one_over_delta_grid_bar,
                one_over_delta_grid_bar_backward=one_over_delta_grid_backward,
                one_over_delta_grid_bar_forward=one_over_delta_grid_forward,
                C_backward=C_backward,
                C_forward=C_forward,
                B_backward=B_backward,
                B_forward=B_forward,
                A = A_k, 
                one_minus_delta_j_minus_one=self._one_minus_delta_j[k - 1],
                delta_j=self._delta_j[k],
            )

            # n_j+1 term
            c[k] = _compute_n_j_plus_one(
                one_over_delta_grid=one_over_delta_grid_bar,
                one_over_delta_grid_bar_forward=one_over_delta_grid_forward,
                C_forward=C_forward,
                B_forward=B_forward,
                A = A_k, 
                one_minus_delta_j=self._one_minus_delta_j[k],
            )

        # now set the end points

        ################
        # right boundary
        # j+1/2 = 0

        one_over_delta_grid_forward = 0.0
        one_over_delta_grid_backward = 1.0 / self._delta_half_grid[-1]

        one_over_delta_grid_bar = 1.0 / self._delta_grid[-1]

        # n_j-1 term
        a[-1] = _compute_n_j_minus_one_term(
            one_over_delta_grid=one_over_delta_grid_bar,
            one_over_delta_grid_bar_backward=one_over_delta_grid_backward,
            C_backward=self._dispersion_term[-1] + self._dispersion_term_kompaneets[-1],
            B_backward=self._heating_term[-1] + self._heating_term_kompaneets[-1],
            A= self._pre_factor_term[-1], 
            delta_j_minus_one=self._delta_j[-1],
        )

        # n_j term
        b[-1] = _compute_n_j(
            one_over_delta_grid=one_over_delta_grid_bar,
            one_over_delta_grid_bar_backward=one_over_delta_grid_backward,
            one_over_delta_grid_bar_forward=one_over_delta_grid_forward,
            C_backward=self._dispersion_term[-1] + self._dispersion_term_kompaneets[-1],
            C_forward=0,
            B_backward =self._heating_term[-1] + self._heating_term_kompaneets[-1],
            B_forward=0,
            A= self._pre_factor_term[-1], 
            one_minus_delta_j_minus_one=self._one_minus_delta_j[-1],
            delta_j=0,
        )

        b[-1] = 0

        # n_j+1 term
        c[-1] = 0

        ###############
        # left boundary
        # j-1/2 = 0

        one_over_delta_grid_forward = 1.0 / self._delta_half_grid[0]
        one_over_delta_grid_backward = 0

        one_over_delta_grid_bar = 1.0 / self._delta_grid[0]

        # n_j-1 term
        a[0] = 0.0

        # n_j term
        b[0] = _compute_n_j(
            one_over_delta_grid=one_over_delta_grid_bar,
            one_over_delta_grid_bar_backward=one_over_delta_grid_backward,
            one_over_delta_grid_bar_forward=one_over_delta_grid_forward,
            C_backward=0,
            C_forward=self._dispersion_term[0] + self._dispersion_term_kompaneets[0],
            B_backward=0,
            B_forward=self._heating_term[0] + self._heating_term_kompaneets[0],
            A= self._pre_factor_term[0],
            one_minus_delta_j_minus_one=0,
            delta_j=self._delta_j[0],
        )
        # n_j+1 term
        c[0] = _compute_n_j_plus_one(
            one_over_delta_grid=one_over_delta_grid_bar,
            one_over_delta_grid_bar_forward=one_over_delta_grid_forward,
            C_forward=self._dispersion_term[0] + self._dispersion_term_kompaneets[0],
            B_forward=self._heating_term[0] + self._heating_term_kompaneets[0],
            A= self._pre_factor_term[0],
            one_minus_delta_j=self._one_minus_delta_j[0],
        )

        # carry terms to the other side to form a tridiagonal equation
        # the escape term is added on but is zero unless created in
        # a child class
        a *= -self._delta_t
        b = (1 - b * self._delta_t) + self._escape_grid * self._delta_t
        c *= -self._delta_t

        return a,b,c
        # now make a tridiagonal_solver for these terms

    def add_source_terms(self, array):
        """Add an array to the source terms of the differential equation. This is an interface to use from outside. 

        Arg:
            array of floats: array to be added, length BIN_X, defined on grid 
        """

        self._source_grid += array

    def add_escape_terms(self, array):
        """Add an array to the escape terms of the differential equation. This is an interface to use from outside. 

        Arg:
            array of floats: array to be added, length BIN_X, defined on grid 
        """
        self._escape_grid += array

    def add_heating_term(self, array):
        """Add an array to the heating terms of the differential equation. This is an interface to use from outside. 

        Arg:
            array of floats: array to be added, length BIN_X, defined on half grid 
        """
        self._heating_term += array

    def add_diffusion_terms(self, array):
        """Add an array to the diffusion terms of the differential equation. This is an interface to use from outside. 

        Arg:
            array of floats: array to be added, length BIN_X, defined on half grid 
        """
        self._diffusion_term += array

    def set_internal_photonarray(self, array):
        """ Set the _n_current to a given array. This is an interface to use from outside. 
        
        Arg:
            array of floats: current photon array of length BIN_X
        """
        self._n_current = deepcopy(array)

    def update_timestep(self, delta_t):  
        """ update the delta t to given value. This is an interface to use from outside. 
        Arg:
            float: delta_t [s] new timestep
        """

        self._delta_t = delta_t

    def solve_time_step(self):
        """
        Solve for the next time step. Note that computation of delta_j and the construction of the kompaneets terms need to be done externally!
        """

        # set up the right side of the tridiagonal equation.
        # This is the current distribution plus the source
        # unless it is zero

        a,b, c = self._setup_vectors()

        for k in range(self._n_grid_points):
            if self._n_current[k] < 1.e-180: self._n_current[k] = 0.0

        d = self._n_current + self._source_grid * self._delta_t

        # Calculate the terms for the Crank-Nicolson solver, see Park & Petrosian for details
        if self.CN_solver: 
            a /= 2.
            b = (b-1)/2. +1
            c /= 2.
            for k in range(self._n_grid_points-2, 1, -1):
                d[k] += self._n_current[k] -a[k]*self._n_current[k-1] -b[k]*self._n_current[k]-c[k]*self._n_current[k+1]
            d[0] += self._n_current[0]  -b[0]*self._n_current[0] -c[0]*self._n_current[1]
            d[-1] += self._n_current[-1] -a[-1]*self._n_current[-2] -b[-1]*self._n_current[-1]

        # compute the next timestep

        self._tridiagonal_solver = TridiagonalSolver(a, b, c)
        self._n_current = self._tridiagonal_solver.solve(d)

        # bump up the iteration number and the time
        self._iterate()

    def clear_arrays(self):
        """
        Clean the internal arrays of the PDE. This is an interface to use from outside. 

        """
        self._dispersion_term = np.zeros(len(self.grid)-1)
        #self._dispersion_term_kompaneets = np.zeros(len(self.grid)-1)
        self._heating_term = np.zeros(len(self.grid)-1)
        #self._heating_term_kompaneets = np.zeros(len(self.grid)-1)
        self._pre_factor_term = np.ones(len(self.grid))
        self._escape_grid = np.zeros(len(self.grid))
        self._source_grid = np.zeros(len(self.grid))

    def _iterate(self):  
        """
        increase the run iterator and the current time
        """

        # bump up the iteration number

        self._iterations += 1

        # increase the time

        self._current_time += self._delta_t

    @property
    def current_time(self):  
        """
        The current time: delta_t * n_iterations
        """

        return self._current_time

    @property
    def n_iterations(self):  
        """
        The number of iterations solved for
        """

        return self._iterations

    @property
    def delta_j(self):  
        """
        the delta_js
        """

        return self._delta_j

    @property
    def grid(self):  
        """
        The energy grid
        """

        return self._grid

    @property
    def half_grid(self):  
        """
        The half energy grid
        """

        return self._half_grid

    @property
    def n(self):  
        """
        The current solution
        """

        return self._n_current


    @property
    def heating_term(self):
        """array (float), the heating term of the PDE """
        return self._heating_term

    @property
    def heating_term_kompaneets(self):
        """array (float), the heating term from the Kompaneets kernel of the PDE """

        return self._heating_term_kompaneets

    @property
    def dispersion_term(self):
        """array (float), The dispersion term of the PDE """

        return self._dispersion_term

    @property
    def dispersion_term_kompaneets(self):
        """array (float), The dispersion term from the Kompaneets kernel of the PDE """

        return self._dispersion_term_kompaneets

    @property
    def pre_factor_term(self):
        """ The pre-factor term in front of dispersion and heating terms.
            Should be 1 for treatment in energy space, 1/x^2 for treatment in momentum space. """
        return self._pre_factor_term

    @property
    def escape_term(self):
        """ array (float), escape term of the equation"""

        return self._escape_grid

    @property
    def source_term(self):
        """ array (float), source term of the PDE"""
        return self._source_grid

    def reset(self):  
        """
        reset the solver (_n_current, _iterations, _current_time) to the distribution. This is an interface to use from outside. 
        """

        self._n_current = self._initial_distribution
        self._iterations = 0
        self._current_time = 0.0



####### Helper functions not part of the class ##############
def _compute_n_j_plus_one(
    one_over_delta_grid,
    one_over_delta_grid_bar_forward,
    C_forward,
    B_forward,
    A, 
    one_minus_delta_j):
     
    """
    equation for the CC n_j +1 term

    Args: 
        one_over_delta_grid (float): the total change in energy
        one_over_delta_grid_bar_backward (float): the backward change in energy for the second derivative
        C_forward (float): the forward dispersion term
        B_forward (float): the forward heating term
        one_minus_delta_j (float): 1 - delta_j
        A (float): the 1/A(x) in front of the kompaneets kernel
    """

    return (A * one_over_delta_grid* (
        one_minus_delta_j * B_forward 
        + one_over_delta_grid_bar_forward * C_forward))


def _compute_n_j(
    one_over_delta_grid,
    one_over_delta_grid_bar_backward,
    one_over_delta_grid_bar_forward,
    C_backward,
    C_forward,
    B_backward,
    B_forward,
    A, 
    one_minus_delta_j_minus_one,
    delta_j):
     
    """
    equation for the CC n_j term
    
    Args: 
        one_over_delta_grid (float): the total change in energy
        one_over_delta_grid_bar_backward (float): the backward change in energy for the second derivative
        one_over_delta_grid_bar_forward (float): the forward change in energy for the second derivative
        C_forward (float): the forward dispersion term
        C_backward (float): the backward dispersion term
        B_forward (float): the forward heating term
        B_backward (float): the backward heating term
        A (float): the 1/A(x) in front of dispersion and advection term
        one_minus_delta_j_minus_one (float): 1 - delta_j-1
    """

    return (- A * one_over_delta_grid* (
        one_over_delta_grid_bar_forward * C_forward 
        + one_over_delta_grid_bar_backward * C_backward 
        + one_minus_delta_j_minus_one * B_backward \
        - delta_j * B_forward))



def _compute_n_j_minus_one_term(
    one_over_delta_grid,
    one_over_delta_grid_bar_backward,
    C_backward,
    B_backward,
    A,
    delta_j_minus_one):
    """
    equation for the CC n_j-1 term

    Args:
        one_over_delta_grid (float): the total change in energy
        one_over_delta_grid_bar_forward (float): the forward change in energy for the second derivative
        C_backward (float): the backward dispersion term
        B_backward (float): the backward heating term
        one_minus_delta_j (float): 1 - delta_j
        A (float): the 1/A(x) in front of the kompaneets kernel
    """

    return (A * one_over_delta_grid* (
        one_over_delta_grid_bar_backward * C_backward 
        -delta_j_minus_one * B_backward ))



def generate_subgrids_from_nonuniformgrid(grid, type_grid):
    """
    Generate sub-grids from a non-uniform grid:
    The half grid (extended towards lower and higher energies)
    and the deltas between the grid points for both grid and half grid.

    Args:
        grid (array, float): Grid to generate the sub-grid from
        type_grid(str, 'log'/'lin'): Log or linear grid

    Returns:
        numpy array : input grid
        numpy array : corresponding half grid (with length -1)
        numpy array : deltas of the grid
        numpy array : deltas of the half grid.
    """
    n_steps = len(grid) 
    half_grid = np.zeros(n_steps - 1)
    delta_half_grid = np.zeros(n_steps+1)
    delta_grid = np.zeros(n_steps)        

    # now build the half grid points
    for i in range(n_steps-1):
        half_grid[i] = 0.5 * (grid[i+1] + grid[i])

    # and build the delta_grids
    # For the delta halfgrid, first fill the parts that truly lie in the array
    for i in range(n_steps-1):
        i+=1
        delta_half_grid[i] = (grid[i] - grid[i-1])


    #then add to the boundaries: extend the grid as a ln grid

    if type_grid=='log': 
        first_step_log = np.log(grid[1]/grid[0])
        gridminusone = np.exp(np.log(grid[0])- first_step_log)
        last_step_log = np.log(grid[-1]/grid[-2])
        gridplusone = np.exp(np.log(grid[-1])+last_step_log)
    elif type_grid == 'lin':
        first_step_log = grid[1]- grid[0]
        gridminusone = grid[0]- first_step_log
        last_step_log = grid[-1] - grid[-2]
        gridplusone = grid[-1]+last_step_log
    
    delta_half_grid[0] = grid[0] - gridminusone
    delta_half_grid[-1] = gridplusone - grid[-1]

    # delta grid from delta halfgrid

    for i in range(n_steps):
        delta_grid[i] = (delta_half_grid[i+1] + delta_half_grid[i])/2.

    return grid, half_grid, delta_grid, delta_half_grid
