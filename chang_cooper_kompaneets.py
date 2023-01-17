import numpy as np

from tridiagonal_solver import TridiagonalSolver
from mpmath import *
from consts import *
from scipy.integrate import trapz, simps


def generate_subgrids_from_nonuniformgrid(grid, type_grid):
    n_steps = len(grid) 
    half_grid = np.zeros(n_steps - 1)

    # now build the half grid points
    for i in range(n_steps-1):
        half_grid[i] = 0.5 * (grid[i+1] + grid[i])

    # and build the delta_grids
    # For the delta halfgrid, first fill the parts that truly lie in the array
    delta_half_grid = np.zeros(n_steps+1)
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

    delta_grid = np.zeros(n_steps)        
    for i in range(n_steps):
        delta_grid[i] = (delta_half_grid[i+1] + delta_half_grid[i])/2.

    return grid, half_grid, delta_grid, delta_half_grid


class ChangCooper(object):
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
        Generic Chang and Cooper base class for Kompaneets equation.
        :param grid: grid to be used
        :param delta_t: the time step in the equation
        :param initial_distribution: an array of an initial electron distribution
        :param Theta_e : electron temperature
        :param n_e: electron number density
        :param N: total number of photons
        :param type_grid: set to logarithmic ('log') or linear ('lin'), default is 'log'
        :param CN_solver: Switch to use Crank-Nicolson solver, default is False
        """
        
        self._source_parameters = source_parameters
        self.N = N
        self._grid = grid
        self._n_grid_points = len(grid)

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

        # now compute the tridiagonal terms
        self._setup_vectors()


    @property
    def _N_internal_units(self):
        """
        Set the solver-internal total number N from the real N (includes correction for pre-factors)

        """
        return self.N/ 8 * np.pi /(c0*h)**3*(m_e*c0**2)**3

    @property
    def _Theta(self):
        """dimensionless electron temperature """
        return self._source_parameters['T']


    @property
    def _n_e(self):
        """
        electron number density
        """
        return self._source_parameters["n_e"]


    def pass_source_parameters(self, source_params):
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
        Calculate the normalisation constant C of the equilibrium function f_e of Chang & Cooper 1970

        :returns: C

        """
        #prefactor = 8 * np.pi /(c0*h)**3*(m_e*c0**2)**3
        prefactor = 8 * np.pi

        last_guess = self.Theta_e**3 *  prefactor/self._N_internal_units

        #def sum_n(x, N):
        #    return sum(x**i/((N+1)-N*i)**3 for i in range(0, N))
        quad_c = -1/(2**3)
        quad_b = -1
        quad_a = self._N_internal_units / self.Theta_e**3 /  prefactor

        current_guess =  (-quad_b + np.sqrt(quad_b**2 - 4 * quad_a * quad_c) )/ (2*quad_a)
        mp.dps = 30; mp.pretty = True
        N = 3
        while np.abs((last_guess-current_guess)/current_guess) > 0.01:
            f = lambda x: self._N_internal_units/(prefactor*self.Theta_e**3)*x**N - sum(x**i/((N+1)-N*i)**3 for i in range(0, N))
            try:
                next_guess = float(findroot(f, current_guess))
            except ValueError: 
                    return 0.0
            last_guess = current_guess
            current_guess = next_guess
            N +=1 
        C = current_guess
        print(C, N)
        #quad_c = -1/(2**3)
        #quad_b = -1
        #quad_a = self._N_internal_units / self.Theta_e**3 /  prefactor

        #first_guess = (-quad_b + np.sqrt(quad_b**2 - 4 * quad_a * quad_c) )/ (2*quad_a)

        #f = lambda x: self._N_internal_units/(prefactor*self.Theta_e**3)*x**6 - \
        #                   (x**5 + 1/2**3 * x**4 + 1/3**3 *x**3 + 1/ 4**3 *x**2 + 1/ 5**3 *x + 1/6**3 ) 

        #mp.dps = 30; mp.pretty = True

        #first_guess = self.Theta_e**3 *  prefactor/self._N_internal_units

        #try:
            #C = float(findroot(f, first_guess))
        #except ValueError: 
        #    C = 0.0

        return C

    def _f_e(self, i, C):
        """
        Calculate the equilibrium function f_e of Chang & Cooper 1970

        :param i: Gridpoint
        :param C: Normalisation constant

        :returns: f_e (i)

        """

        res = 0.0

        lgr = self._grid[i]/self.Theta_e
        if np.abs(lgr) < 1.e-80: lgr = 1.e-80
        
        res = 1 / (C * np.exp(lgr) -1)

        return res



    def compute_delta_j_kompaneets(self):
        """
        Calculate delta_j for the Kompaneets kernel following Chang & Cooper 1970

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
                    fjplusone = self.f_e(j+1, C)
                    quad_c = self.Theta_e/self._delta_half_grid[j]*(fjplusone - fj) + fjplusone + fjplusone**2
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
                except ZeroDivisionError: self._delta_j[j] = 1/2.

        #print(self._delta_j)

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
        self._pre_factor_term_kompaneets = np.ones(len(self._grid))

        for j in range(self._n_grid_points):
            if self._n_current[j]< 0: self._n_current[j] = 0

        for j in range(self._n_grid_points - 1):
            #if self._n_current[j+1]/max(self._n_current) > 1.e-150:
            self._heating_term_kompaneets[j] = 1 + (1 - self._delta_j[j])*self._n_current[j+1] + self._delta_j[j]*self._n_current[j]
                #only n2 term:
                #self._heating_term[j] = (1 - self._delta_j[j])*self._n_current[j+1] + self._delta_j[j]*self._n_current[j]
                #self._heating_term[j] = 1
            self._heating_term_kompaneets[j] *= self._half_grid[j]**4
            #else: self._heating_term[j] = 0.

            self._dispersion_term_kompaneets[j] = self.Theta_e
            self._dispersion_term_kompaneets[j] *= self._half_grid[j]**4

        for j in range(self._n_grid_points):
            self._pre_factor_term_kompaneets[j] = self._n_e * sigma_t *c0/ self.grid[j]**2


    def construct_terms_kompaneets_extended_by_nu(self):
        """
        Constract the dispersion, heating and pre-factor terms of the Kompaneets equation following Chang & Cooper 1970
        """
        self._dispersion_term_kompaneets = np.zeros(len(self._grid)-1)
        self._heating_term_kompaneets = np.zeros(len(self._grid)-1)
        self._pre_factor_term_kompaneets = np.ones(len(self._grid))

        for j in range(self._n_grid_points):
            if self._n_current[j]< 0: self._n_current[j] = 0

        for j in range(self._n_grid_points - 1):
            #if self._n_current[j+1]/max(self._n_current) > 1.e-150:
            self._heating_term_kompaneets[j] = 1 + (1 - self._delta_j[j])*self._n_current[j+1] + self._delta_j[j]*self._n_current[j]
                #only n2 term:
                #self._heating_term[j] = (1 - self._delta_j[j])*self._n_current[j+1] + self._delta_j[j]*self._n_current[j]
                #self._heating_term[j] = 1
            self._heating_term_kompaneets[j] *= self._half_grid[j]**4* (1 + 7/10/self.Theta_e * self._half_grid[j]**2)
            #else: self._heating_term[j] = 0.

            self._dispersion_term_kompaneets[j] = self.Theta_e
            self._dispersion_term_kompaneets[j] *= self._half_grid[j]**4* (1 + 7/10/self.Theta_e*self._half_grid[j]**2)

        for j in range(self._n_grid_points):
            self._pre_factor_term_kompaneets[j] = self._n_e * sigma_t *c0/ self.grid[j]**2


    def construct_terms_kompaneets_extended_by_p(self):
        """
        Constract the dispersion, heating and pre-factor terms of the Kompaneets equation following Chang & Cooper 1970
        """
        self._dispersion_term_kompaneets = np.zeros(len(self._grid)-1)
        self._heating_term_kompaneets = np.zeros(len(self._grid)-1)
        self._pre_factor_term_kompaneets = np.ones(len(self._grid))

        for j in range(self._n_grid_points):
            if self._n_current[j]< 0: self._n_current[j] = 0

        for j in range(self._n_grid_points - 1):
            #if self._n_current[j+1]/max(self._n_current) > 1.e-150:
            self._heating_term_kompaneets[j] = 1 + (1 - self._delta_j[j])*self._n_current[j+1] + self._delta_j[j]*self._n_current[j]
                #only n2 term:
                #self._heating_term[j] = (1 - self._delta_j[j])*self._n_current[j+1] + self._delta_j[j]*self._n_current[j]
                #self._heating_term[j] = 1
            self._heating_term_kompaneets[j] *= self._half_grid[j]**4* (1 + 14/5*self._half_grid[j])
            #else: self._heating_term[j] = 0.

            self._dispersion_term_kompaneets[j] = self.Theta_e
            self._dispersion_term_kompaneets[j] *= self._half_grid[j]**4* (1 + 14/5*self._half_grid[j])

        for j in range(self._n_grid_points):
            self._pre_factor_term_kompaneets[j] = self._n_e * sigma_t *c0/ self.grid[j]**2

    def compute_delta_j(self):
        """
        Compute the delta_js as in original CC70, for a non-Kompaneets kernel.

        delta_j controls where the differences are computed. If there are no dispersion
        terms, then delta_j is zero
        """

        # set to zero. note delta_j[n] = 0 by default

        self._delta_j = np.zeros(self._n_grid_points - 1)

        for j in range(self._n_grid_points - 1):

            # if the dispersion term is 0 => delta_j = 0
            if self._dispersion_term[j] != 0:

                w = (
                    self._delta_half_grid[1:-1][j] * self._heating_term[j]
                ) / self._dispersion_term[j]

                # w asymptotically approaches 1/2, but we need to set it manually
                if w == 0:

                    self._delta_j[j] = 0.5

                # otherwise, we use appropriate bounds
                else:

                    self._delta_j[j] = (1.0 / w) - 1.0 / (np.exp(w) - 1.0)

        # precomoute 1- delta_j
        self._one_minus_delta_j = 1 - self._delta_j


    def compute_delta_j_mix(self):
        """
        Compute delta_j in old way, but also accounting for Kompaneets.

        delta_j controls where the differences are computed. If there are no dispersion
        terms, then delta_j is zero
        """

        # set to zero. note delta_j[n] = 0 by default

        self._delta_j = np.zeros(self._n_grid_points - 1)

        dispersion_term_combined  = self._dispersion_term +self._dispersion_term_kompaneets*sigma_t*self._n_e*c0/self._half_grid**2

        heating_term_combined = self._heating_term + self._heating_term_kompaneets*sigma_t*self._n_e*c0/self._half_grid**2

        for j in range(self._n_grid_points - 1):

            # if the dispersion term is 0 => delta_j = 0
            if dispersion_term[j] != 0:

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

    def _setup_vectors(self):
        """
        from the specified terms in the subclasses, setup the tridiagonal terms

        """

        # initialize everything to zero

        a = np.zeros(self._n_grid_points)
        b = np.zeros(self._n_grid_points)
        c = np.zeros(self._n_grid_points)

        # walk backwards in j starting from the second to last index
        # then set the end points
        for k in range(self._n_grid_points - 2, 1, -1):
            # pre compute one over the delta of the grid
            # this is the 1/delta_grid in front of the F_j +/- 1/2.

            one_over_delta_grid_forward = 1.0 / self._delta_half_grid[k + 1]
            one_over_delta_grid_backward = 1.0 / self._delta_half_grid[k]

            # this is the delta grid in front of the full equation

            one_over_delta_grid_bar = 1.0 / self._delta_grid[k]

            # The B_j +/- 1/2 from CC, Kompaneets part
            B_forward_kompaneets = self._heating_term_kompaneets[k]
            B_backward_kompaneets =self._heating_term_kompaneets[k - 1]

            # The C_j +/- 1/2 from CC, Kompaneets part
            C_forward_kompaneets = self._dispersion_term_kompaneets[k]
            C_backward_kompaneets = self._dispersion_term_kompaneets[k - 1]


            # The B_j +/- 1/2 from CC
            B_forward = self._heating_term[k]
            B_backward =self._heating_term[k - 1]

            # The C_j +/- 1/2 from CC
            C_forward = self._dispersion_term[k]
            C_backward = self._dispersion_term[k - 1]


            A_k = self._pre_factor_term_kompaneets[k]

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
                C_backward_kompaneets=C_backward_kompaneets,
                B_backward=B_backward,
                B_backward_kompaneets=B_backward_kompaneets,
                A = A_k, 
                delta_j_minus_one=self._delta_j[k - 1],
            )

            # n_j term
            b[k] = _compute_n_j(
                one_over_delta_grid=one_over_delta_grid_bar,
                one_over_delta_grid_bar_backward=one_over_delta_grid_backward,
                one_over_delta_grid_bar_forward=one_over_delta_grid_forward,
                C_backward=C_backward,
                C_backward_kompaneets=C_backward_kompaneets,
                C_forward=C_forward,
                C_forward_kompaneets=C_forward_kompaneets,
                B_backward=B_backward,
                B_backward_kompaneets=B_backward_kompaneets,
                B_forward=B_forward,
                B_forward_kompaneets=B_forward_kompaneets,
                A = A_k, 
                one_minus_delta_j_minus_one=self._one_minus_delta_j[k - 1],
                delta_j=self._delta_j[k],
            )

            # n_j+1 term
            c[k] = _compute_n_j_plus_one(
                one_over_delta_grid=one_over_delta_grid_bar,
                one_over_delta_grid_bar_forward=one_over_delta_grid_forward,
                C_forward=C_forward,
                C_forward_kompaneets=C_forward_kompaneets,
                B_forward=B_forward,
                B_forward_kompaneets=B_forward_kompaneets,
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
            C_backward=self._dispersion_term[-1],
            C_backward_kompaneets=self._dispersion_term_kompaneets[-1],
            B_backward=self._heating_term[-1],
            B_backward_kompaneets=self._heating_term_kompaneets[-1],
            A= self._pre_factor_term_kompaneets[-1], 
            delta_j_minus_one=self._delta_j[-1],
        )

        # n_j term
        b[-1] = _compute_n_j(
            one_over_delta_grid=one_over_delta_grid_bar,
            one_over_delta_grid_bar_backward=one_over_delta_grid_backward,
            one_over_delta_grid_bar_forward=one_over_delta_grid_forward,
            C_backward=self._dispersion_term[-1],
            C_backward_kompaneets=self._dispersion_term_kompaneets[-1],
            C_forward=0,
            C_forward_kompaneets=0,
            B_backward_kompaneets=self._heating_term_kompaneets[-1],
            B_backward =self._heating_term[-1],
            A= self._pre_factor_term_kompaneets[-1], 
            B_forward=0,
            B_forward_kompaneets=0,
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
            C_backward_kompaneets=0,
            C_backward=0,
            C_forward_kompaneets=self._dispersion_term_kompaneets[0],
            C_forward=self._dispersion_term[0],
            B_backward_kompaneets=0,
            B_backward=0,
            B_forward_kompaneets=self._heating_term_kompaneets[0],
            B_forward=self._heating_term[0],
            A= self._pre_factor_term_kompaneets[0],
            one_minus_delta_j_minus_one=0,
            delta_j=self._delta_j[0],
        )
        # n_j+1 term
        c[0] = _compute_n_j_plus_one(
            one_over_delta_grid=one_over_delta_grid_bar,
            one_over_delta_grid_bar_forward=one_over_delta_grid_forward,
            C_forward_kompaneets=self._dispersion_term_kompaneets[0],
            C_forward=self._dispersion_term[0],
            B_forward_kompaneets=self._heating_term_kompaneets[0],
            B_forward=self._heating_term[0],
            A= self._pre_factor_term_kompaneets[0],
            one_minus_delta_j=self._one_minus_delta_j[0],
        )

        # carry terms to the other side to form a tridiagonal equation
        # the escape term is added on but is zero unless created in
        # a child class
        a *= -self._delta_t
        b = (1 - b * self._delta_t) + self._escape_grid * self._delta_t
        c *= -self._delta_t

        if self.CN_solver: 
            a /= 2.
            b = (b-1)/2. +1
            c /= 2.

        # now make a tridiagonal_solver for these terms

        self._tridiagonal_solver = TridiagonalSolver(a, b, c)

    def add_source_terms(self, array):
        """Add an array to the source terms of the differential equation

        :param array: array to be added, length BIN_X, defined on grid 
        """

        self._source_grid += array

    def add_escape_terms(self, array):
        """Add an array to the escape terms of the differential equation

        :param array: array to be added, length BIN_X, defined on grid 
        """
        self._escape_grid += array

    def add_heating_terms(self, array):
        """Add an array to the heating terms of the differential equation

        :param array: array to be added, length BIN_X-1, defined on half_grid 
        """
        self._heating_term += array

    def add_diffusion_terms(self, array):
        """Add an array to the diffusion terms of the differential equation

        :param array: array to be added, length BIN_X-1, defined on half_grid 
        """
        self._diffusion_term += array

    def solve_time_step(self):
        """
        Solve for the next time step. Note that computation fo delta_j and the construction of the kompaneets terms need to be done externally!
        """

        # set up the right side of the tridiagonal equation.
        # This is the current distribution plus the source
        # unless it is zero

        #self._compute_delta_j_kompaneets()
        #self._delta_j_onehalf()
        #self._construct_terms_kompaneets()
        self._setup_vectors()


        d = self._n_current + self._source_grid * self._delta_t

        # Calculate the terms for the Crank-Nicolson solver, see Park & Petrosian for details
        if self.CN_solver: 
            a = self._tridiagonal_solver._a
            b = self._tridiagonal_solver._b
            c = self._tridiagonal_solver._c
            for k in range(self._n_grid_points-2, 1, -1):
                d[k] += self._n_current[k] -a[k]*self._n_current[k-1] -b[k]*self._n_current[k]-c[k]*self._n_current[k+1]
            d[0] += self._n_current[0]  -b[0]*self._n_current[0] -c[0]*self._n_current[1]
            d[-1] += self._n_current[-1] -a[-1]*self._n_current[-2] -b[-1]*self._n_current[-1]
        # set the new solution to the current one

        self._n_current = self._tridiagonal_solver.solve(d)


        # bump up the iteration number and the time

        self._iterate()

    def clear_arrays(self):
        """
        Clean the internal arrays of the PDE.

        """
        self._dispersion_term = np.zeros(len(self.grid)-1)
        self._dispersion_term_kompaneets = np.zeros(len(self.grid)-1)
        self._heating_term = np.zeros(len(self.grid)-1)
        self._heating_term_kompaneets = np.zeros(len(self.grid)-1)
        self._pre_factor_term_kompaneets = np.ones(len(self.grid))
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
        return self._heating_term

    @property
    def heating_term_kompaneets(self):
        return self._heating_term_kompaneets

    @property
    def dispersion_term(self):
        return self._dispersion_term

    @property
    def dispersion_term_kompaneets(self):
        return self._dispersion_term_kompaneets

    @property
    def pre_factor_term_kompaneets(self):
        return self._pre_factor_term_kompaneets

    def reset(self):
        """
        reset the solver (_n_current, _iterations, _current_time) to the distribution
        """

        self._n_current = self._initial_distribution
        self._iterations = 0
        self._current_time = 0.0


def _compute_n_j_plus_one(
    one_over_delta_grid,
    one_over_delta_grid_bar_forward,
    C_forward,
    C_forward_kompaneets,
    B_forward,
    B_forward_kompaneets,
    A, 
    one_minus_delta_j):
    """
    equation for the CC n_j +1 term

    :param one_over_delta_grid: the total change in energy
    :param one_over_delta_grid_bar_backward: the backward change in energy for the second derivative
    :param C_forward: the forward dispersion term
    :param C_forward_kompaneets: the forward dispersion term of the kompaneets kernel
    :param B_forward: the forward heating term
    :param B_forward_kompaneets: the forward heating term of the kompaneets kernel
    :param one_minus_delta_j: 1 - delta_j
    :param A: the 1/A(x) in front of the kompaneets kernel
    """

    return (A * one_over_delta_grid* (
        one_minus_delta_j * B_forward_kompaneets 
        + one_over_delta_grid_bar_forward * C_forward_kompaneets)
        + one_over_delta_grid* (
        one_minus_delta_j * B_forward 
        + one_over_delta_grid_bar_forward * C_forward))


def _compute_n_j(
    one_over_delta_grid,
    one_over_delta_grid_bar_backward,
    one_over_delta_grid_bar_forward,
    C_backward,
    C_backward_kompaneets,
    C_forward,
    C_forward_kompaneets,
    B_backward,
    B_backward_kompaneets,
    B_forward,
    B_forward_kompaneets,
    A, 
    one_minus_delta_j_minus_one,
    delta_j):
    """
    equation for the CC n_j term

    :param one_over_delta_grid: the total change in energy
    :param one_over_delta_grid_bar_backward: the backward change in energy for the second derivative
    :param one_over_delta_grid_bar_forward: the forward change in energy for the second derivative
    :param C_forward: the forward dispersion term
    :param C_forward: the forward dispersion term of the kompaneets kernel
    :param C_backward: the backward dispersion term
    :param C_backward: the backward dispersion term of the kompaneets kernel
    :param B_forward: the forward heating term
    :param B_forward: the forward heating term of the komaneets kernel
    :param B_backward: the backward heating term
    :param B_backward: the backward heating term of the kompaneets kernel
    :param A: the 1/A(x) in front of the kompaneets kernel
    :param one_minus_delta_j_minus_one: 1 - delta_j-1
    """

    return (-A * one_over_delta_grid* ( 
        one_over_delta_grid_bar_forward * C_forward_kompaneets 
        + one_over_delta_grid_bar_backward * C_backward_kompaneets 
        + one_minus_delta_j_minus_one * B_backward_kompaneets 
        - delta_j * B_forward_kompaneets)
        - one_over_delta_grid* (
        one_over_delta_grid_bar_forward * C_forward 
        + one_over_delta_grid_bar_backward * C_backward 
        + one_minus_delta_j_minus_one * B_backward \
        - delta_j * B_forward))



def _compute_n_j_minus_one_term(
    one_over_delta_grid,
    one_over_delta_grid_bar_backward,
    C_backward,
    C_backward_kompaneets,
    B_backward,
    B_backward_kompaneets,
    A,
    delta_j_minus_one):
    """
    equation for the CC n_j-1 term

    :param one_over_delta_grid: the total change in energy
    :param one_over_delta_grid_bar_forward: the forward change in energy for the second derivative
    :param C_backward: the backward dispersion term
    :param C_backward_kompaneets: the backward dispersion term of the kompaneets kernel
    :param B_backward: the backward heating term
    :param B_backward_kompaneets: the backward heating term of the kompaneets kernel
    :param one_minus_delta_j: 1 - delta_j
    :param A: the 1/A(x) in front of the kompaneets kernel
    """

    return (A * one_over_delta_grid* (
        one_over_delta_grid_bar_backward * C_backward_kompaneets 
        - delta_j_minus_one * B_backward_kompaneets)
        + one_over_delta_grid* ( 
        one_over_delta_grid_bar_backward * C_backward 
        -delta_j_minus_one * B_backward ))

