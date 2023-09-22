import numpy as np
import math
from tridiagonal_solver import TridiagonalSolver
from mpmath import *
from consts import *
from copy import deepcopy
from scipy.integrate import trapz, simps
from scipy.interpolate import CubicSpline

def generate_subgrids_from_nonuniformgrid(grid, type_grid):
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
        :param source_parameters: source parameters containing the electron number density and dimensionless temperature
        :param delta_t: the time step in the equation, default is 1
        :param initial_distribution: an array of an initial electron distribution, default is none
        :param N: total number of photons, default is 0
        :param type_grid: set to logarithmic ('log') or linear ('lin'), default is 'log'
        :param CN_solver: Switch to use Crank-Nicolson solver, default is False
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
        self._CFL = 1.e-4
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
        return self.N/ 8 * np.pi /(c0*h)**3*(m_e*c0**2)**3

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

        last_guess = self._Theta_e**3 *  prefactor/self._N_internal_units

        #def sum_n(x, N):
        #    return sum(x**i/((N+1)-N*i)**3 for i in range(0, N))
        quad_c = -1/(2**3)
        quad_b = -1
        quad_a = self._N_internal_units / self._Theta_e**3 /  prefactor

        current_guess =  (-quad_b + np.sqrt(quad_b**2 - 4 * quad_a * quad_c) )/ (2*quad_a)
        mp.dps = 30; mp.pretty = True
        N = 3
        while np.abs((last_guess-current_guess)/current_guess) > 0.01:
            f = lambda x: self._N_internal_units/(prefactor*self._Theta_e**3)*x**N - sum(x**i/((N+1)-N*i)**3 for i in range(0, N))
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
        #quad_a = self._N_internal_units / self._Theta_e**3 /  prefactor

        #first_guess = (-quad_b + np.sqrt(quad_b**2 - 4 * quad_a * quad_c) )/ (2*quad_a)

        #f = lambda x: self._N_internal_units/(prefactor*self._Theta_e**3)*x**6 - \
        #                   (x**5 + 1/2**3 * x**4 + 1/3**3 *x**3 + 1/ 4**3 *x**2 + 1/ 5**3 *x + 1/6**3 ) 

        #mp.dps = 30; mp.pretty = True

        #first_guess = self._Theta_e**3 *  prefactor/self._N_internal_units

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

        lgr = self._grid[i]/self._Theta_e
        if np.abs(lgr) < 1.e-80: lgr = 1.e-80
        
        res = 1 / (C * np.exp(lgr) -1)

        return res

    def _initialise_splines(self):

        heating_term_combined = self._heating_term + self._heating_term_kompaneets

        heating_term_combined_spline = CubicSpline(self._half_grid, heating_term_combined)

        self._heating_term_combined_fullgrid = heating_term_combined_spline(self._grid)

        for i in range(len(self._heating_term_combined_fullgrid)):
            if np.abs(self._heating_term_combined_fullgrid[i] == 0): self._heating_term_combined_fullgrid[i] = -1.e-100

        ivA = 1.0/self._heating_term_combined_fullgrid

        a = self._escape_grid/self._pre_factor_term * ivA
        self._spline_fn_mivA = CubicSpline(self._grid, ivA)
        self._spline_fn_n =CubicSpline(self._grid, self._n_current)
        self._spline_fn_a = CubicSpline(self._grid, a)

        self._spline_fn_q = CubicSpline(self._grid, self._source_grid/self._pre_factor_term)

    def _compute_boundary(self):
        """ Compute the boundary where :math:'\dot(x) \delta T / \delta x >  0.01' , here the cooling
                term :math:'\dot(x)' is the one in the 1/x2 parenthesis, accounting for Kompaneets and other contributions.
         """

        heating_term_combined = self._heating_term + self._heating_term_kompaneets

        heating_term_combined_spline = CubicSpline(self._half_grid, heating_term_combined)

        heating_term_combined_fullgrid = heating_term_combined_spline(self._grid)

        heating_term_combined_fullgrid *= self._pre_factor_term

        index_shifted_by_one_grid = 3
        i = index_shifted_by_one_grid

        while (heating_term_combined_fullgrid[i]/self._delta_grid[i] * self._delta_t< self._CFL and i<self._n_grid_points-2):
            i+=1
            index_shifted_by_one_grid = i

        boundary = min(index_shifted_by_one_grid, self._n_grid_points)

        return boundary


    def _compute_y(self):
        """ For each gridpoint, compute the y parameter for the analytical solution. EQ C93 of Gao et al 2017 """

        b_lin = np.zeros(self._n_grid_points)
        b_lin[0] = 0.0

        for i in range(self._n_grid_points-1):
            i+=1
            delta_b = self._rk4_gslspl_b(self._grid[i-1],self._grid[i],10)
            set_delta_b_min = b_lin[i-1] * 1.0e-15

            delta_b = max(set_delta_b_min, delta_b)
            b_lin[i] = b_lin[i-1] + delta_b;

            if not (b_lin[i] > b_lin[i-1]):
                while not (b_lin[i] > b_lin[i-1]):
                    direction = max(b_lin[i-1]+ 1.0, b_lin[i-1]*1.0) 
                    nextvalue = float(math.nextafter(b_lin[i-1], direction))
                    b_lin[i] = nextvalue

        self._spline_fn_ivB = CubicSpline(b_lin, self._grid) # mind the order : (b_lin, x_ln)

        b_max = b_lin[self._n_grid_points-1]
        x_max = self._grid[self._n_grid_points-1]

        self._y_ln= np.zeros(self._n_grid_points)

        for i in range(self._n_grid_points):
            self._y_ln[i] = x_max-1.0e-3*(self._delta_grid[i]) # default value set near the right border of x_grid

        for i in range(self._n_grid_points-1):
            if(b_lin[i]+self._delta_t<b_max):
                self._y_ln[i] = self._spline_fn_ivB(b_lin[i]+self._delta_t)


    def _rk4_gslspl_b(self, lower, upper, N_INTG): # related to function B(x)
        """ Helper function to perform the integral over 1/A in the analytical solver for computing y (EQ C93 of Gao et al 2017 )
                Uses Runge-Kutta 4th order method. """

        x = lower # to avoid touching lower boundary of interpolation.
        y = 0.0

        diff_x = (upper-lower)/N_INTG
        x_min = self._grid[0]
        x_max = self._grid[self._n_grid_points-1]

        for i in range(N_INTG):

            if(x<x_min or x+1.0*diff_x>x_max):
                pass
            else:
                #k1 = diff_x * np.exp(self._spline_fn_lnmivA(x))
                k1 = diff_x * self._spline_fn_mivA(x)
                #k2 = diff_x * np.exp(self._spline_fn_lnmivA( x+0.5*diff_x ))
                k2 = diff_x * self._spline_fn_mivA( x+0.5*diff_x )
                #k4 = diff_x * np.exp(self._spline_fn_lnmivA( x+1.0*diff_x ))
                k4 = diff_x * self._spline_fn_mivA( x+1.0*diff_x )

                y += (1.0/6.0)*(k1+k4)+(2.0/3.0)*k2
                x += diff_x
        return y


    def _rk4_gslspl_a(self, lower, upper, N_INTG): # inner integration result, not the exp(\int)
        """ Helper function to compute the integral over alpha/A  in the analytical solver for EQ C92 in Gao et al 2017. 
                Uses Runge-Kutta 4th order integration method."""

        if(lower<self._grid[0] or upper>self._grid[self._n_grid_points-1]):
            print("in _rk4_gslspl_a, ", upper, " > ", self._grid[self._n_grid_points-1])
            raise Exception("in function _rk4_gslspl_a, integration range out of bound")

        diff_x = (upper-lower)/N_INTG

        x = lower
        y = 0.0

        for i in range(N_INTG):
            k1_lnloss =self._spline_fn_a(x)
            k2_lnloss =self._spline_fn_a( x+0.5*diff_x)
            k4_lnloss =self._spline_fn_a( x+1.0*diff_x)

            k1_loss = max(k1_lnloss, 1.e-200)
            k2_loss = max(k2_lnloss, 1.e-200)
            k4_loss = max(k4_lnloss, 1.e-200)

            k1 = -diff_x * k1_loss
            k2 = -diff_x * k2_loss
            k4 = -diff_x * k4_loss
            y += (1.0/6.0)*(k1+k4)+(2.0/3.0)*k2
            x += diff_x

        return y


    def _rk4_gslspl_q(self, lower, upper, N_INTG_OUTER_PER_BIN):
        """ Compute the double integral for the analytical solution for EQ C92 in Gao et al 2017. 
                Uses Runge-Kutta 4th order integration method."""

        if(lower<self._grid[0] or upper>self._grid[self._n_grid_points-1]):
            raise Exception("in function _rk4_gslspl_q, integration range out of bound")

        if(lower>=upper):
            raise Exception("in function _rk4_gslspl_q, integration bound lower >= upper occurred ")

        y = 0.0

        abs_alpha_ivA = self._spline_fn_a(lower)

        I_alpha_ivA = (upper-lower) * abs_alpha_ivA

        if(I_alpha_ivA<1.0e-3):
            #if alpha is ~ 0.0, inner integral ~ 0.0; perform outer integration only (single layer)
            #adjusting N_intg_points  

            N_min = 10
            N_eff = int((upper-lower)/0.1)
            N_pts = max(N_min, N_eff)

            diff_x = (upper-lower)/N_pts
            x_prime = lower

            for i in range(N_pts):   
                k1 = diff_x * self._spline_fn_q(x_prime)
                k2 = diff_x * self._spline_fn_q(x_prime+0.5*diff_x)
                k4 = diff_x * self._spline_fn_q( x_prime+1.0*diff_x)

                y += (1.0/6.0)*(k1+k4)+(2.0/3.0)*k2
                x_prime += diff_x
            y *= self._spline_fn_mivA(lower)
        else:
            #search for effective upper boundary of the integration

            upper_eff = upper
            set_range = upper-lower
            N_OUTER_INIT = 10
            N_INNER_INIT = 10

            diff_x = set_range/N_OUTER_INIT

            if (diff_x/lower < 1.0e-10):
                print("lower bound = ", lower)
                print("upper bound = ", upper_eff)
                raise Exception("in function _rk4_gslspl_q, integration range too narrow")

            stat_reduced_range = False

            while(self._rk4_gslspl_a(lower, lower+diff_x, N_INNER_INIT) < -2.0 ):
                upper_eff -= 0.5 * set_range
                set_range = upper_eff - lower
                diff_x = set_range/N_OUTER_INIT
                stat_reduced_range = True

            while( self._rk4_gslspl_a(lower, lower+diff_x, N_INNER_INIT) > -0.8 and stat_reduced_range ):
                upper_eff += 0.5 * set_range

            # adjusting N_intg_points  

            N_min = 10
            N_eff = int((upper_eff-lower)/0.1)
            N_pts = max(N_min, N_eff)

            diff_x = (upper_eff-lower)/N_pts

            x_prime = lower

            for i in range(N_pts):
                k1_lnloss = self._rk4_gslspl_a(lower, x_prime,            N_INNER_INIT)
                k2_lnloss = self._rk4_gslspl_a(lower, x_prime+0.5*diff_x, N_INNER_INIT)
                k4_lnloss = self._rk4_gslspl_a(lower, x_prime+1.0*diff_x, N_INNER_INIT)

            if k1_lnloss < -300:
                k1_loss = 0.0
            else:
                k1_loss = np.exp(k1_lnloss)

            if k2_lnloss < -300:
                k2_loss = 0.0
            else:
                k2_loss = np.exp(k2_lnloss)

            if k4_lnloss < -300:
                k4_loss = 0.0
            else:
                k4_loss = np.exp(k4_lnloss)


                k1 = diff_x * self._spline_fn_q(x_prime)* k1_loss
                k2 = diff_x * self._spline_fn_q(x_prime+0.5*diff_x)* k2_loss
                k4 = diff_x * self._spline_fn_q( x_prime+1.0*diff_x)* k4_loss

                y += (1.0/6.0)*(k1+k4)+(2.0/3.0)*k2
                x_prime += diff_x

            y *= self._spline_fn_mivA(lower)

        return y

    def _compute_n1(self):
        """ Compute the distribution at the next timestep from the analytical solver """
        self._HighE = np.zeros(self._n_grid_points)

        for i in range(self._n_grid_points-1):

            if( self._y_ln[i]-self._grid[i] < 1.0e-4 ): #use analytical asymptotic solution if energy loss rate is low

                if self._escape_grid[i]*self._delta_t > 1.e-4:
                    effective_duration = (np.exp(-self._escape_grid[i]*self._delta_t)-1.0)/self._escape_grid[i]

                else: 
                    effective_duration = -self._delta_t
                self._HighE[i] = self._n_current[i] * np.exp(-self._escape_grid[i]*self._delta_t) - self._source_grid[i]* effective_duration
            else: #// numerical integration
                ny = self._spline_fn_n(self._y_ln[i])
                ivAx = -self._spline_fn_mivA(self._grid[i])
                ivAy = -self._spline_fn_mivA(self._y_ln[i])

                Aratio = ivAx/ivAy;

                lnloss = self._rk4_gslspl_a(self._grid[i],self._y_ln[i],20)

                if lnloss < -100:
                    loss = 0.0
                else:
                    loss = np.exp(lnloss)

                outer_intg_points_per_bin = 2.0

                injection = self._rk4_gslspl_q(self._grid[i], self._y_ln[i], outer_intg_points_per_bin)

                self._HighE[i] = ny * Aratio * loss + injection
        
        self._HighE[self._n_grid_points-1] = 0.0 # fix the boundary condition



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
                except ZeroDivisionError: self._delta_j[j] = 1/2.

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
        Constract the dispersion, heating and pre-factor terms of the Kompaneets equation following Chang & Cooper 1970
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
        Constract the dispersion, heating and pre-factor terms of the Kompaneets equation following Chang & Cooper 1970
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

    def _setup_vectors(self):
        """
        from the specified terms in the subclasses, setup the tridiagonal terms

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
        """Add an array to the source terms of the differential equation

        :param array: array to be added, length BIN_X, defined on grid 
        """

        self._source_grid += array

    def add_escape_terms(self, array):
        """Add an array to the escape terms of the differential equation

        :param array: array to be added, length BIN_X, defined on grid 
        """
        self._escape_grid += array

    def add_heating_term(self, array):
        """Add an array to the heating terms of the differential equation

        :param array: array to be added, length BIN_X-1, defined on half_grid 
        """
        self._heating_term += array

    def add_diffusion_terms(self, array):
        """Add an array to the diffusion terms of the differential equation

        :param array: array to be added, length BIN_X-1, defined on half_grid 
        """
        self._diffusion_term += array

    def set_internal_photonarray(self, array):
        """ Set the _n_current to a given array
        :param: array of length BIN_X
        """
        self._n_current = deepcopy(array)

    def update_timestep(self, delta_t):
        """ update the delta t to given value
        :param: delta_t [s] new timestep
        """

        self._delta_t = delta_t

    def solve_time_step(self, solver = 'matrix'):
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

        if solver == 'matrix':
            self._tridiagonal_solver = TridiagonalSolver(a, b, c)
            self._n_current = self._tridiagonal_solver.solve(d)

        elif solver == 'hybrid':
            self._initialise_splines()
            boundary = self._compute_boundary()

            self._compute_y()
            self._compute_n1()

            d[boundary -2 ] += self._heating_term_combined_fullgrid[boundary-1]*self._pre_factor_term[boundary-1] * self._HighE[boundary-1] / self._delta_grid[boundary-1] * self._delta_t

            self._tridiagonal_solver = TridiagonalSolver(a[:boundary-2], b[:boundary-2], c[:boundary-2])
            lowE = self._tridiagonal_solver.solve(d[:boundary-2])

            for i in range(self._n_grid_points):
                if i < boundary-2 :
                    self._n_current[i] = lowE[i]
                else:
                    self._n_current[i] = self._HighE[i]
        elif solver == 'analytic':
            
            self._initialise_splines()
            boundary = self._compute_boundary()

            self._compute_y()
            self._compute_n1()
            self._n_current = deepcopy(self._HighE)
        else:
            raise Exception("No valid solver type specified")

        # bump up the iteration number and the time

        self._iterate()

    def clear_arrays(self):
        """
        Clean the internal arrays of the PDE.

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
    def pre_factor_term(self):
        return self._pre_factor_term

    @property
    def escape_term(self):
        return self._escape_grid

    @property
    def source_term(self):
        return self._source_grid

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
    B_forward,
    A, 
    one_minus_delta_j):
    """
    equation for the CC n_j +1 term

    :param one_over_delta_grid: the total change in energy
    :param one_over_delta_grid_bar_backward: the backward change in energy for the second derivative
    :param C_forward: the forward dispersion term
    :param B_forward: the forward heating term
    :param one_minus_delta_j: 1 - delta_j
    :param A: the 1/A(x) in front of the kompaneets kernel
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

    :param one_over_delta_grid: the total change in energy
    :param one_over_delta_grid_bar_backward: the backward change in energy for the second derivative
    :param one_over_delta_grid_bar_forward: the forward change in energy for the second derivative
    :param C_forward: the forward dispersion term
    :param C_backward: the backward dispersion term
    :param B_forward: the forward heating term
    :param B_backward: the backward heating term
    :param A: the 1/A(x) in front of dispersion and advection term
    :param one_minus_delta_j_minus_one: 1 - delta_j-1
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

    :param one_over_delta_grid: the total change in energy
    :param one_over_delta_grid_bar_forward: the forward change in energy for the second derivative
    :param C_backward: the backward dispersion term
    :param B_backward: the backward heating term
    :param one_minus_delta_j: 1 - delta_j
    :param A: the 1/A(x) in front of the kompaneets kernel
    """

    return (A * one_over_delta_grid* (
        one_over_delta_grid_bar_backward * C_backward 
        -delta_j_minus_one * B_backward ))

