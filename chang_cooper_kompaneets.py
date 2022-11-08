import numpy as np

from tridiagonal_solver import TridiagonalSolver
from scipy import fsolve

def generate_subgrids_from_nonuniformgrid(grid):
    n_steps = len(grid) 

    half_grid = np.zeros(n_steps - 1)

    # now build the half grid points
    for i in range(n_steps-1):
        half_grid[i] = 0.5 * (grid[i+1] - grid[i])

    # and build the delta_grids
    # For the delta halfgrid, first fill the parts that truly lie in the array
    delta_half_grid = np.zeros(n_steps+1)
    for i in range(n_steps-1):
        i+=1
        delta_half_grid[i] = (grid[i+1] - grid[i])
    
    #then add to the boundaries: extend the grid as a ln grid

    first_step_log = np.log(grid[1]/grid[0])
    gridminusone = np.exp(np.log(grid[0])- first_step_log)
    delta_half_grid[0] = grid[1] - gridminusone

    last_step_log = np.log(grid[nsteps]/grid[nsteps-1])
    gridplusone = np.exp(np.log(grid[nsteps])+last_step_log)
    delta_half_grid[n_steps+1] = gridplusone - grid[nsteps]

    # delta grid from delta halfgrid

    delta_grid = np.zeros(n_steps)        
    for i in range(n_steps):
        delta_grid[i] = (grid[i+1] - grid[i])

    return grid, half_grid, delta_grid, delta_half_grid


class ChangCooper(object):
    def __init__(
        self,
        grid,
        delta_t=1.0,
        initial_distribution=None
    ):
        """
        Generic Chang and Cooper base class. Currently, the dispersion and heating terms
        are assumed to be time-independent

        :param n_grid_points: number of grid points on the x-axis
        :param max_grid: the maximum energy of the grid
        :param delta_t: the time step in the equation
        :param initial_distribution: an array of an initial electron distribution
        :param store_progress: store the history of the runs
        """
        self._grid = grid
        self._n_grid_points = len(grid)
        self._max_grid = np.max(grid)
        self._step = np.log(grid[1]) - np.log(grid[0])

        self._dispersion_term = np.zeros(len(grid))
        self._heating_term = np.zeros(len(grid))
        self._escape_grid = np.zeros(len(grid))
        self._source_grid = np.zeros(len(grid))

        self._delta_t = delta_t
        self._iterations = 0
        self._current_time = 0.0

        # first build the grid which is independent of the scheme
        self._build_grid()

        if initial_distribution is None:

            # initalize the grid of electrons
            self._n_current = np.zeros(self._n_grid_points)
            self._initial_distribution = np.zeros(n_grid_points)

        else:

            assert len(initial_distribution) == self._n_grid_points

            self._n_current = np.array(initial_distribution)
            self._initial_distribution = initial_distribution

        # compute the delta_js which control the upwind and downwind scheme
        self._compute_delta_j()

        # now compute the tridiagonal terms
        self._setup_vectors()

    def _build_grid(self):
        """
        setup the grid for the calculations and initialize the
        solution
        """

        # generate the other grids from the input grid

        self._grid, self._half_grid, self._delta_grid, self._delta_half_grid = generate_subgrids_from_nonuniformgrid(self._grid)
        self._grid2 = self._grid ** 2
        self._half_grid2 = self._half_grid ** 2


    def find_C_equilibrium_solution(self):

        self.N 
        return C

    def f_e(self, i, C):

        res = 1 / (C * np.exp(self._grid[i]/self.Theta_e) -1)

        return res



    def _compute_delta_j_kompaneets(self):

        C = self.find_C_equilibrium_solution()

        self._delta_j = np.zeros(self._n_grid_points - 1)

        for j in range(self._n_grid_points - 1):
            fj = self.f_e(j, C)
            fjplusone = self.f_e(j+1, C)
            quad_c = self.Theta_e/self._delta_half_grid[j]*(fjplusone - fj) + fjplusone + fjplusone**2
            quad_b = fj -fjplusone - 2* fjplusone**2 +2*fjplusone*fj
            quad_a = fjplusone**2 + fj**2

            root_1 = (-quad_b + np.sqrt(quad_b**2 - 4 * quad_a * quad_c) )/ (2*quad_a)

            if root_1 < 0:
                root_2 = (-quad_b - np.sqrt(quad_b**2 - 4 * quad_a * quad_c) )/ (2*quad_a)
                self._delta_j[j] = root_2
            else:
                self._delta_j[j] = root_1

    def _construct_terms_kompaneets(self):

        for j in range(self._n_grid_points - 1):
            self._heating_term[j] = 1 + (1 - self._delta_j[j])*self._n_current[j+1] + self._delta_j[j]*self._n_current[j]
            self._heating_term[j] *= self._half_grid[j]**4

            self._dispersion_term[j] = self.Theta_e/self._half_grid[j]
            self._dispersion_term[j] *= self._half_grid[j]**4


    def _compute_delta_j(self):
        """
        delta_j controls where the differences are computed. If there are no dispersion
        terms, then delta_j is zero
        """

        # set to zero. note delta_j[n] = 0 by default

        self._delta_j = np.zeros(self._n_grid_points - 1)

        for j in range(self._n_grid_points - 1):

            # if the dispersion term is 0 => delta_j = 0
            if self._dispersion_term[j] != 0:

                w = (
                    self._delta_grid[1:-1][j] * self._heating_term[j]
                ) / self._dispersion_term[j]

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
        for k in range(self._n_grid_points - 2, 0, -1):
            # pre compute one over the delta of the grid
            # this is the 1/delta_grid in front of the F_j +/- 1/2.

            one_over_delta_grid_forward = 1.0 / self._delta_half_grid[k + 1]
            one_over_delta_grid_backward = 1.0 / self._delta_half_grid[k]

            # this is the delta grid in front of the full equation

            one_over_delta_grid_bar = 1.0 / self._delta_grid[k]

            # The B_j +/- 1/2 from CC
            B_forward = self._heating_term[k]
            B_backward = self._heating_term[k - 1]

            # The C_j +/- 1/2 from CC
            C_forward = self._dispersion_term[k]
            C_backward = self._dispersion_term[k - 1]

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
                one_minus_delta_j_minus_one=self._one_minus_delta_j[k - 1],
                delta_j=self._delta_j[k],
            )

            # n_j+1 term
            c[k] = _compute_n_j_plus_one(
                one_over_delta_grid=one_over_delta_grid_bar,
                one_over_delta_grid_bar_forward=one_over_delta_grid_forward,
                C_forward=C_forward,
                B_forward=B_forward,
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
            B_backward=self._heating_term[-1],
            delta_j_minus_one=self._delta_j[-1],
        )

        # n_j term
        b[-1] = _compute_n_j(
            one_over_delta_grid=one_over_delta_grid_bar,
            one_over_delta_grid_bar_backward=one_over_delta_grid_backward,
            one_over_delta_grid_bar_forward=one_over_delta_grid_forward,
            C_backward=self._dispersion_term[-1],
            C_forward=0,
            B_backward=self._heating_term[-1],
            B_forward=0,
            one_minus_delta_j_minus_one=self._one_minus_delta_j[-1],
            delta_j=0,
        )

        # n_j+1 term
        c[-1] = 0

        ###############
        # left boundary
        # j-1/2 = 0

        one_over_delta_grid = 1.0 / (
            self._half_grid[0] - self._grid[0] / np.sqrt(self._step)
        )

        one_over_delta_grid_bar_forward = 1.0 / self._delta_grid[0]
        one_over_delta_grid_bar_backward = 0.0

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
            C_forward=self._dispersion_term[0],
            B_backward=0,
            B_forward=self._heating_term[0],
            one_minus_delta_j_minus_one=0,
            delta_j=self._delta_j[0],
        )
        # n_j+1 term
        c[0] = _compute_n_j_plus_one(
            one_over_delta_grid=one_over_delta_grid_bar,
            one_over_delta_grid_bar_forward=one_over_delta_grid_forward,
            C_forward=self._dispersion_term[0],
            B_forward=self._heating_term[0],
            one_minus_delta_j=self._one_minus_delta_j[0],
        )

        # carry terms to the other side to form a tridiagonal equation
        # the escape term is added on but is zero unless created in
        # a child class

        a *= -self._delta_t
        b = (1 - b * self._delta_t) + self._escape_grid * self._delta_t
        c *= -self._delta_t

        # now make a tridiagonal_solver for these terms

        self._tridiagonal_solver = TridiagonalSolver(a, b, c)

    def pass_source_terms(self, array):
        self._source_grid= array

    def pass_escape_terms(self, array):
        self.escape_grid= array

    def pass_heating_terms(self, array):
        self._heating_term = array

    def pass_diffusion_terms(self, array):
        self._dispersion_term = array

    def solve_time_step(self):
        """
        Solve for the next time step.
        """

        # set up the right side of the tridiagonal equation.
        # This is the current distribution plus the source
        # unless it is zero

        d = self._n_current + self._source_grid * self._delta_t

        # set the new solution to the current one

        self._n_current = self._tridiagonal_solver.solve(d)

        # clean any numerical diffusion if needed
        # this must be customized

        self._clean()

        # bump up the iteration number and the time

        self._iteratate()

    def _clean(self):

        pass

    def _iteratate(self):
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
    def history(self):
        """
        The history of the solution
        """

        return np.array(self._saved_grids)

    def reset(self):
        """
        reset the solver to the initial electron distribution

        :return:
        """

        self._n_current = self._initial_distribution
        self._iterations = 0
        self._current_time = 0.0


def _compute_n_j_plus_one(
    one_over_delta_grid,
    one_over_delta_grid_bar_forward,
    C_forward,
    B_forward,
    one_minus_delta_j,
):
    """
    equation for the CC n_j +1 term

    :param one_over_delta_grid: the total change in energy
    :param one_over_delta_grid_bar_backward: the backward change in energy for the second derivative
    :param C_forward: the forward dispersion term
    :param B_forward: the forward heating term
    :param one_minus_delta_j: 1 - delta_j
    """

    return one_over_delta_grid * (
        one_minus_delta_j * B_forward + one_over_delta_grid_bar_forward * C_forward
    )


def _compute_n_j(
    one_over_delta_grid,
    one_over_delta_grid_bar_backward,
    one_over_delta_grid_bar_forward,
    C_backward,
    C_forward,
    B_backward,
    B_forward,
    one_minus_delta_j_minus_one,
    delta_j,
):
    """
    equation for the CC n_j term

    :param one_over_delta_grid: the total change in energy
    :param one_over_delta_grid_bar_backward: the backward change in energy for the second derivative
    :param one_over_delta_grid_bar_forward: the forward change in energy for the second derivative
    :param C_forward: the forward dispersion term
    :param C_backward: the backward dispersion term
    :param B_forward: the forward heating term
    :param B_backward: the backward heating term
    :param one_minus_delta_j_minus_one: 1 - delta_j-1
    """

    return -one_over_delta_grid * (
        (
            one_over_delta_grid_bar_forward * C_forward
            + one_over_delta_grid_bar_backward * C_backward
        )
        + one_minus_delta_j_minus_one * B_backward
        - delta_j * B_forward
    )


def _compute_n_j_minus_one_term(
    one_over_delta_grid,
    one_over_delta_grid_bar_backward,
    C_backward,
    B_backward,
    delta_j_minus_one,
):
    """
    equation for the CC n_j-1 term

    :param one_over_delta_grid: the total change in energy
    :param one_over_delta_grid_bar_forward: the forward change in energy for the second derivative
    :param C_backward: the backward dispersion term
    :param B_backward: the backward heating term
    :param one_minus_delta_j: 1 - delta_j
    """

    return one_over_delta_grid * (
        one_over_delta_grid_bar_backward * C_backward - delta_j_minus_one * B_backward
    )
