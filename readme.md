# Welcome to the THErmal COmptonisation Software (THECOS)!

THECOS is designed to compute the time evolution of photon spectra from a thermal population of leptons. It comes with modules accounting for emission/absorption terms thermal bremsstrahlung, cyclotron and double Compton, as well as adiabatic cooling effects. Compton scatterings between thermal electrons and the photons are included through the Kompaneets equation. 

*Requirements*: Python 3.x [numpy, numba, scipy, copy]

*Quick start*: No installation is required. After downloading the sources files, open one of the jupyter notebooks contained in examples_and_tests/ to see how it works! 

## Background

THECOS is designed evolve the photon occupation number $f$ as a function of dimensionless energy $x=E_\gamma/m_e c^2$ in time:

$$ \frac{\partial f(x)}{\partial t}  = \frac{1}{x^2}\frac{\partial}{\partial x}\left[   x^4 \sigma_\mathrm{T} n_\mathrm{e} c \left(\theta_e \frac{\partial f(x)}{\partial x} + f(x)^2 + f(x) \right)  -  x^2 a (x)  f(x) \right]
    + \epsilon(x) - \tau^{-1} (x) f (x) $$

Here 
$x^4 \sigma_\mathrm{T} n_\mathrm{e} c \left(\theta_e \frac{\partial f(x)}{\partial x} + f(x)^2 + f(x) \right)$ 
is the well-known *Kompaneets kernel* \[Kompaneets 1957\] accounting for Compton scatterings with a thermal electron population defined through their number density $n_e$ and dimensionless energy $\theta_e = k_b T_e /m_e c^2$.
The additional termas account for cooling $\left[a(x)\right]$, source/injection $\left[\epsilon(x)\right]$ and sink/escape $\left[\tau^{-1}(x)\right]$. Currently implemented are thermal Bremsstrahlung, cyclotron and double comption emission/absorption, as well as adiabatic cooling.

Modules and functions contributing to the cooling/source/sink arrays can be easily be added/removed; The inclusion of the kompaneets kernel is also optional. 

## Code design

The software consists of four main blocks:

* The SimulationManager contained in core.py handles a single simulation
* It makes use of the solver contained in solver.py (which in turn uses the matrix solver in tridiagonal_solver.py)
* Physics processes are contained in separate modules (e.g. adiabatic.py, double_compton.py). They are added to the SimulationManager and have a pre-defined structure with integration hooks for the SimulationManager.
* For convenience, constants and some simple functions are collected in consts.py

## Extending the code
Built up modularly, the code can easily be extended: 
1. Arbitrary injection/sink terms may be added during runtime by accessing `sim.add_source_terms()` / `sim.add_escape_terms()` 
2. Similarly, users can define new modules. They should follow the structure laid out in 'dummy.py' in order to use the integration hooks to the SimulationManater correctly: The basic idea of the code is that all radiation modules are built with the same structure, offering the same hooks to the SimulationManater who calls them for each module currently attached to it.
3. The solver and simulation may also be used for other particle species. If the equation should be treated in energy instead of momentum space, just activate the corresponding flag and the pre-factor $1/x^2 \rightarrow 1$. Then, if the Kompaneets Kernel is also de-activated, it can just be used as a simple Chang \& Cooper code.

