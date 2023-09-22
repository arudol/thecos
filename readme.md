# Welcome to the THErmal COmptonisation Software (THECOS)!

THECOS is designed to compute the time evolution of photon spectra from a thermal population of leptons. It comes with modules accounting for emission/absorption terms thermal bremsstrahlung, cyclotron and double Compton emission, as well as adiabatic cooling effects. Compton scatterings between thermal electrons and the photons are included through the Kompaneets equation. 

*Requirements*: Python 3.x
*Quick start*: 

**Maths principle**

To evolve the photon distribution in time, 

THECOS is designed to solve the differential equation in momentum space

$$ \frac{\partial f(x)}{\partial t}  = \frac{1}{x^2}\frac{\partial}{\partial x}\left[   x^4 \sigma_\mathrm{T} n_\mathrm{e} c \left(\theta_e \frac{\partial f(x)}{\partial x} + f(x)^2 + f(x) \right)  -  x^2 a_\mathrm{exp} (x)  f(x) \right]
    + \epsilon(x) - \tau^{-1} (x) f (x) $$

Here 

$x^4 \sigma_\mathrm{T} n_\mathrm{e} c \left(\theta_e \frac{\partial f(x)}{\partial x} + f(x)^2 + f(x) \right)$ is the well-known *Kompaneets kernel* accounting for Compton scatterings with a thermal electron population defined through their number density $n_e$ and dimensionless energy $\theta_e = k_b T_e /m_e c^2$.

**Code design**

**Extending the code**
Built up modularly, beyond the standard options the code can easily be extended: 
1. Arbitrary injection/sink terms may be added to the photon PDE simply accessing sim.add_source_terms() 
2. In principle the solver and wrappers can also be used for other particle species. 

