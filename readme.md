# Welcome to the thermal compton scattering code!

This software is designed to compute the time evolution of photon spectra from a thermal population of leptons. It includes emission/absorption terms thermal bremsstrahlung, cyclotron and double Compton emission, as well as adiabatic cooling effects. Compton scatterings between the thermal population and the photons are included through the Kompaneets equation. 

*Requirements*: Python 3.x
*Quick start*: 

**Maths principle**

$$ \frac{\partial f(x)}{\partial t}  = \frac{1}{x^2}\frac{\partial}{\partial x}\left[   x^4 \sigma_\mrm{T} n_\mrm{e} c \left(\theta_e \frac{\partial f(x)}{\partial x} + f(x)^2 + f(x) \right)  -  x^2 a_\mrm{exp} (x)  f(x) \right]
    + \epsilon(x) - \tau^{-1} (x) f (x) $$


**Extending the code**
Arbitrary injection/sink terms may be added to the photon PDE.

