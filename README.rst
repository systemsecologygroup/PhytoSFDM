PhytoSFDM: Phytoplankton Size and Functional Diversity Model
============================================================

PhytoSFDM is a modelling framework for studying the size structure and the 
functional diversity of marine phytoplankton communities. The provided software 
consist of three modules that allows the user to calculate aggregate community 
properties of phytoplankton in a 0 dimensional physical setting at any particular 
location of the world’s ocean.

The structure of all model variants are based on the familiar NPZD form as in 
the seminal work of Fasham (1990), where ordinary differential equations for 
nutrients, phytoplankton, zooplankton, and detritus trace the fluxes between 
these state variables. Here we extended this traditional modelling structure 
by characterising the phytoplankton community with a trait (i.e. cell size) 
and a trade-off emerging from three allometric relationships between, cell 
size and: 1) phytoplankton nutrient uptake, 2) zooplankton grazing and 3) 
phytoplankton sinking. In these models the size structure and size diversity 
of the phytoplankton community is modelled by explicitly quantifying a 
finite number of phytoplankton morphotypes, each type with a specific size 
value (i.e. full model); or by approximating the size distribution using a 
moment closure technique, where we only quantified the total biomass, the mean 
size and the size variance of the community (i.e. aggregate models). These 
approaches are inspired by early works of Wirtz & Eckhardt (1996), Norberg et al.
(2001), Bruggeman & Koojiman (2007), Bruggeman (2009) and Merico et al. (2009). 
Some examples of more recent applications of the moment-based approximation are 
Wirtz (2013), Wirtz & Sommer (2013), Terseleer et al. (2014), and Acevedo-Trejos et al.
(2015).  

The three main modules of the package are: example, sizemodels, and envforcing.
The example module is the entry point of the package, which computes and compares
the two main model structures (full and aggregate) and the four variance 
treatments (Unsustained, Fixed, Immigration and Trait Diffusion) at a testing
location in the north Atlantic Ocean. The module sizemodels contains a single
class with methods to quantify the phytoplankton size structure and their 
functional diversity. Also within this class we provide methods to: a) modify 
the default parameters, b) symbolically solve the derivatives of the fitness 
function with respect to the trait, and c) log-transform the mean trait and
the trait variance. The last module envforcing consist of one class with methods
to extract a spatially averaged forcing data provided in the NetCDF files. The 
climatological data is at a monthly resolution, thus, a method to interpolate to
daily time step is also included in this module. 

How to Install
--------------

We assume the user have a running version of Python 2.7.x and have permissions
to write in the folder where the python distribution is installed. Still the 
package have not been tested in Python 3.x, but further developments of the
package will be compatible to newer versions of Python. To install it the user
would require the latest versions of pip and setuptools. Additional dependencies
are: matplotlib (version 1.4.3 or greater), numpy (version 1.9.2 or greater), 
scipy (version 0.15.1 or greater) and sympy (version 0.7.6.1 or greater).

To install the package using pip just type in a terminal (Unix like systems) 
or in a command prompt window (Windows systems):

$ pip install PhytoSFDM

To install the package from the tarball, just download the file from GitHub. 
Then untar and unzip the file with a specific software like WinRAR (in Windows) 
or type in a terminal (Unix like systems): 

$ tar xvfz PhytoSFDM-X.X.X.tar.gz

where the Xs are the respective version of the package. Then inside the extracted
folder "PhytoSFDM", type the following command:

$ python setup.py install

If you do not have permission to write in the python distribution folder then
use command sudo before the suggested installation lines (Unix like systems).

Usage
-----

The example calculates all the model variants at a testing location in the north
Atlantic Ocean (47.5° N and 15.5° W). To run the example just type in a terminal:

$ PhytoSFDM_example

or alternatively in an interactive python console you can import the example and 
run it by typing:

>>> import phytosfdm.Example.example as exmp
>>> exmp.main() 

To calculate one of the five model variants (Full, Immigration, Trait Diffusion, 
Fix Variance and Unsustained Variance) at a specific set of coordinates, one
can import the required library in an interactive python console as:

>>> from phytosfdm.SizeModels.sizemodels import SM
>>> Lat= 47.5
>>> Lon= 344.5
>>> RBB= 2.5
>>> SM1=SM(Lat,Lon,RBB,"Imm")

where SM is the class that contains all the methods to calculate a specific size
model, Lat and Lon are Latitude and Longitude (notice that Lon should be written 
in a scale from 0° to 360°), RBB is the range of the bounding box (in degrees) 
for averaging the environmental forcing variables and SM1 is an object that 
contains the results of the size model with an immigration treatment. After 
execution the results of the model can be accessed by:

>>> SM1.outvariables

In the multidimensional array "SM1.outvariables" the first dimension is time (e.g. 
3650 days if the model is run with default parameters) and the second dimension 
contains the state variables, for the full model, or the state variables and the 
dummy variables for the aggregate models. In the case of SM1, i.e. a 
size model of aggregate properties based on an immigration mechanism, 


To access all attribute values of the class instance "SM1" one can type in an 
interactive python console:

>>> SM1.__dict__

To modify the default parameter values, for example, the user can call a new class
instance with a tuple list with the parameter name and its new value:

>>> SM2.SM(Lat,Lon,RBB,"Imm",defaultParams=False,ListParams=[("timeyears",5),("muP",1.5])

Please refer to the documentation inside of the class and its methods
for further details.

Acknowledgements
----------------
I would like to thank Jorn Bruggeman for his valuable contribution to an 
earlier version of the size-based model and my colleagues, Gunnar Brandt,
S. Lan Smith and Agostino Merico for their continuous support and encouragement
to complete this project.

References
----------

Acevedo-Trejos, E., Brandt, G., Bruggeman, J. & Merico, A. Mechanisms shaping phytoplankton community structure and diversity in the ocean. Sci. Rep. 5, 8918 (2015).

Bruggeman, J. & Kooijman, S. A. L. M. A biodiversity-inspired approach to aquatic ecosystem modeling. Limnol. Oceanogr. 52, 1533–1544 (2007).

Bruggeman, J. Succession in plankton communities: A trait-based perspective. (2009).

Fasham, M., Ducklow, H. W. & Mckelvie, S. M. A nitrogen-based model of plankton dynamics in the oceanic mixed layer. J. Mar. Res. 48, 591–639 (1990).

Merico, A., Bruggeman, J. & Wirtz, K. A trait-based approach for downscaling complexity in plankton ecosystem models. Ecol. Modell. 220, 3001–3010 (2009).

Norberg, J. et al. Phenotypic diversity and ecosystem functioning in changing environments: a theoretical framework. Proc. Natl. Acad. Sci. 98, 11376–81 (2001).

Terseleer, N., Bruggeman, J., Lancelot, C. & Gypens, N. Trait-based representation of diatom functional diversity in a plankton functional type model of the eutrophied Southern North Sea. Limnol. Oceanogr. 59, 1–16 (2014).

Wirtz, K. W. Mechanistic origins of variability in phytoplankton dynamics: Part I: niche formation revealed by a size-based model. Mar. Biol. 160, 2319–2335 (2013).

Wirtz, K. W. & Sommer, U. Mechanistic origins of variability in phytoplankton dynamics. Part II: analysis of mesocosm blooms under climate change scenarios. Mar. Biol. 160, 2503–2516 (2013).

Wirtz, K. W. & Eckhardt, B. Effective variables in ecosystem models with an application to phytoplankton succession. Ecol. Modell. 92, 33–53 (1996).


