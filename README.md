# Structure and Dynamics in Glasses - a Machine Learning approach 

**Investigation of relevant ones among many types of structure descriptors for determining the mobility of particles in a supercooled liquid** using supervised machine learning techniques.

It showcases also how to gain insighgts of a pratical problem with a data driven approach by combining different machine learning models of different properties.

## Problem Context 
We are dealing with a problem in an active area of current research in statistical physics of glassy materials, such as SiO2. A type of material is glassy when the crystallisation (or more generally called the first order transition), such as water becoming ice, is avoided, bypassed or absent with a changing of its environment, usually a dropping temperature. Instead, these materials vitrify, that is they behave like a liquid but becoming more and more viscous with a decreasing temperature until at some point, they apparently stop flowing and stand there like a solid, which we call glasses. This apparent transition from a liquid to a "solid", allegedly call the "glass transition", is unconventional compared to crystallisation where the microscopic structure changes qualitatively from a disordered one of liquids to an ordered structure of crystals, like FCC, BCC, etc. In a glass transition, the microscopic structure is always disordered seemingly unchanged from a liquid, except that molecules have more and more difficulties to move under thermal aggitation, giving rise to its macroscopic drastic increase in viscosity, so large they apperently stop flowing. Physicists havn't come up with a proper explaination of this phenomenon using elementary concepts and hypothesis in physics for more than half a centry, although splended progress has been made. 

Accompanying this dynamical slowing down in a glass transition, molecules move also more and more unevenly over locations and periods (space and time), a phenomenal called "dynamical heterogeneity". It basically states the fact that while particles at some places significantly wonder around over a period of time, particles at some other places vibrate in place like struggling to get out of their cages, and the fact that while any particle could be free to move for some periods of time, the same particle stucks over some other periods of time. This dynamical complexity has a central role in decyferring the glass transition as it is a signature of glassy dynamics. 
It has been shown by computer simulations that the levels of mobitiy of particles across an entire system over certain period of time starting from a given moment, say $t=0$, is largely decided only by the configuration at that moment $t=0$. A configuration is specified by the positions of particles in the entire system and the mobility of a particle is characterized by a quantity called "propensity". There are several definitions of propensity used in the literature, but the one we are using here is simply the statistical average of the distance being trevalled by a particle over a duration $\tau_\alpha$, that is the time scale over which the configuration has been significantly changed and directly related with the macrosopic viscosity. 
The strong link between the configuration at some moment and the propensity at some moment later, as evidenced by numerical experiments, points out one possible direction in demysifying the glass transition, that is  