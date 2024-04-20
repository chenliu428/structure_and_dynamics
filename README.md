# Structure and Dynamics in Glasses - a Machine Learning approach 

**Investigation of relevant ones among many types of structure descriptors for determining the mobility of particles in a supercooled liquid** using supervised machine learning techniques.

It showcases also how to gain insighgts of a pratical problem with a data driven approach by combining different machine learning models of different properties.

## Problem Context 
We are dealing with a problem in an active area of current research in statistical physics of glassy materials, such as SiO2. A type of material is glassy when the crystallisation (or more generally called the first order transition), such as water becoming ice, is avoided, bypassed or absent with a changing of its environment, usually a dropping temperature. Instead, these materials vitrify, that is they behave like a liquid but becoming more and more viscous with a decreasing temperature until at some point, they apparently stop flowing and stand there like a solid, which we call glasses. This apparent transition from a liquid to a "solid", allegedly call the "glass transition", is unconventional compared to crystallisation where the microscopic structure changes qualitatively from a disordered one of liquids to an ordered structure of crystals, like FCC, BCC, etc. In a glass transition, the microscopic structure is always disordered seemingly unchanged from a liquid, except that molecules have more and more difficulties to move under thermal aggitation, giving rise to its macroscopic drastic increase in viscosity, so large they apperently stop flowing. Physicists havn't come up with a proper explaination of this phenomenon using elementary concepts and hypothesis in physics for more than half a centry, although splended progress has been made. 

Accompanying this dynamical slowing down in a glass transition, molecules move also more and more unevenly over locations and periods (space and time), a phenomenal called "dynamical heterogeneity". It basically states the fact that while particles at some places significantly wonder around over a period of time, particles at some other places vibrate in place like struggling to get out of their cages, and the fact that while any particle could be free to move for some periods of time, the same particle stucks over some other periods of time. This dynamical complexity has a central role in decyferring the glass transition as it is a signature of glassy dynamics. 
It has been shown by computer simulations that the levels of mobitiy of particles across an entire system over certain period of time starting from a given moment, say $t=0$, is largely decided only by the configuration at that moment $t=0$. A configuration is specified by the positions of particles in the entire system, denoted $` \{\mathbf{R}_i\}_{1,2,\ldots,N} `$, and the mobility of a particle $i$ is characterized by a quantity called "propensity", denoted $p_i$, where $N$ the number of particles in a system. There are several definitions of propensity used in the literature, but the one we are using here is simply the statistical average of the distance being trevalled by a particle over a duration $\tau_\alpha$, that is the time scale over which the configuration has been significantly changed and directly related with the macrosopic viscosity. Hence the propensity is a positive scalar associated with some particle. 
The strong link between a configuration and the propensity, as evidenced by numerical experiments, indicates that a careful characterisation of this relation may pave the way in demystifying the glass transition. 

By "careful characterisation", we look for the information encoded in a configuration that is actually relevant to the propensity. A configuration, represented by $` \{\mathbf{R}_i\}_{1,2,\ldots,N} `$, is raw data containing both relevant and irrelevant information for our problem. To name one type of irrelevant information, consider translation of the entire system, which gives rise to a new list $` \{\mathbf{R}'_i\}_{1,2,\ldots,N} `$, but does not affect the propensity of any particle, as implied by the translation invariance of physical laws. The approach we are following in this project, as a physicists' tradition, is to measure physically motivated structure descriptors for each particle based on the relative positions of its surrounding particles. Beacause of the lack of prior knowledge, we include many ($M=96$) structure descriptors for each particle to form a list of structure descriptors $` \{\mathbf{x}_i\}_{i=1,2,\ldots,N} `$ with each $` \mathbf{x}_i \in \mathbb{R}^M `$. The task of "careful characterisation" of the link between configuration and propensity, becomes searching for an approximating function that describes well the relation between $` \{ \mathbf{x} _i \} _{i} `$ and $` \{p_i\}_{i} `$. That is actually a machine learning problem. By applying proper machine learning methods and by comparing machine learning models of different intrinsic properties, we may not only find a reasonable approximate target function, but also hopeful to gain physical interpretations, such as identifying relevant structure descriptors.  A further simplification, which actually carries physical assumptions, is adopted that the propensity of a particle $i$ is only related to its own structure descriptors $` \mathbf{x}_i `$. Hence, instead of searching for a target application between the two list $` \{\mathbf{x}_i\}_i `$ and $` \{p_i\}_i `$, we search for a target function $` \varphi (\cdot) `$ mapping a $M$ dimensional vector $`\mathbf{x}`$ to a scalar $p$, which constitutes the core objective of this project.  

Finally, it is worth emphasizing that whenever we callibrate a model against some data, the actual objective is to optimise the model with respect to certrain distribution underlying that dataset, to obtain a good generalisation for unseen data. Unlike most of real life problem,  for the physical purpose of our problem, the underlying probabilty distribution is theoretically known. Formally, it is the one particle Gibbs-Boltzman weight over the $M$ structure descriptors and the propensity $p$, denoted $` P_\text{GB}(\mathbf{x},p) `$.
This remark suggests us the correct way to use the available dataset to calibrate our models, instead of blindly treating each input-output pair as an independent data point. In other paragraphs of this presentation, the input-output pairs are carefully selected from the raw dataset, so that in either training or validation set, data points can be regarded as independently generated from the Gibbs-Boltzman weight. 