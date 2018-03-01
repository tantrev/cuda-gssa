## CUDA-GSSA

## INTRODUCTION

Stochastic simulations of biological networks are invaluable for modeling networks with small species counts compared to traditional deterministic methods. Unfortunately, stochastic simulations of many biological systems often involve large, tightly coupled, networks that take prohibitively large amounts of computation time on modern CPUs.   With their increasingly widespread availability and low cost, graphics processing units (GPUs) have recently been explored as viable vehicles for accelerating such calculations.   In this research, we demonstrate a simple, parallelized version of the direct Gillespie Stochastic Simulation Algorithm (GSSA) using NVIDIA’s Compute Unified Device Architecture (CUDA).

## MATERIALS AND METHODS

A well-stirred, homogenous mixture neglecting both spatial considerations and reactions with time “delays” was assumed to provide the simplest implementation of Gillespie’s Direct Stochastic Simulation Algorithm.  The pseudo code of our serial implementation was as follows:

![Figure 0](https://i.imgur.com/e2jS9Dm.png|alt=figure0)

**Figure 1.  Fundamental Program Aspects Overview.**
Figure 1(a) presents the general user workflow in a standard flowchart.  Figure 1(b) shows the core data structures used in the GSSA code.  The Reaction Matrix, Parameter Array, and Propensity Array ranges span the number of different reactions within a particular network, the Specie Array spans the number of unique species and the Reaction Fired Matrix spans the number of reactions fired.  The Reaction Fired Matrix, Parameter Array, and Propensity Array consist of 32-bit floats, while the Reaction Matrix and Specie Array consist of 32-bit integers.  The square brackets used in the Reaction Matrix denote reactant or product indices in the Species Array, while the Δ denotes respective changes in their specie counts.  The row index of each reaction in the reaction matrix corresponds to its respective parameter and propensities in the Parameter and Propensity Arrays.  Figure 3(c) visualizes a Hillis-Steele parallel inclusive scan (Harris).  Note that the algorithm’s step complexity is log_2⁡n, while the work complexity is n*log_2⁡n, where n is the number of elements in an array (limited to 1048 elements in our naive implementation on the latest CUDA architecture).
