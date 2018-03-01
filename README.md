## CUDA-GSSA

## INTRODUCTION

Stochastic simulations of biological networks are invaluable for modeling networks with small species counts compared to traditional deterministic methods. Unfortunately, stochastic simulations of many biological systems often involve large, tightly coupled, networks that take prohibitively large amounts of computation time on modern CPUs.   With their increasingly widespread availability and low cost, graphics processing units (GPUs) have recently been explored as viable vehicles for accelerating such calculations.   In this research, we demonstrate a simple, parallelized version of the direct Gillespie Stochastic Simulation Algorithm (GSSA) using NVIDIA’s Compute Unified Device Architecture (CUDA).

## MATERIALS AND METHODS

A well-stirred, homogenous mixture neglecting both spatial considerations and reactions with time “delays” was assumed to provide the simplest implementation of Gillespie’s Direct Stochastic Simulation Algorithm.  The pseudo code of our serial implementation was as follows:

![Figure 0](https://i.imgur.com/e2jS9Dm.png|alt=figure0)

![Figure 1](https://i.imgur.com/NRdToWX.png)

**Figure 1.  Fundamental Program Aspects Overview.**
> Figure 1(a) presents the general user workflow in a standard flowchart.  Figure 1(b) shows the core data structures used in the GSSA code.  The Reaction Matrix, Parameter Array, and Propensity Array ranges span the number of different reactions within a particular network, the Specie Array spans the number of unique species and the Reaction Fired Matrix spans the number of reactions fired.  The Reaction Fired Matrix, Parameter Array, and Propensity Array consist of 32-bit floats, while the Reaction Matrix and Specie Array consist of 32-bit integers.  The square brackets used in the Reaction Matrix denote reactant or product indices in the Species Array, while the Δ denotes respective changes in their specie counts.  The row index of each reaction in the reaction matrix corresponds to its respective parameter and propensities in the Parameter and Propensity Arrays.  Figure 3(c) visualizes a Hillis-Steele parallel inclusive scan (Harris).  Note that the algorithm’s step complexity is log_2⁡n, while the work complexity is n*log_2⁡n, where n is the number of elements in an array (limited to 1048 elements in our naive implementation on the latest CUDA architecture).

To facilitate the simulation of reaction networks published in the scientific literature and to streamline the simulation process, a simple workflow was chosen as described in Figure 1(a).  The core data structures used to realize the serial version of the direct GSSA are described in Figure 1(b).   The parallel variants of the data structures were virtually identical, except the Reaction Fired Matrix, Species Array, and Propensity Array structures are duplicated according to how many simultaneous trajectories were being simulated.

![Figure 2](https://i.imgur.com/bZv3v4u.png)

**Figure 2. Basic CUDA Memory Hierarchy (CUDA C Programming Guide).**
> A CUDA programmer generally does not have control of per-thread local memory but can easily declare low-latency, high-bandwidth per-block shared memory.  Global memory accesses are high-latency (400-800 clock cycles), substantially lower bandwidth and highly dependent on coalesced, aligned access patterns to maximize bandwidth and minimize bank conflicts.

To utilize the basic CUDA programming model described in Figure 2, three primary parallel functions – “kernels” – were created to implement the direct GSSA.  These functions include the propensity calculations, the inclusive scan, and the reaction searching/firing methods.  All three methods varied in their level of parallelism and exploitation of the CUDA architecture.  The first kernel used to calculate reaction propensities simply assigned one thread to each propensity to be calculated and relied on global memory reads and writes (many, if not all, of the subsequent reads are presumably then kept in the L2 cache).  
The second kernel was slightly more complicated as it implemented a Hillis-Steele inclusive scan as described in Figure 1(c).  This kernel utilized constant memory to minimize global read accesses.  The third kernel that executed reaction searching and firing was virtually identical to its serial host code, except that its “embarrassing parallel” implementation assigned one trajectory per thread.  The reason for this simple implementation stemmed from the few opportunities for more advanced parallel exploitation since a binary search already has a small maximum step complexity of log_2⁡n.  The exact number of threads/block and number of blocks launched for each particular kernel are listed below (note that the minimum number of threads/block was chosen as 32 to take advantage of “warps” that can execute 32 simultaneous threads with a single instruction):

| Kernel | Number of Threads/Block | Number of Blocks |
| --- | --- | --- |
| calculatePropensities | 32 | (numReactions*numTrajectories+31)/32 |
| sumAllPropensities | numReactions | numTrajectories |
| findTargets | 32 | (numTrajectories+31)/32 |

Algorithm performance was benchmarked against randomly generated networks and the largest network found in the scientific literature at the time of research.  To find the most sizeable network amenable to stochastic simulation, the BioModels.net database was downloaded and each of its networks was inputted to the StochPy program (Maarleveld) for attempted stochastic simulation.  90% of simulations failed but the largest network for a successful simulation was found to consist of 132 reactions and 76 species describing cartilage breakdown (Proctor).   Random networks were generated in a logarithmic fashion according to the total number of species and reactions (where the number of species was equal to the number of reactions), ranging from 2 to 2048 species.  1,000 simulation runs were executed with approximately 10,000 updates on each network.  CPU code was compiled separately by Intel’s C Compiler with automatic vectorization and NVIDIA’s CUDA Compiler.

A quad-core Intel Q6600 processor (2.4 GHz, 8M L2 cache) with 6GB of DDR2-800 (PC6400) RAM was used in conjunction with an NVIDIA GTX 750 Ti graphics card (5 streaming multiprocessors, 128 arithmetic cores/multiprocessor, 1.02 GHz, Maxwell 5.0 architecture) in a 64-bit Windows 7 environment.

## RESULTS

![Figure 3](https://i.imgur.com/uwFVZ3K.png)
**Figure 3.  CPU Random Network Performance.**
> Both CPU codes scaled in a linear fashion once the network size reached a minimum of 64 reactants and species.  Note that both chart axes are of log_2 scale.

![Figure 4](https://i.imgur.com/xxa5RT8.png)
**Figure 4.  GPU Random Network Performance.**
> Graphics card resources became saturated between network sizes of 128 and 256 total reactants and species.  Note that both chart axes are of log_2 scale.

![Figure 5](https://i.imgur.com/uwFVZ3K.png)
**Figure 5.  GPU/CPU Random Network Speed-Up.**
> The parallel GPU code outperformed the serial, single-processor CPU code for all network sizes tested.

![Figure 6](https://i.imgur.com/HBQONkn.png)
**Figure 6.  Cartilage Breakdown Performance.**
> The GPU code outperformed the vectorized CPU code and the regular CPU code by 3.70x and 7.47x, respectively. 

## DISCUSSION

All networks tested were found to execute faster on the GPU compared to a single CPU core.  The exact speedup, however, was dependent on the CPU’s utilization of vectorized instructions.  For networks with total amounts of species and reactants less than 64 reactants, the non-vectorized CPU code was occasionally faster than the vectorized CPU code (Figs. 3 and 5).   This surprising behavior likely resulted from penalties paid for non-contiguous memory accesses and 64 byte line sizes in the Q6600’s L1 cache.  For all network sizes tested, however, the benchmarks suggest that a parallel CPU implementation with sufficient single-chip resources could potentially outperform the GPU.  Nonetheless, substantial shortcomings with the current GPU implementation still remain.

Increasing the work per thread could dramatically improve the inclusive scan kernel both in efficiency and in removing its maximum propensity array length limitation of 1024 elements.  A radically different approach in utilizing “fat threads” could also result in substantial performance gains (Maarleveld).  Such methods involve merging all of the kernels into one and storing specie and propensity arrays in shared memory.  While the performance of such methods have been demonstrated (Petzold), their capacity to fully utilize GPU resources can be limited because of the limited network sizes that fit in shared memory (Komarov).  Additionally, all of the cited GPU methods incorporated reaction dependency graphs (which are absent from Gillespie’s original algorithm) in their implementations and hence may not necessarily accelerate our implementation. 

Finally, the limitations of attempting to compare the performance of a parallel algorithm to a serial one by comparing an entire, specific graphics card to a specific, single CPU core are emphasized.  Should a higher-end consumer-grade NVIDIA card such as the GTX 980 have been employed, the GPU’s average simulation times would likely have been 2-3x lower than the ones provided by the low-end 750 Ti we utilized.  Additionally, a more recent CPU with larger SIMD register files could have dramatically accelerated the vectorized CPU code.  Intel’s latest Advanced Vector Extensions 512, for example, offer up to 16 simultaneous 32-bit floating point calculations – 4 times more than the SSSE3 instruction set tested in the Q6600.

## References

Figure 7. Memory Hierarchy. Digital image. CUDA C Programming Guide. NVIDIA Corporation. <http://docs.nvidia.com/cuda/cuda-c-programming-guide/>.

Harris, Mark, Shubhabrata Sengupta, and John D. Owens. Figure 39-2. Digital image. GPU Gems 3: Chapter 39. Parallel Prefix Sum (Scan) with CUDA. http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html

Klingbeil, G., Erban, R., Giles, M., & Maini, P. K. (2012). Fat versus Thin Threading Approach on GPUs: Application to Stochastic Simulation of Chemical Reactions. IEEE Transactions on Parallel and Distributed Systems, 23(2), 280–287. doi:10.1109/TPDS.2011.157

Komarov, I., D’Souza, R. M., & Tapia, J.-J. (2012). Accelerating the Gillespie τ-Leaping Method using graphics processing units. PloS One, 7(6), e37370. doi:10.1371/journal.pone.0037370

Maarleveld, T. R., Olivier, B. G., & Bruggeman, F. J. (2013). StochPy : A Comprehensive , User-Friendly Tool for Simulating Stochastic Biological Processes, 8(11). doi:10.1371/journal.pone.0079345

Petzold, L. (2009). Efficient Parallelization of the Stochastic Simulation Algorithm for Chemically Reacting Systems On the Graphics Processing Unit. International Journal of High Performance Computing Applications, 24(2), 107–116. doi:10.1177/1094342009106066

Proctor, C. J., Macdonald, C., Milner, J. M., Rowan, a D., & Cawston, T. E. (2014). A computer simulation approach to assessing therapeutic intervention points for the prevention of cytokine-induced cartilage breakdown. Arthritis & Rheumatology (Hoboken, N.J.), 66(4), 979–89. doi:10.1002/art.38297
