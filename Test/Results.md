# Scalability Results

| MPI ranks | Wall time (s) |
|---:|---:|
| 1 | 30.90 |
| 2 | 23.72 |
| 4 | 23.73 |

From the results above and the plots we can conclude the following:

1- MPI process = 2 is better than 1 becaus it exploits the parallel implementation, and it is better than 4 because it is the best match for the hardware architecture.

2- For small values of n, parallelizaion doesn't improve the computational time and sometimes it gets slower due to the communication time between the processes.

3- For large values of n, parallelization becomes faster than serial implementation and it gets more efficient as n increases.

4- Schwarz method with direct solver and parallel implementation is the fastet method so far.