# Parallel Solver for the Laplace Equation

A high-performance C++ implementation of a 2D Laplace equation solver featuring:

- Serial Jacobi solver
- MPI + OpenMP parallel Jacobi solver
- Schwarz domain decomposition solver
- Error analysis against an analytical solution
- VTK, CSV, and TXT output generation for visualization and post-processing
- Scalability testing utilities

---

## Overview

This project solves the 2D Laplace equation

$$
-\Delta u = f(x,y)
$$

on a square domain with homogeneous Dirichlet boundary conditions.

The implementation compares three approaches:

1. **Serial Solver**
   - Classical Jacobi iteration.

2. **Parallel Solver**
   - Domain decomposition using MPI.
   - Shared-memory acceleration using OpenMP.
   - Ghost-cell communication between neighboring MPI processes.

3. **Schwarz Solver**
   - Domain decomposition based on the Schwarz iterative method.
   - Local sparse systems solved using Eigen's SparseLU solver.

The numerical solution is compared against the analytical solution:

$$
u(x,y)=\sin(2\pi x)\sin(2\pi y)
$$

with forcing term:

$$
f(x,y)=8\pi^2 \sin(2\pi x)\sin(2\pi y)
$$

---

## Project Structure

```text
.
├── Include/
│   ├── Grid.hpp
│   ├── IO.hpp
│   ├── Parallel_Solver.hpp
│   ├── Schwarz_Solver.hpp
│   ├── Serial_Solver.hpp
│   └── Utils.hpp
│
├── Src/
│   ├── Grid.cpp
│   ├── IO.cpp
│   ├── Parallel_Solver.cpp
│   ├── Schwarz_Solver.cpp
│   ├── Serial_Solver.cpp
│   ├── Utils.cpp
│   └── main.cpp
│
├── Outputs/
│   ├── Serial_solution_*.vtk
│   ├── Parallel_solution_*.vtk
│   └── Schwarz_solution_*.vtk
│
├── Test/
│   ├── Scalability_test.sh
│   ├── Results.md
│   └── data/
│
├── Makefile
└── README.md
```

---

## Requirements

### Compiler

- C++20 compatible compiler

### Libraries

- MPI
- OpenMP
- Eigen3

### Optional

- Gnuplot (for automatic result plotting)

---

## Build

Compile the project using:

```bash
make
```

This produces the executable:

```text
main
```

---

## Run

### Default Run

```bash
make run
```

### Specify Number of MPI Processes

```bash
make run NPROCS=4
```

Equivalent command:

```bash
mpirun -np 4 ./main
```

### Control OpenMP Threads

```bash
export OMP_NUM_THREADS=4
make run NPROCS=4
```

---

## Numerical Setup

The solver runs automatically on the following grid sizes:

```text
16 × 16
32 × 32
64 × 64
128 × 128
256 × 256
```

Parameters:

```text
Maximum iterations : 200000
Tolerance          : 1e-9
```

---

## Outputs

For each grid size and solver, the following files are generated:

### VTK

```text
Serial_solution_64.vtk
Parallel_solution_64.vtk
Schwarz_solution_64.vtk
```

These files can be visualized with:

- ParaView
- VisIt

### CSV

```text
Serial_solution_64.csv
Parallel_solution_64.csv
Schwarz_solution_64.csv
```

### Text

```text
Serial_solution_64.txt
Parallel_solution_64.txt
Schwarz_solution_64.txt
```

---

## Error Analysis

The code computes the L2 error norm with respect to the exact analytical solution.

Results are stored in:

```text
Test/data/l2_errors.dat
```

---

## Performance Measurements

Execution times are recorded for:

- Serial solver
- Parallel MPI/OpenMP solver
- Schwarz solver

Results are stored in:

```text
Test/data/timings.dat
```

If Gnuplot is available, performance and error plots are generated automatically at the end of execution.

---

## Scalability Testing

Run the scalability benchmark:

```bash
bash Test/Scalability_test.sh
```

The script:

- Builds the project
- Executes runs with 1, 2, and 4 MPI processes
- Stores logs
- Generates a summary table

Outputs:

```text
Test/run_1.log
Test/run_2.log
Test/run_4.log
Test/Results.md
```

---

## Parallelization Strategy

### MPI

The computational domain is partitioned row-wise across MPI ranks.

Each process owns:

- Interior rows
- Ghost rows

Neighboring processes exchange boundary information through MPI communication.

### OpenMP

OpenMP is used to parallelize local computational loops inside each MPI process.

### Schwarz Method

The Schwarz solver:

1. Decomposes the domain into subdomains.
2. Builds local sparse matrices.
3. Solves local systems using Eigen SparseLU.
4. Exchanges interface values between neighboring subdomains.
5. Iterates until convergence.

---

## License

This project is distributed under the terms of the LICENSE file included in the repository.