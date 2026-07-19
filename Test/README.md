# Scalability Test

## Prerequisites
```text
- MPI (mpirun) and an MPI C++ compiler (mpicxx)
- make
- Optional: gnuplot (for plotting)
```


## Reproduce
```bash
bash Test/Scalability_test.sh
```

## What it does
```text
- Builds the project.
- Runs the benchmark with 1, 2, and 4 MPI ranks (script sets OMP_NUM_THREADS=1).
- Saves per-run logs to Test/run_*.log.
- Writes a summary Markdown table to Test/Results.md.
```

## Manual build / run
```bash
# build
make 

# run a single configuration (example: 2 MPI ranks)
export OMP_NUM_THREADS=1
make run NPROCS=2
```

## Outputs
```text
- Test/Results.md — Markdown table of timings.
- Test/run_1.log, Test/run_2.log, Test/run_4.log — per-run logs.
- Test/data/ — data files produced by the program (e.g. timings.dat, l2_errors.dat).
```

## Notes
```text
- The script sets OMP_NUM_THREADS=1 for more accurate measurements depending only on the number of MPI processes.