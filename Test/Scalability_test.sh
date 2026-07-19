#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p Test
make -j

export OMP_NUM_THREADS=1

RESULTS_FILE="Test/Results.md"

{
    echo "# Scalability Results"
    echo
    echo "| MPI ranks | Wall time (s) |"
    echo "|---:|---:|"
} > "$RESULTS_FILE"

for nprocs in 1 2 4; do
    echo "Running with ${nprocs} MPI ranks"
    time_file="Test/time_${nprocs}.txt"
    /usr/bin/time -f "%e" -o "$time_file" \
        make run NPROCS="$nprocs" > "Test/run_${nprocs}.log" 2>&1

    wall_time=$(<"$time_file")
    rm -f "$time_file"
    echo "| ${nprocs} | ${wall_time} |" >> "$RESULTS_FILE"
done

echo
echo "Done. Results written to $RESULTS_FILE"
echo "Logs are in Test/"
echo "Data files are in Test/data/"