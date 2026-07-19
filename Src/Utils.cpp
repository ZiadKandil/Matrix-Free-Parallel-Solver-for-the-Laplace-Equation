#include "Utils.hpp"

#include <algorithm>

/**
 * @brief Function to compute the local number of rows and the starting
 *  row index for each process
 * @param global_n The global grid size (number of rows/columns).
 * @param size The total number of processes.
 * @param rank The rank of the current process.
 * @param local_rows Store the computed number of local rows for the current process.
 * @param start_row  Store the computed starting row index for the current process.
 */
void Utils::compute_partitions(
    int global_n,
    int size,
    int rank,
    int& local_rows,
    int& start_row
){
    int base_rows = global_n / size;    /// < Base number of rows for each process
    int remainder = global_n % size;    /// < Remaining rows to be distributed among the first 'remainder' processes

    local_rows = base_rows + (rank < remainder); /// < If the rank is less than the remainder, it gets one extra row
    
    start_row = rank * base_rows + std::min(rank, remainder);  
}