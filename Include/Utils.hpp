#ifndef UTILS_HPP
#define UTILS_HPP

class Utils{

    public:

    /**
     * @brief Function to compute the local number of rows and the starting
     *  row index for each process
     * @param global_n The global grid size (number of rows/columns).
     * @param size The total number of processes.
     * @param rank The rank of the current process.
     * @param local_rows Store the computed number of local rows for the current process.
     * @param start_row  Store the computed starting row index for the current process.
     */
    static void compute_partitions(
        int global_n,
        int size,
        int rank,
        int& local_rows,
        int& start_row
    );
};

#endif