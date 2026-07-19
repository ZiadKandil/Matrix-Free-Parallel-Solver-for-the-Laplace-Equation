#ifndef Parallel_Solver_HPP
#define Parallel_Solver_HPP

#include "Grid.hpp"

#include <functional>

class Parallel_Solver{

    public:

    /**
     * @brief Initializes the grid with the given values.
     * @param grid The local grid structure 
     * @param f The forcing function (right-hand side)
     */
    static void initialize(Grid& grid, std::function<double(double, double)> f);


    /**
     * @brief Function to apply boundary conditions to the grid values.
     * @param grid The local grid structure containing the solution vectors and grid parameters.
     * @param BC_func A function that computes the boundary condition value at given coordinates
     */
    static void apply_BC(Grid& grid, std::function<double(double, double)> BC_func);
    
    /**
     * @brief Performs one iteration of the Jacobi method to update the grid values.
     */
    static void jacobi_step(Grid& grid);

    /**
     * @brief Computes the local error of the current grid values compared to the previous iteration.
     * @return The computed local error as a double value.
     */
    static double compute_local_error(const Grid& grid);

    /**
     * @brief Computes the local L2 error of the current grid values compared to the exact solution.
     * @param grid The local grid structure
     * @param solution The current solution vector to compare against the exact solution
     * @param exact_sol A function that computes the exact solution at given coordinates (x, y)
     * @return The computed local L2 error as a double value.
     */
    static double compute_l2_error(const Grid& grid, const std::vector<double>& solution, std::function<double(double, double)> exact_sol);

    /**
     * @brief Function to exchange ghost rows between neighboring processes using MPI.
     * Each process will send its top row to the process above and
     * receive the bottom ghost row from the process below, and vice versa.
     * @param grid The local grid structure containing the solution vectors and grid parameters.
     * @param rank The rank of the current process.
     * @param size The total number of processes in the MPI communicator.
     */
    static void exchange_ghost_rows(Grid& grid, int rank, int size);

    /**
     * @brief Function to solve the equation in parallel using MPI and OpenMP.
     * @param grid The local grid structure.
     * @param rank The rank of the current process.
     * @param size The total number of processes in the MPI communicator.
     * @param tol The convergence tolerance for the iterative solver.
     * @param max_iters The maximum number of iterations to perform before stopping.
     * @param l2_error Reference to store the computed L2 error after convergence.
     * @param forcing_term A function that computes the forcing term (right-hand side) at given coordinates (x, y).
     * @param exact_solution A function that computes the exact solution at given coordinates (x, y) for error analysis.
     * @param BC_func A function that computes the boundary condition value at given coordinates
     */
    static void Solve_Parallel(Grid &grid, int rank, int size, double tol, int max_iters, double &l2_error,
                    std::function<double(double, double)> forcing_term,
                    std::function<double(double, double)> exact_solution,
                    std::function<double(double, double)> BC_func);
};


#endif