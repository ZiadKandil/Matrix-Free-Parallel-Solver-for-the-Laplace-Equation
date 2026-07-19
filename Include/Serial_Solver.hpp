#ifndef SERIAL_SOLVER_HPP
#define SERIAL_SOLVER_HPP

#include "Grid.hpp"
#include <functional>

class Serial_Solver{

    public:

     /**
     * @brief Initializes the grid with the given values.
     * @param grid The grid structure 
     * @param f The forcing function (right-hand side)
     */
    static void initialize(Grid& grid, std::function<double(double, double)> f);

    /**
     * @brief Function to apply boundary conditions to the grid values.
     * @param grid The grid structure containing the solution vectors and grid parameters.
     * @param BC_func A function that computes the boundary condition value at given coordinates
     */
    static void apply_BC(Grid& grid, std::function<double(double, double)> BC_func);

    /**
     * @brief Performs one iteration of the Jacobi method to update the grid values.
     */
    static void jacobi_step(Grid& grid);

    /**
     * @brief Computes the error of the current grid values compared to the previous iteration.
     * @return The computed error as a double value.
     */
    static double compute_error(const Grid& grid);

    /**
     * @brief Computes the L2 error of the current grid values compared to the exact solution.
     * @param grid The grid structure
     * @param solution The current solution vector to compare against the exact solution
     * @param exact_sol A function that computes the exact solution at given coordinates (x, y)
     * @return The computed L2 error as a double value.
     */
    static double compute_l2_error(const Grid& grid, const std::vector<double>& solution, std::function<double(double, double)> exact_sol);

    /**
     * @brief Function to solve the equation in serial using the Jacobi method.
     * @param grid The grid structure containing the solution vectors and grid parameters.
     * @param tol The convergence tolerance for the iterative solver.
     * @param max_iters The maximum number of iterations to perform before stopping.
     * @param l2_error Reference to store the computed L2 error after convergence.
     * @param forcing_term A function that computes the forcing term (right-hand side) at given coordinates (x, y).
     * @param exact_solution A function that computes the exact solution at given coordinates (x, y) for error analysis.
     * @param BC_func A function that computes the boundary condition value at given coordinates
     */
    static void Solve_Serial(Grid &grid, double tol, int max_iters, double &l2_error,
                std::function<double(double, double)> forcing_term,
                std::function<double(double, double)> exact_solution,
                std::function<double(double, double)> BC_func);

};


#endif