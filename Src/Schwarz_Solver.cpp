#include "Schwarz_Solver.hpp"

#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>


/**
 * @brief Initializes the grid with the given values.
 * @param grid The local grid structure 
 * @param f The forcing function (right-hand side)
 */
void Schwarz_Solver::initialize(Grid& grid, std::function<double(double, double)> f){

    #pragma omp parallel for

        for(int i = 1; i <= grid.local_rows; ++i){

            for(int j = 0; j < grid.ny; ++j){

                double x_coord = grid.x(j);
                double y_coord = grid.y(i);

                grid.rhs[grid.idx(i,j)] = f(x_coord, y_coord);
            }
        }
}

/**
 * @brief Function to apply boundary conditions to the grid values.
 * @param grid The local grid structure containing the solution vectors and grid parameters.
 * @param BC_func A function that computes the boundary condition value at given coordinates 
 */
void Schwarz_Solver::apply_BC(Grid& grid, std::function<double(double, double)> BC_func){

    #pragma omp parallel for

    for(int i = 1; i <= grid.local_rows; ++i){

        double x_coord = grid.x(0);
        double y_coord = grid.y(i);

        grid.u_old[grid.idx(i,0)] = BC_func(x_coord, y_coord);
        grid.u_old[grid.idx(i, grid.ny - 1)] = BC_func(grid.x(grid.ny - 1), y_coord);
        grid.u_new[grid.idx(i,0)] = BC_func(x_coord, y_coord);
        grid.u_new[grid.idx(i, grid.ny - 1)] = BC_func(grid.x(grid.ny - 1), y_coord);

    }

    const bool is_first = (grid.start_row == 0);
    const bool is_last = (grid.start_row + grid.local_rows == grid.global_n);

    if(is_first){

        #pragma omp parallel for

        for(int j = 0; j < grid.ny; ++j){

            double x_coord = grid.x(j);

            grid.u_old[grid.idx(1,j)] = BC_func(x_coord, 0.0);
            grid.u_new[grid.idx(1,j)] = BC_func(x_coord, 0.0);
        }
    }

    if(is_last){

        #pragma omp parallel for

        for(int j = 0; j < grid.ny; ++j){

            double x_coord = grid.x(j);

            grid.u_old[grid.idx(grid.local_rows, j)] = BC_func(x_coord, 1.0);
            grid.u_new[grid.idx(grid.local_rows, j)] = BC_func(x_coord, 1.0);
        }
    }
}

/**
 * @brief Builds the local sparse matrix for the equation based on the grid parameters.
 * @param grid The local grid structure containing the solution vectors and grid parameters.
 * @param A Reference to an Eigen::SparseMatrix that will be filled with the local matrix values.
 */
void Schwarz_Solver::build_local_matrix(const Grid& grid, Eigen::SparseMatrix<double>& A){

    using Triplet = Eigen::Triplet<double>;

    const bool is_first = (grid.start_row == 0);

    const bool is_last = (grid.start_row + grid.local_rows == grid.global_n);

    const int i_begin = is_first ? 2 : 1;

    const int i_end = is_last ? grid.local_rows - 1 : grid.local_rows;

    const int ny_local = grid.ny - 2;

    const int nx_local = i_end - i_begin + 1;

    const int N = nx_local * ny_local;

    auto lid = [=](int i,int j){

        return (i-i_begin)*ny_local +(j-1);
    };

    std::vector<Triplet> triplets;
    triplets.reserve(5*N);

    for(int i=i_begin;i<=i_end;++i)
    {
        for(int j=1;j<grid.ny-1;++j)
        {
            int row = lid(i,j);

            triplets.emplace_back(row,row,4.0);

            if(i>i_begin)
            {
                triplets.emplace_back(row, lid(i-1,j),-1.0);
            }

            if(i<i_end)
            {
                triplets.emplace_back(row, lid(i+1,j),-1.0);
            }

            if(j>1)
            {
                triplets.emplace_back(
                    row,
                    lid(i,j-1),
                    -1.0);
            }

            if(j<grid.ny-2)
            {
                triplets.emplace_back(row, lid(i,j+1), -1.0);
            }
        }
    }

    A.resize(N,N);

    A.setFromTriplets(triplets.begin(), triplets.end());

    }

/**
 * @brief Solves the equation in parallel using the Schwarz method.
 * @param grid The local grid structure.
 * @param solver The sparse LU solver.
 */
void Schwarz_Solver::local_solve(Grid& grid, Eigen::SparseLU<Eigen::SparseMatrix<double>>& solver){

    const bool is_first =
        (grid.start_row == 0);

    const bool is_last =
        (grid.start_row + grid.local_rows
         == grid.global_n);

    const int i_begin =
        is_first ? 2 : 1;

    const int i_end =
        is_last
        ? grid.local_rows - 1
        : grid.local_rows;

    const int ny_local =
        grid.ny - 2;

    auto lid =
        [=](int i,int j)
        {
            return
                (i-i_begin)*ny_local
                +(j-1);
        };

    const int N =
        (i_end-i_begin+1)
        * ny_local;

    Eigen::VectorXd b(N);

    const double h2 =
        grid.h * grid.h;

    #pragma omp parallel for
    for(int i=i_begin;i<=i_end;++i)
    {
        for(int j=1;j<grid.ny-1;++j)
        {
            int row =
                lid(i,j);

            double rhs =
                h2 *
                grid.rhs[
                    grid.idx(i,j)];

            if(i==i_begin)
            {
                rhs +=
                    grid.u_old[
                        grid.idx(i-1,j)];
            }

            if(i==i_end)
            {
                rhs +=
                    grid.u_old[
                        grid.idx(i+1,j)];
            }

            if(j==1)
            {
                rhs +=
                    grid.u_old[
                        grid.idx(i,j-1)];
            }

            if(j==grid.ny-2)
            {
                rhs +=
                    grid.u_old[
                        grid.idx(i,j+1)];
            }

            b(row)=rhs;
        }
    }

    Eigen::VectorXd x =
        solver.solve(b);

    #pragma omp parallel for
    for(int i=i_begin;i<=i_end;++i)
    {
        for(int j=1;j<grid.ny-1;++j)
        {
            grid.u_new[
                grid.idx(i,j)]
                =
                x(
                    lid(i,j));
        }
    }

}

/**
 * @brief Computes the local error of the current grid values compared to the previous iteration.
 * @return The computed local error as a double value.
 */
double Schwarz_Solver::compute_local_error(const Grid& grid){

    double local_sum = 0.0;

    #pragma omp parallel for reduction (+: local_sum)

    for(int i = 1; i <= grid.local_rows; ++i){

        for(int j = 0; j < grid.ny; ++j){

            double diff = grid.u_new[grid.idx(i,j)] - grid.u_old[grid.idx(i,j)];
            local_sum += diff * diff;
        }
    }

    return std::sqrt(local_sum * grid.h);
}

/**
 * @brief Computes the local L2 error of the current grid values compared to the exact solution.
 * @param grid The local grid structure
 * @param solution The current solution vector to compare against the exact solution
 * @param exact_sol A function that computes the exact solution at given coordinates (x, y)
 * @return The computed local L2 error as a double value.
 */
double Schwarz_Solver::compute_l2_error(const Grid& grid, const std::vector<double>& solution, std::function<double(double, double)> exact_sol){

    double local_sum = 0.0;

    #pragma omp parallel for reduction (+: local_sum)

    for(int i = 1; i <= grid.local_rows; ++i){

        for(int j = 0; j < grid.ny; ++j){

            double x_coord = grid.x(j);
            double y_coord = grid.y(i);

            double exact = exact_sol(x_coord, y_coord);
            double diff = solution[grid.idx(i,j)] - exact;
            local_sum += diff * diff;
        }
    }

    double error = 0.0;

    MPI_Allreduce(
        &local_sum,
        &error,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    error = std::sqrt(grid.h * error);

    return error;
}

/**
 * @brief Function to exchange ghost rows between neighboring processes using MPI.
 * Each process will send its top row to the process above and
 * receive the bottom ghost row, and vice versa.
 * @param grid The local grid structure containing the solution vectors and grid parameters.
 * @param rank The rank of the current process.
 * @param size The total number of processes in the MPI communicator.
 */
void Schwarz_Solver::exchange_ghost_rows(Grid& grid, int rank, int size){

    int up = rank - 1;      /// < Rank of the process above
    int down = rank + 1;    /// < Rank of the process below

    if(up >= 0){

        /// < Send top row to the process above
        /// < Receive top ghost row from upward neighbor

        MPI_Sendrecv(
            &grid.u_old[grid.idx(1, 0)],
            grid.ny,
            MPI_DOUBLE,
            up,
            0,

            &grid.u_old[grid.idx(0, 0)],
            grid.ny,
            MPI_DOUBLE,
            up,
            1,

            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }

    if(down < size){

        /// < Send bottom row to the process below
        /// < Receive bottom ghost row from downward neighbor

        MPI_Sendrecv(
            &grid.u_old[grid.idx(grid.local_rows, 0)],
            grid.ny,
            MPI_DOUBLE,
            down,
            1,

            &grid.u_old[grid.idx(grid.local_rows + 1, 0)],
            grid.ny,
            MPI_DOUBLE,
            down,
            0,

            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }
}


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
void Schwarz_Solver::Solve_Schwarz(Grid &grid, int rank, int size, double tol, int max_iters, double &l2_error,
                std::function<double(double, double)> forcing_term,
                std::function<double(double, double)> exact_solution,
                std::function<double(double, double)> BC_func){

    Schwarz_Solver::initialize(grid, forcing_term);

    Schwarz_Solver::apply_BC(grid, BC_func);

    Eigen::SparseMatrix<double> A;

    build_local_matrix(grid,A);

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;

    solver.analyzePattern(A);

    solver.factorize(A);

    bool converged = false;
    int iter = 0;

    while(!converged && iter < max_iters){

        Schwarz_Solver::exchange_ghost_rows(grid, rank, size);

        Schwarz_Solver::local_solve(grid, solver);

        Schwarz_Solver::apply_BC(grid, BC_func);

        double local_error = Schwarz_Solver::compute_local_error(grid);

        bool local_converged = (local_error < tol);

        int local_flag = local_converged ? 1 : 0;
        int global_flag = 0;

        MPI_Allreduce(
            &local_flag,
            &global_flag,
            1,
            MPI_INT,
            MPI_MIN,
            MPI_COMM_WORLD
        );

        converged = (global_flag == 1);

        std::swap(grid.u_old, grid.u_new);

        ++iter;
    }

    l2_error = Schwarz_Solver::compute_l2_error(
        grid,
        grid.u_old,
        exact_solution
    );

    if(rank == 0){
        std::cout << "n = " << grid.global_n << " Schwarz Solver Converged in " 
            << iter << " iterations with L2 error: " << l2_error << std::endl;
    }
}
