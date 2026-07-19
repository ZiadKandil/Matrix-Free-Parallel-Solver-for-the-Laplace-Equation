#ifndef GRID_HPP
#define GRID_HPP

#include <vector>

struct Grid{

    int global_n;       /// < Global grid size
    int local_rows;     /// < Number of rows in the local grid
    int start_row;      /// < Starting row index in the local grid
    int nx;             /// < Number of local rows + 2 (for ghost cells)
    int ny;             /// < Number of columns in the grid (same for all processes = global_n)
    double h;           /// < Grid spacing

    std::vector<double> u_old;  /// < Previous time step solution
    std::vector<double> u_new;  /// < Current time step solution
    std::vector<double> rhs;    /// < Right-hand side of the Poisson equation

    /**
     * @brief Constructor to initialize the grid parameters and allocate memory for the solution vectors.
     * @param global_n Global grid size
     * @param local_rows Number of rows in the local grid
     * @param start_row Starting row index in the global grid
     */
    Grid(int global_n, int local_rows, int start_row);

    /**
     * @brief Function to convert index from 2D grid coordinates to 1D vector index
     * @param i Row index in the local grid
     * @param j Column index in the grid
     * @return The corresponding index in the 1D solution vectors
     */
    inline int idx(int i, int j) const{
        return i * ny + j;
    }

    /**
     * @brief Function to initialize the right-hand side
     */
    void initialize();

    /**
     * @brief Function to get the x coordinates of the grid point
     * @param j Column index in the grid
     * @return The x coordinate of the grid point
     */
    double x(int j) const;

    /**
     * @brief Function to get the y coordinates of the grid point
     * @param i Row index in the local grid
     * @return The y coordinate of the grid point
     */
    double y(int i) const;
};

#endif