#include "Grid.hpp"

#include <cmath>
#include <numbers>

constexpr double pi = std::numbers::pi_v<double>;


/**
 * @brief Constructor to initialize the grid parameters and allocate memory for the solution vectors.
 * @param global_n Global grid size
 * @param local_rows Number of rows in the local grid
 * @param start_row Starting row index in the global grid
 */
Grid::Grid(int global_n, int local_rows, int start_row):

    global_n(global_n),
    local_rows(local_rows),
    start_row(start_row){

        nx = local_rows + 2;        /// < Add 2 for ghost cells
        ny = global_n;              /// < Number of columns is the same for all processes
        h = 1.0 / (global_n - 1);   /// < Grid spacing
        int total_size = nx * ny;   /// < Total size of the local grid including ghost cells

        u_old.resize(total_size, 0.0); 
        u_new.resize(total_size, 0.0);
        rhs.resize(total_size, 0.0);
    }

/**
 * @brief Function to get the x coordinates of the grid point
 * @param j Column index in the grid
 * @return The x coordinate of the grid point
 */
double Grid::x(int j) const{
    return j * h;
}

/**
 * @brief Function to get the y coordinates of the grid point
 * @param i Row index in the local grid
 * @return The y coordinate of the grid point
 */
double Grid::y(int i) const{
    int global_i = start_row + (i - 1);  /// < Convert local row index to global row index
    return global_i * h;
}

/**
 * @brief Function to initialize the right-hand side
 */
void Grid::initialize(){

    for(int i = 1; i <= local_rows; ++i){

        for(int j = 0; j < ny; ++j){

            double x_coord = x(j);
            double y_coord = y(i);

            rhs[idx(i, j)] = 8.0 * pi * pi * 
                            sin(2.0 * pi * x_coord) * 
                            sin(2.0 * pi * y_coord); 
        }
    }
}