#ifndef IO_HPP
#define IO_HPP

#include "Grid.hpp"
#include "Utils.hpp"

#include <string>

class IO{

    public:

    /**
     * @brief Function to write the grid values to a VTK file for visualization.
     * @param grid The local grid structure containing the solution vectors and grid parameters.
     * @param filename The name for the output VTK files.
     * @param rank The rank of the current process
     * @param size The total number of processes 
     */
    static void write_vtk(
        const Grid& grid,
        const std::string& filename,
        int rank,
        int size
    );

    /**
     * @brief Function to write the solution and grid coordinates to a text file.
     * @param grid The local grid structure containing the solution vectors and grid parameters.
     * @param filename The name for the output text files.
     * @param rank The rank of the current process
     * @param size The total number of processes 
     */
    static void write_text(
        const Grid& grid,
        const std::string& filename,
        int rank,
        int size
    );

    /**
     * @brief Function to write the solution and grid coordinates to a csv file.
     * @param grid The local grid structure containing the solution vectors and grid parameters.
     * @param filename The name for the output csv files.
     * @param rank The rank of the current process
     * @param size The total number of processes 
     */
    static void write_csv(
        const Grid& grid,
        const std::string& filename,
        int rank,
        int size
    );

    /**
     * @brief Function to write the grid values to a VTK file for visualization in serial.
     * @param grid The local grid structure containing the solution vectors and grid parameters.
     * @param filename The name for the output VTK file.
     */
    static void write_vtk_serial(
        const Grid& grid,
        const std::string& filename
    );

    /**
     * @brief Function to write the solution and grid coordinates to a text file in serial.
     * @param grid The local grid structure containing the solution vectors and grid parameters.
     * @param filename The name for the output text file.
     */
    static void write_text_serial(
        const Grid& grid,
        const std::string& filename
    );

    /**
     * @brief Function to write the solution and grid coordinates to a csv file in serial.
     * @param grid The local grid structure containing the solution vectors and grid parameters.
     * @param filename The name for the output csv file.
     */
    static void write_csv_serial(
        const Grid& grid,
        const std::string& filename
    );


    /**
     * @brief Function to save data to a dat file for plotting.
     * @param data_Parallel The parallel data to be saved.
     * @param data_Serial The serial data to be saved.
     * @param data_schwarz The Schwarz data to be saved.
     * @param grid_sizes The sizes of the grid.
     * @param filename The name for the output dat file.
     */
    static void save_data(const std::vector<double>& data_Parallel, const std::vector<double>& data_Serial,
         const std::vector<double>& data_schwarz, const std::vector<int>& grid_sizes, const std::string& filename);
};

#endif