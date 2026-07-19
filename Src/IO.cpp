#include "IO.hpp"

#include <fstream>
#include <vector>
#include <mpi.h>
#include <filesystem>

/**
 * @brief Function to write the grid values to a VTK file for visualization.
 * @param grid The local grid structure containing the solution vectors and grid parameters.
 * @param filename The name for the output VTK files.
 * @param rank The rank of the current process
 * @param size The total number of processes 
 */
void IO:: write_vtk(
    const Grid& grid,
    const std::string& filename,
    int rank,
    int size
){
    int local_size = grid.local_rows * grid.ny; /// < Number of points in the local grid (excluding ghost cells)

    std::vector<int> recv_counts;               /// < Number of elements to receive from each process
    std::vector<int> displs;                    /// < Displacements for MPI_Gatherv

    if(rank == 0){

        recv_counts.resize(size);
        displs.resize(size);
    }

    MPI_Gather(
        &local_size,
        1,
        MPI_INT,
        recv_counts.data(),
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    std::vector<double> global_solution;

    if(rank == 0){

        int total_size = 0;

        for(int i = 0; i < size; ++i){

            displs[i] = total_size;
            total_size += recv_counts[i];
        }

        global_solution.resize(total_size);
    }

    MPI_Gatherv(
        &grid.u_old[grid.idx(1,0)],
        local_size,
        MPI_DOUBLE,

        global_solution.data(),
        recv_counts.data(),
        displs.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    if (rank != 0){
        return;         /// < Only the root process will write the output file
    }

    std::filesystem::create_directories("Outputs");
    std::filesystem::path out_path = std::filesystem::path("Outputs") / filename;
    std::ofstream out(out_path.string());

    out << "# vtk DataFile Version 3.0\n";

    out << "Laplace Solution\n";
    
    out << "ASCII\n";

    out << "DATASET STRUCTURED_POINTS\n";

    out << "DIMENSIONS " << grid.global_n << " " << grid.global_n << " 1\n";

    out << "ORIGIN 0 0 0\n";

    out << "SPACING " << grid.h << " " << grid.h << " 1\n";

    out << "POINT_DATA " << grid.global_n * grid.global_n << "\n";

    out << "SCALARS solution double\n";

    out << "LOOKUP_TABLE default\n";

    for(double v: global_solution){

        out << v << "\n";
    }

}

/**
 * @brief Function to write the solution and grid coordinates to a text file.
 * @param grid The local grid structure containing the solution vectors and grid parameters.
 * @param filename The name for the output text files.
 * @param rank The rank of the current process
 * @param size The total number of processes 
 */
void IO::write_text(
    const Grid& grid,
    const std::string& filename,
    int rank,
    int size
){

    int local_size = grid.local_rows * grid.ny; /// < Number of points in the local grid (excluding ghost cells)

    std::vector<int> recv_counts;               /// < Number of elements to receive from each process
    std::vector<int> displs;                    /// < Displacements for MPI_Gatherv

    if(rank == 0){

        recv_counts.resize(size);
        displs.resize(size);
    }

    MPI_Gather(
        &local_size,
        1,
        MPI_INT,
        recv_counts.data(),
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    std::vector<double> global_solution;

    if(rank == 0){

        int total_size = 0;

        for(int i = 0; i < size; ++i){

            displs[i] = total_size;
            total_size += recv_counts[i];
        }

        global_solution.resize(total_size);
    }

    MPI_Gatherv(
        &grid.u_old[grid.idx(1,0)],
        local_size,
        MPI_DOUBLE,

        global_solution.data(),
        recv_counts.data(),
        displs.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    if (rank != 0){
        return;         /// < Only the root process will write the output file
    }

    std::filesystem::create_directories("Outputs");
    std::filesystem::path out_path = std::filesystem::path("Outputs") / filename;
    std::ofstream out(out_path.string());

    out << "x y solution\n";

    int offset = 0;

    for(int r = 0; r < size; ++r){

        int local_rows = 0;
        int start_row = 0;

        Utils::compute_partitions(
            grid.global_n,
            size,
            r,
            local_rows,
            start_row
        );

        for(int i = 1; i <= local_rows; ++i){

            int global_i = start_row + (i - 1);

            for(int j = 0; j < grid.ny; ++j){

                double x_coord = grid.x(j);
                double y_coord = global_i * grid.h;

                out << x_coord << " "
                    << y_coord << " "
                    << global_solution[offset++] << "\n";

            }
        }
    }

}

/**
 * @brief Function to write the solution and grid coordinates to a csv file.
 * @param grid The local grid structure containing the solution vectors and grid parameters.
 * @param filename The name for the output csv files.
 * @param rank The rank of the current process
 * @param size The total number of processes 
 */
void IO::write_csv(
    const Grid& grid,
    const std::string& filename,
    int rank,
    int size
){

    int local_size = grid.local_rows * grid.ny; /// < Number of points in the local grid (excluding ghost cells)

    std::vector<int> recv_counts;               /// < Number of elements to receive from each process
    std::vector<int> displs;                    /// < Displacements for MPI_Gatherv

    if(rank == 0){

        recv_counts.resize(size);
        displs.resize(size);
    }

    MPI_Gather(
        &local_size,
        1,
        MPI_INT,
        recv_counts.data(),
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    std::vector<double> global_solution;

    if(rank == 0){

        int total_size = 0;

        for(int i = 0; i < size; ++i){

            displs[i] = total_size;
            total_size += recv_counts[i];
        }

        global_solution.resize(total_size);
    }

    MPI_Gatherv(
        &grid.u_old[grid.idx(1,0)],
        local_size,
        MPI_DOUBLE,

        global_solution.data(),
        recv_counts.data(),
        displs.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    if (rank != 0){
        return;         /// < Only the root process will write the output file
    }

    std::filesystem::create_directories("Outputs");
    std::filesystem::path out_path = std::filesystem::path("Outputs") / filename;
    std::ofstream out(out_path.string());

    out << "x, y, solution\n";

    int offset = 0;

    for(int r = 0; r < size; ++r){

        int local_rows = 0;
        int start_row = 0;

        Utils::compute_partitions(
            grid.global_n,
            size,
            r,
            local_rows,
            start_row
        );

        for(int i = 1; i <= local_rows; ++i){

            int global_i = start_row + (i - 1);

            for(int j = 0; j < grid.ny; ++j){

                double x_coord = grid.x(j);
                double y_coord = global_i * grid.h;

                out << x_coord << ","
                    << y_coord << ","
                    << global_solution[offset++] << "\n";

            }
        }
    }

}

/**
 * @brief Function to write the grid values to a VTK file for visualization in serial.
 * @param grid The local grid structure containing the solution vectors and grid parameters.
 * @param filename The name for the output VTK file.
 */
void IO::write_vtk_serial(
    const Grid& grid,
    const std::string& filename
){

    std::filesystem::create_directories("Outputs");
    std::filesystem::path out_path = std::filesystem::path("Outputs") / filename;
    std::ofstream out(out_path.string());

    out << "# vtk DataFile Version 3.0\n";

    out << "Laplace Solution\n";
    
    out << "ASCII\n";

    out << "DATASET STRUCTURED_POINTS\n";

    out << "DIMENSIONS " << grid.global_n << " " << grid.global_n << " 1\n";

    out << "ORIGIN 0 0 0\n";

    out << "SPACING " << grid.h << " " << grid.h << " 1\n";

    out << "POINT_DATA " << grid.global_n * grid.global_n << "\n";

    out << "SCALARS solution double\n";

    out << "LOOKUP_TABLE default\n";

    for(double v: grid.u_old){

        out << v << "\n";
    }
}

/**
 * @brief Function to write the solution and grid coordinates to a text file in serial.
 * @param grid The local grid structure containing the solution vectors and grid parameters.
 * @param filename The name for the output text file.
 */
void IO::write_text_serial(
    const Grid& grid,
    const std::string& filename
){
    std::filesystem::create_directories("Outputs");
    std::filesystem::path out_path = std::filesystem::path("Outputs") / filename;
    std::ofstream out(out_path.string());

    out << "x y solution\n";
    
    for(int i = 0; i < grid.global_n; ++i){

        for(int j = 0; j < grid.ny; ++j){

            double x_coord = grid.x(j);
            double y_coord = grid.y(i);

            out << x_coord << " "
                << y_coord << " "
                << grid.u_old[grid.idx(i,j)] << "\n";

        }
    }
}

/**
 * @brief Function to write the solution and grid coordinates to a csv file in serial.
 * @param grid The local grid structure containing the solution vectors and grid parameters.
 * @param filename The name for the output csv file.
 */
void IO::write_csv_serial(
    const Grid& grid,
    const std::string& filename
){
    std::filesystem::create_directories("Outputs");
    std::filesystem::path out_path = std::filesystem::path("Outputs") / filename;
    std::ofstream out(out_path.string());

    out << "x, y, solution\n";

    for(int i = 0; i < grid.global_n; ++i){

        for(int j = 0; j < grid.ny; ++j){

            double x_coord = grid.x(j);
            double y_coord = grid.y(i);

            out << x_coord << ","
                << y_coord << ","
                << grid.u_old[grid.idx(i,j)] << "\n";

        }
    }
}

/**
 * @brief Function to save data to a dat file for plotting.
 * @param data_Parallel The parallel data to be saved.
 * @param data_Serial The serial data to be saved.
 * @param data_schwarz The Schwarz data to be saved.
 * @param grid_sizes The sizes of the grid.
 * @param filename The name for the output dat file.
 */
void IO::save_data(const std::vector<double>& data_Parallel, const std::vector<double>& data_Serial,
         const std::vector<double>& data_schwarz, const std::vector<int>& grid_sizes, const std::string& filename){

    std::filesystem::create_directories("Test/data");
    std::filesystem::path out_path = std::filesystem::path("Test/data") / filename;
    std::ofstream out(out_path.string());

    for(size_t i = 0; i < data_Parallel.size(); ++i){

        out << grid_sizes[i] << " " << data_Parallel[i] << " " << data_Serial[i] << " " << data_schwarz[i] << "\n";
    }
}