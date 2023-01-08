/** Serial Program - Sparse Matrix - Fat Vector Multiplication */

#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <iomanip>
#include <fstream>
#include <chrono>

int main(int argc, char** argv) {

	/*
	* define the number of rowsand columns of the sparse matrix as constants
	*/
	const int MATRIX_NB_ROWS = 7;
	const int MATRIX_NB_COLS = 5;

	/*
	* define the sparse matrix
	*/
	int sparseMatrix[MATRIX_NB_ROWS][MATRIX_NB_COLS] = {};

	/*
	* define the number of rows and columns of the vector as constants
	*/
	const int VECTOR_NB_ROWS = 5;
	const int VECTOR_NB_COLS = 3;

	/*
	* define the fat vector
	*/
	int fatVector[VECTOR_NB_ROWS][VECTOR_NB_COLS] = {};

	/*
	* define and initiase the size of the result matrix of size MATRIX_NB_ROWS * VECTOR_NB_COLS
	*/
	int resultMatrix[MATRIX_NB_ROWS][VECTOR_NB_COLS] = {};

	/*
	* define the rank and number of processes
	*/
	int procs_rank, nb_procs;
	MPI_Status status;

	/*
	* define variable to compute the communication and computation time
	*/
	double start, end, start_timer, end_timer, start_comm, end_comm, start_computation, end_computation;

	/*
	* initialise the MPI Environment
	*/
	MPI_Init(&argc, &argv);

	/*
	* define the communicator
	*/
	MPI_Comm_size(MPI_COMM_WORLD, &nb_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procs_rank);

	/*
	* initialise the Timer
	*/
	//srand(time(NULL));

	/*
	* start the wall clock
	*/
	start = MPI_Wtime();

	// PROCESSOR 0 - MASTER NODE
	/*
	* the job start at the process 0, acting as a master node and distribute tasks to the worker nodes
	*/
	if (procs_rank == 0) {
		/*
		* job begins for procs 0, start the wall clock
		*/
		start_timer = MPI_Wtime();

		/*
		* generate a random sparse matrix. random numbers generated between 1 and 10
		*/
		for (int i = 0; i < MATRIX_NB_ROWS; i++) {
			for (int j = 0; j < MATRIX_NB_COLS; j++) {
				int rand_Number = (rand() % 10) + 1;
				if (rand_Number > 3) {
					sparseMatrix[i][j] = 0;
				}
				else {
					sparseMatrix[i][j] = (rand() % 10) + 1;
				}
			}
		}

		/*
		* print the sparse matrix in 2D format
		*/
		std::cout << "The generated sparse matrix: " << std::endl;
		std::cout << std::endl;
		for (int i = 0; i < MATRIX_NB_ROWS; i++) {
			for (int j = 0; j < MATRIX_NB_COLS; j++) {
				std::cout << sparseMatrix[i][j] << "  ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;

		/*
		* generate a random fat vector. random numbers generated between 1 and 10
		*/
		for (int i = 0; i < VECTOR_NB_ROWS; i++) {
			for (int j = 0; j < VECTOR_NB_COLS; j++) {
				int rand_Number = (rand() % 10) + 1;
				fatVector[i][j] = (rand() % 10) + 1;
			}
		}

		/*
		* print the fat vector in 2D format
		*/
		std::cout << "The generated fat vector: " << std::endl;
		std::cout << std::endl;
		for (int i = 0; i < VECTOR_NB_ROWS; i++) {
			for (int j = 0; j < VECTOR_NB_COLS; j++) {
				std::cout << fatVector[i][j] << "  ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;

		// end of Timer for procs 0
		end_timer = MPI_Wtime();
	}
	std::cout << std::endl;

	// START PARALLEL COMPUTING. DISTRIBUTE TASKS TO WORKER NODES

	/*
	* start the Timer of the communication
	*/
	start_comm = MPI_Wtime();

	/*
	* Start Timer for the computation for all processes
	*/
	start_computation = MPI_Wtime();

	/*
	* sparse matrix fat-vector multiplication
	* each processor compute a certain number of rows of the sparse matrix
	*/
	for (int m = 0; m < MATRIX_NB_ROWS; m++) {
			for (int k = 0; k < VECTOR_NB_COLS; k++) {
				for (int n = 0; n < VECTOR_NB_ROWS; n++) {
					resultMatrix[m][k] += sparseMatrix[m][n] * fatVector[n][k];
				}
			}
	}

	/*
	* end Timer for the computation
	*/
	end_computation = MPI_Wtime();
	std::cout << std::endl;

	/*
	* print computation time for each processor
	*/
	std::cout << "Process Rank: " << procs_rank << "\t" << "Computation Time : " << (end_computation - start_computation) << std::endl;
	std::cout << std::endl;

	/*
	* print the Elapsed time
	*/
	end = MPI_Wtime();
	std::cout << "Process Rank: " << procs_rank << "\t" << "Elapsed time = " << (end - start) << "	";
	std::cout << std::endl;
	std::cout << std::endl;

	/*
	* print the result of the multiplication
	*/
	//std::cout << "The result of the sparse matrix fat-vector multiplication using CSR format: " << std::endl;
	std::cout << "The multiplication result: " << std::endl;
	for (int m = 0; m < MATRIX_NB_ROWS; m++) {
		for (int j = 0; j < VECTOR_NB_COLS; j++) {
			std::cout << resultMatrix[m][j] << "	";
		}
		std::cout << std::endl;
	}

	MPI_Finalize();

	return 0;
}