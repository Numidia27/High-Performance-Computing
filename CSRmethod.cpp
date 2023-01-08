/** Parallel Program - Sparse Matrix - Fat Vector Multiplication */
/** using CSR format and Broadcast communication */

#include <mpi.h>
#include <stdio.h>
#include<iostream>
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
	* nzCount store the number of non-zero elements of the sparse matrix
	*/
	int nz_count = 0;

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
	* define the three rows of the CSR matrix. @see report how the CSR is constructed
	*/
	const int CSR_NB_ROWS = 3;

	/*
	* define the number of columns of the CSR matrix. the max value we use in the test is 10000,
	* in a sparse matrix the number of zero elements > non-zero elements, 10000 is enough space to store the values
	*/
	const int CSR_NB_COLS = 20000;

	/*
	* define and initialise the CSR matrix CSR
	* the size of matrix CSR changes when the size of the sparse matrix sparseMatrix and the Fat-vector fatVector change
	*/
	int matrixCSR[CSR_NB_ROWS][CSR_NB_COLS] = {};

	/*
	* define and initiase the size of the result matrix of size MATRIX_NB_ROWS * VECTOR_NB_COLS
	*/
	int resultMatrix[MATRIX_NB_ROWS][VECTOR_NB_COLS] = {};

	/*
	* define the rank and number of processes 
	*/
	int procs_rank, nb_procs;

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
	srand(time(NULL));

	/* 
	* start the wall clock
	*/
	start = MPI_Wtime();

	// PROCESSOR 0 - MASTER NODE
	/* 
	* the job start at the process 0, acting as a master nodeand distribute tasks to the worker nodes
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
					nz_count++;
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
				std::cout << "   " << sparseMatrix[i][j] << "	";
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
				std::cout << "   " << fatVector[i][j] << "	";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;

		// CONSTRUCT THE CSR MATRIX
		/*
		* n is the number of columns needed to store all the non-zero elements
		* matrixCSR[0][n] is the row index that store the row of the non-zero value of the sparse matrix
		* matrixCSR[1][n] is the column index of the non-zero values of the sparse matrix
		* matrixCSR[2][n] is the non-zero elements. @see report how CSR matrix constructed
		*/
		int n = 0;
		for (int i = 0; i < MATRIX_NB_ROWS; i++) {
			for (int j = 0; j < MATRIX_NB_COLS; j++) {
				if (sparseMatrix[i][j] != 0) {
					matrixCSR[0][n] = i;
					matrixCSR[1][n] = j;
					matrixCSR[2][n] = sparseMatrix[i][j];
					n++;
				}
			}
		}

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
	* broadcast the fat vector, number of non-zero elements, and CSR matrix from procs 0 to all worker nodes
	* and set each process to wait until all processes receive all the broadcasted elements before proceeding further
	*/
	MPI_Bcast(&fatVector, VECTOR_NB_ROWS * VECTOR_NB_COLS, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Bcast(&nz_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Bcast(&matrixCSR, CSR_NB_ROWS * CSR_NB_COLS, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	/*
	* end Timer for the communication
	*/
	end_comm = MPI_Wtime();

	/*
	* Start Timer for the computation for all processes
	*/
	start_computation = MPI_Wtime();

	/*
	* m % nb_procs allow to distribute the number of the matrix rows equitably on the number of processors
	* each processor compute a part of the matrix
	*/
	for (int m = 0; m < MATRIX_NB_ROWS; m++) {
		if (m % nb_procs == procs_rank) {
			for (int k = 0; k < VECTOR_NB_COLS; k++) {
				resultMatrix[m][k] = 0;
				for (int j = 0; j < nz_count; j++) {
					if (matrixCSR[0][j] == m) {
						resultMatrix[m][k] += matrixCSR[2][j] * fatVector[matrixCSR[1][j]][k];
					}
				}
			}
		}
	}

	/*
	* end Timer for the computation
	*/
	end_computation = MPI_Wtime();
	std::cout << std::endl;

	/*
	* print communication time for each processor
	*/
	std::cout << "Process Rank: " << procs_rank << "\t" << "Communication Time: " << (end_comm - start_comm) << std::endl;
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
	std::cout << std::endl;
	for (int m = 0; m < MATRIX_NB_ROWS; m++) {
		for (int j = 0; j < VECTOR_NB_COLS; j++) {
			std::cout << "   " << resultMatrix[m][j] << "	";
		}
		std::cout << std::endl;
	}	

	MPI_Finalize();
	
	return 0;
}
