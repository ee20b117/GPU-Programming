// CS6023 Assignment 2 (Efficienty compute AB+CDt)
// Jan-may 2023

// Author: Saravnan S V
// Roll Number: EE20B117
// Last modified: 04-03-2023

#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;

#define tileDimx 32 // 32 is chosen for coalesced access
#define tileDimy 32 

// kernel tiledSOP (SumOfProducts)
__global__ void tiledSOP(int p, int q, int r, int *d_matrixA, int *d_matrixB, int *d_matrixC, int *d_matrixD, int *d_matrixE){
	// Exploiting shared memory
	__shared__ int tileA[tileDimx][tileDimy]; 
	__shared__ int tileB[tileDimx][tileDimy]; 
	__shared__ int tileC[tileDimx][tileDimy]; 
	__shared__ int tileD[tileDimx][tileDimy]; 

	// Giving alias names to blockIds and threadIds
	unsigned tIdx = threadIdx.x;
	unsigned tIdy = threadIdx.y;
	unsigned bIdx = blockIdx.x;
	unsigned bIdy = blockIdx.y;

    // Calculating the row and column of the thread with bIdx, bIdy, tIdy, tIdx
    unsigned row = bIdx * tileDimx + tIdx;
    unsigned column = bIdy * tileDimy + tIdy;

    // Taking turns to fill in the shared memory, do the computations and accumulate in the temp variable
    int temp = 0; // to accumulate computations in the loop and finally into the result matrix, E (d_matrixE)
    for (int i=0; i<ceil((float) q / tileDimx); i++){
      if ((row<p) && ((i * tileDimx + tIdy)<q)) 
      tileA[tIdx][tIdy] = d_matrixA[row * q + i * tileDimx + tIdy]; // d_matrixA[row][i * tileDimx + tIdy];
      else tileA[tIdx][tIdy] = 0;

      if (((i * tileDimy + tIdx)<q) && (column<r))
      tileB[tIdx][tIdy] = d_matrixB[(i * tileDimy + tIdx) * r + column]; // d_matrixB[i * tileDimy + tIdx][column];
      else tileB[tIdx][tIdy] = 0;

      if ((row<p) && ((i * tileDimx + tIdy)<q))
      tileC[tIdx][tIdy] = d_matrixC[row * q + i * tileDimx + tIdy]; // d_matrixC[row][i * tileDimx + tIdy];
      else tileC[tIdx][tIdy] = 0;
      
      if (((i * tileDimy + tIdx)<q) && (column<r))
      tileD[tIdx][tIdy] = d_matrixD[column * q + i * tileDimy + tIdx]; // d_matrixDtranspose[i * tileDimy + tIdx][column];
      else tileD[tIdx][tIdy] = 0;

      __syncthreads();

      for (int k=0; k<tileDimx; k++){
        temp += tileA[tIdx][k] * tileB[k][tIdy];
        temp += tileC[tIdx][k] * tileD[k][tIdy];
        __syncthreads();
      }
    }

    // Storing the result
    if (row<p && column<r)
    d_matrixE [row * r + column] = temp;
}

// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */
	// Launch configuration
 	dim3 grid1(ceil((float) p / tileDimx), ceil((float)r / tileDimy), 1);
	dim3 block1(tileDimx, tileDimy, 1);
	// Launching the kernel tiledSOP (SumOfProducts)
	tiledSOP<<<grid1, block1>>>(p, q, r, d_matrixA, d_matrixB, d_matrixC, d_matrixD, d_matrixE);
	/* ****************************************************************** */

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken bIdy the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
	
