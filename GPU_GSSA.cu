//Gillespie's Direct Stochastic Simulation Algorithm Program
//Parallel NVIDIA GPU Simulation Code
//Final Project for BIOEN 6760, Modeling and Analysis of Biological Networks
//Trevor James Tanner
//Copyright 2013-2015

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

//Error checking code for CUDA-related functions
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

//Rudimentary version of a Hillis-Steele Scan
__global__ void scan(float* inputArray, int n)
{
	extern __shared__ float sdata[];

	int myId = threadIdx.x + blockDim.x*blockIdx.x;
	int tid = threadIdx.x;

	sdata[tid] = inputArray[myId];
	__syncthreads();

	for (int i = 1; i < n; i *= 2)
	{
		if (tid >= i)
		{
			sdata[tid] += sdata[tid - i];
		}
		__syncthreads();
	}

	inputArray[myId] = sdata[tid];

}

//Binary Search Tree - Upper Bound Search
__host__ __device__ int findTarget(float* inputArray, int startingIndex, int endingIndex, float targetValue)
{
	int length = endingIndex - startingIndex;
	if (length > 1)
	{
		int leftSearchIndex = startingIndex + length / 2 + length % 2;
		int rightSearchIndex = endingIndex;
		float leftSearchValue = inputArray[leftSearchIndex];
		float rightSearchValue = inputArray[rightSearchIndex];
		if (leftSearchValue >= targetValue)
		{
			return findTarget(inputArray, startingIndex, leftSearchIndex, targetValue);
		}
		else if (rightSearchValue >= targetValue)
		{
			return findTarget(inputArray, leftSearchIndex + 1, rightSearchIndex, targetValue);
		}
		else
		{
			return -1;
		}
	}
	else if (inputArray[startingIndex] >= targetValue)
	{
		return startingIndex;
	}
	else if (inputArray[endingIndex] >= targetValue)
	{
		return endingIndex;
	}
	else
	{
		return -1;
	}
}

//Initiates Random States for NVIDIA's Random Number Generator (cuRAND)
__global__ void initStates(curandState* globalStateArray, int numTrajectories)
{
	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	while (tId < numTrajectories)
	{
		curand_init((unsigned long long)clock(), tId, 0, &globalStateArray[tId]);
		tId += blockDim.x * gridDim.x;
	}
}


int* get2DIntArray(int arraySizeX, int arraySizeY)
{
	int *returnArray = (int*)malloc(arraySizeX*arraySizeY*sizeof(int));
	return returnArray;
}

int** get2DIntArrayOLD(int arraySizeX, int arraySizeY)
{
	int ** returnArray = (int**)malloc(arraySizeX*sizeof(int*));
	for (int i = 0; i < arraySizeX; ++i)
	{
		returnArray[i] = (int*)malloc(sizeof(int)*arraySizeY);
	}
	return returnArray;
}

//Generates random network for simulation
int** getRandom2DIntArrayOLD(int arraySizeX, int arraySizeY, int inputNumSpecies)
{
	int ** returnArray = get2DIntArrayOLD(arraySizeX, arraySizeY);
	for (int i = 0; i < arraySizeX; ++i)
	{
		returnArray[i][0] = rand() % 3; //reactionType
		if (returnArray[i][0] == 0)
		{
			returnArray[i][5] = -1;
			returnArray[i][6] = 0;
			returnArray[i][7] = 1;
			returnArray[i][8] = 0;
			returnArray[i][1] = rand() % inputNumSpecies; //reactantIndex1
			returnArray[i][2] = 0; //reactantIndex2
			returnArray[i][3] = rand() % inputNumSpecies; //productIndex1
			returnArray[i][4] = 0; //productIndex2
		}
		else if (returnArray[i][0] == 1)
		{
			returnArray[i][5] = -1;
			returnArray[i][6] = -1;
			returnArray[i][7] = 1;
			returnArray[i][8] = 0;
			returnArray[i][1] = rand() % inputNumSpecies; //reactantIndex1
			returnArray[i][2] = rand() % inputNumSpecies; //reactantIndex2
			returnArray[i][3] = rand() % inputNumSpecies; //productIndex1
			returnArray[i][4] = 0; //productIndex2
		}
		else
		{
			returnArray[i][5] = -2;
			returnArray[i][6] = 0;
			returnArray[i][7] = 1;
			returnArray[i][8] = 0;
			returnArray[i][1] = rand() % inputNumSpecies; //reactantIndex1
			returnArray[i][2] = 0; //reactantIndex2
			returnArray[i][3] = rand() % inputNumSpecies; //productIndex1
			returnArray[i][4] = 0; //productIndex2
		}
	}
	return returnArray;
}

void free2DArray(int** inputArray, int arraySizeX)
{
	for (int i = 0; i < arraySizeX; ++i)
	{
		free(inputArray[i]);
	}
	free(inputArray);
}
int * getRandomIntArray(int inputSize, int maxSize)
{
	int* r = (int *)malloc(sizeof(int)*inputSize);
	int i;

	for (i = 0; i < inputSize; ++i)
	{
		r[i] = rand() % maxSize;
	}

	return r;
}

float * getRandomFloatArray(int inputSize)
{
	float* r = (float *)malloc(sizeof(float)*inputSize);
	int i;

	for (i = 0; i < inputSize; ++i)
	{
		r[i] = (float)rand() / float(RAND_MAX);
	}

	return r;
}

void calculatePropensities(float* inputPropensityArray, int* inputSpeciesArray, float* inputKeffArray, int* inputReactantMatrix, int inputReactantMatrixWidth, int inputNumReactants)
{
	for (int i = 0; i < inputNumReactants; i++)
	{
		int reactantType = inputReactantMatrix[i*inputReactantMatrixWidth + 0];
		if (reactantType == 0)
		{
			inputPropensityArray[i] = inputKeffArray[i] * inputSpeciesArray[inputReactantMatrix[i*inputReactantMatrixWidth + 1]];
		}
		else if (reactantType == 1)
		{
			inputPropensityArray[i] = inputKeffArray[i] * inputSpeciesArray[inputReactantMatrix[i*inputReactantMatrixWidth + 1]] * inputSpeciesArray[inputReactantMatrix[i*inputReactantMatrixWidth + 2]];
		}
		else
		{
			inputPropensityArray[i] = inputKeffArray[i] * inputSpeciesArray[inputReactantMatrix[i*inputReactantMatrixWidth + 1]] * (inputSpeciesArray[inputReactantMatrix[i*inputReactantMatrixWidth + 1]] - 1) / 2;
		}
	}
}

__global__ void calculatePropensitiesCUDAv2(float* inputPropensityArray, int* inputSpeciesArray, float* inputKeffArray, int* inputReactantMatrix, int inputReactantMatrixWidth, int inputNumSubReactants, int inputTotalNumReactants, int inputNumSubSpecies)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < inputTotalNumReactants)
	{
		int scaledReactantIndex = tid % inputNumSubReactants;

		int scaledSpeciesFactor = tid / inputNumSubReactants;

		int reactantType = inputReactantMatrix[scaledReactantIndex*inputReactantMatrixWidth + 0];

		if (reactantType == 0)
		{
			inputPropensityArray[tid] = inputKeffArray[scaledReactantIndex] * inputSpeciesArray[(scaledSpeciesFactor*inputNumSubSpecies) + inputReactantMatrix[scaledReactantIndex*inputReactantMatrixWidth + 1]];
		}
		else if (reactantType == 1)
		{
			inputPropensityArray[tid] = inputKeffArray[scaledReactantIndex] * inputSpeciesArray[(scaledSpeciesFactor*inputNumSubSpecies) + inputReactantMatrix[scaledReactantIndex*inputReactantMatrixWidth + 1]] * inputSpeciesArray[(scaledSpeciesFactor*inputNumSubSpecies) + inputReactantMatrix[scaledReactantIndex*inputReactantMatrixWidth + 2]];
		}
		else
		{
			inputPropensityArray[tid] = inputKeffArray[scaledReactantIndex] * inputSpeciesArray[(scaledSpeciesFactor*inputNumSubSpecies) + inputReactantMatrix[scaledReactantIndex*inputReactantMatrixWidth + 1]] * (inputSpeciesArray[(scaledSpeciesFactor*inputNumSubSpecies) + inputReactantMatrix[scaledReactantIndex*inputReactantMatrixWidth + 1]] - 1) / 2;
		}
	}
}

void sumPropensities(float *inputPropensityArray, float *inputSummedPropensityArray, int inputNumReactions)
{
	for (int i = 0; i < inputNumReactions; i++)
	{
		if (i > 0)
		{
			inputSummedPropensityArray[i] = inputSummedPropensityArray[i - 1] + inputPropensityArray[i];
		}
		else
		{
			inputSummedPropensityArray[i] = inputPropensityArray[i];
		}
	}
}

typedef struct tauReactantIndex tauReactantIndex;
struct tauReactantIndex
{
	float tau;
	int reactantIndex;
};

typedef struct inputArrays inputArrays;
struct inputArrays
{
	int* speciesArray;
	float* parameterArray;
	int* reactionMatrix;
	int numSpecies;
	int numReactions;
};

inputArrays readInputFiles()
{
	//Read Species File
	FILE *speciesFile;
	char *mode = "r";
	speciesFile = fopen("speciesArray.txt", mode);

	if (speciesFile == NULL) {
		fprintf(stderr, "Can't open species file!\n");
	}

	const size_t line_size = 300;
	char* line = (char*)malloc(line_size);

	fgets(line, line_size, speciesFile);
	int numSpecies;
	sscanf(line, "# %i rows", &numSpecies);
	int* speciesArray = (int*)malloc(numSpecies*sizeof(int));

	int currentSpecieNumber;
	for (int i = 0; i < numSpecies; i++)
	{
		fgets(line, line_size, speciesFile);
		sscanf(line, "%i", &currentSpecieNumber);
		speciesArray[i] = currentSpecieNumber;
	}

	//Read Parameter File
	FILE *parameterFile;
	parameterFile = fopen("parameterArray.txt", mode);

	if (parameterFile == NULL) {
		fprintf(stderr, "Can't open parameter file!\n");
	}

	int numParameters;
	fgets(line, line_size, parameterFile);
	sscanf(line, "# %i rows", &numParameters);

	float* parameterArray = (float*)malloc(numParameters*sizeof(float));
	float currentParameterValue;
	for (int i = 0; i < numParameters; i++)
	{
		fgets(line, line_size, parameterFile);
		sscanf(line, "%e", &currentParameterValue);
		parameterArray[i] = currentParameterValue;
	}

	//Read ReactionMatrix File
	FILE *reactionMatrixFile;
	reactionMatrixFile = fopen("reactionMatrix.txt", mode);

	if (parameterFile == NULL) {
		fprintf(stderr, "Can't open reaction matrix file!\n");
	}

	int numReactions;
	fgets(line, line_size, reactionMatrixFile);
	sscanf(line, "# %i rows", &numReactions);

	int* reactionMatrixArray = (int*)malloc(numReactions * 9 * sizeof(int));
	int reactionType, reactantIndex1, reactantIndex2, productIndex1, productIndex2, reactantDelta1, reactantDelta2, productDelta1, productDelta2;
	for (int i = 0; i < numReactions; i++)
	{
		fgets(line, line_size, reactionMatrixFile);
		sscanf(line, "%i %i %i %i %i %i %i %i %i", &reactionType, &reactantIndex1, &reactantIndex2, &productIndex1, &productIndex2, &reactantDelta1, &reactantDelta2, &productDelta1, &productDelta2);
		reactionMatrixArray[i * 9 + 0] = reactionType;
		reactionMatrixArray[i * 9 + 1] = reactantIndex1;
		reactionMatrixArray[i * 9 + 2] = reactantIndex2;
		reactionMatrixArray[i * 9 + 3] = productIndex1;
		reactionMatrixArray[i * 9 + 4] = productIndex2;
		reactionMatrixArray[i * 9 + 5] = reactantDelta1;
		reactionMatrixArray[i * 9 + 6] = reactantDelta2;
		reactionMatrixArray[i * 9 + 7] = productDelta1;
		reactionMatrixArray[i * 9 + 8] = productDelta2;
	}

	fclose(parameterFile); fclose(speciesFile); fclose(reactionMatrixFile);
	inputArrays returnInputArrays = {speciesArray,parameterArray,reactionMatrixArray,numSpecies,numReactions};

	return returnInputArrays;
}


void fireReaction(int *inputReactionMatrix, int inputReactionMatrixWidth, int *inputSpeciesMatrix, int inputReactionIndex)
{
	int reactantIndex1 = inputReactionMatrix[inputReactionIndex*inputReactionMatrixWidth + 1];
	int reactantIndex2 = inputReactionMatrix[inputReactionIndex*inputReactionMatrixWidth + 2];
	int reactantIndex3 = inputReactionMatrix[inputReactionIndex*inputReactionMatrixWidth + 3];
	int reactantIndex4 = inputReactionMatrix[inputReactionIndex*inputReactionMatrixWidth + 4];

	int reactantDelta1 = inputReactionMatrix[inputReactionIndex*inputReactionMatrixWidth + 5];
	int reactantDelta2 = inputReactionMatrix[inputReactionIndex*inputReactionMatrixWidth + 6];
	int reactantDelta3 = inputReactionMatrix[inputReactionIndex*inputReactionMatrixWidth + 7];
	int reactantDelta4 = inputReactionMatrix[inputReactionIndex*inputReactionMatrixWidth + 8];

	int end1 = inputSpeciesMatrix[reactantIndex1] + reactantDelta1;
	int end2 = inputSpeciesMatrix[reactantIndex2] + reactantDelta2;
	int end3 = inputSpeciesMatrix[reactantIndex3] + reactantDelta3;
	int end4 = inputSpeciesMatrix[reactantIndex4] + reactantDelta4;

	if ((end1 < 0) || (end2 < 0) || (end3 < 0) || (end4 < 0))
	{
	}
	else
	{
		inputSpeciesMatrix[reactantIndex1] = end1;
		inputSpeciesMatrix[reactantIndex2] = end2;
		inputSpeciesMatrix[reactantIndex3] = end3;
		inputSpeciesMatrix[reactantIndex4] = end4;
	}
}

__device__ void fireReactionCUDA(int *inputReactionMatrix, int inputReactionMatrixWidth, int *inputSpeciesMatrix, int inputReactionIndex)
{
	int reactantIndex1 = inputReactionMatrix[inputReactionIndex*inputReactionMatrixWidth + 1];
	int reactantIndex2 = inputReactionMatrix[inputReactionIndex*inputReactionMatrixWidth + 2];
	int reactantIndex3 = inputReactionMatrix[inputReactionIndex*inputReactionMatrixWidth + 3];
	int reactantIndex4 = inputReactionMatrix[inputReactionIndex*inputReactionMatrixWidth + 4];

	int reactantDelta1 = inputReactionMatrix[inputReactionIndex*inputReactionMatrixWidth + 5];
	int reactantDelta2 = inputReactionMatrix[inputReactionIndex*inputReactionMatrixWidth + 6];
	int reactantDelta3 = inputReactionMatrix[inputReactionIndex*inputReactionMatrixWidth + 7];
	int reactantDelta4 = inputReactionMatrix[inputReactionIndex*inputReactionMatrixWidth + 8];

	int end1 = inputSpeciesMatrix[reactantIndex1] + reactantDelta1;
	int end2 = inputSpeciesMatrix[reactantIndex2] + reactantDelta2;
	int end3 = inputSpeciesMatrix[reactantIndex3] + reactantDelta3;
	int end4 = inputSpeciesMatrix[reactantIndex4] + reactantDelta4;

	if ((end1 < 0) || (end2 < 0) || (end3 < 0) || (end4<0))
	{
	}
	else
	{
		inputSpeciesMatrix[reactantIndex1] = end1;
		inputSpeciesMatrix[reactantIndex2] = end2;
		inputSpeciesMatrix[reactantIndex3] = end3;
		inputSpeciesMatrix[reactantIndex4] = end4;
	}
}

__device__ void fireReactionCUDAv2(int *inputReactionMatrix, int inputReactionMatrixWidth, int *inputSpeciesMatrix, int inputReactionIndex, int inputNumSubReactants, int inputNumSubSpecies)
{
	int scaledReactantIndex = inputReactionIndex % inputNumSubReactants;

	int scaledSpeciesFactor = inputReactionIndex / inputNumSubReactants;

	int reactantIndex1 = scaledSpeciesFactor*inputNumSubSpecies + inputReactionMatrix[scaledReactantIndex*inputReactionMatrixWidth + 1];
	int reactantIndex2 = scaledSpeciesFactor*inputNumSubSpecies + inputReactionMatrix[scaledReactantIndex*inputReactionMatrixWidth + 2];
	int reactantIndex3 = scaledSpeciesFactor*inputNumSubSpecies + inputReactionMatrix[scaledReactantIndex*inputReactionMatrixWidth + 3];
	int reactantIndex4 = scaledSpeciesFactor*inputNumSubSpecies + inputReactionMatrix[scaledReactantIndex*inputReactionMatrixWidth + 4];

	int reactantDelta1 = inputReactionMatrix[scaledReactantIndex*inputReactionMatrixWidth + 5];
	int reactantDelta2 = inputReactionMatrix[scaledReactantIndex*inputReactionMatrixWidth + 6];
	int reactantDelta3 = inputReactionMatrix[scaledReactantIndex*inputReactionMatrixWidth + 7];
	int reactantDelta4 = inputReactionMatrix[scaledReactantIndex*inputReactionMatrixWidth + 8];

	int end1 = inputSpeciesMatrix[reactantIndex1] + reactantDelta1;
	int end2 = inputSpeciesMatrix[reactantIndex2] + reactantDelta2;
	int end3 = inputSpeciesMatrix[reactantIndex3] + reactantDelta3;
	int end4 = inputSpeciesMatrix[reactantIndex4] + reactantDelta4;

	if ((end1 < 0) || (end2 < 0) || (end3 < 0) || (end4 < 0))
	{
		//if the reactions would have caused negative species, do nothing
	}
	else
	{
		inputSpeciesMatrix[reactantIndex1] = end1;
		inputSpeciesMatrix[reactantIndex2] = end2;
		inputSpeciesMatrix[reactantIndex3] = end3;
		inputSpeciesMatrix[reactantIndex4] = end4;
	}
}

__global__ void findTargets(float* inputArray, int numSubElements, int numTrajectories, curandState* globalStateArray, int *inputReactionMatrix, int inputReactionMatrixWidth, int *inputSpeciesMatrix, int inputCurrentTimeStep, float* inputReactionFiredMatrix, int inputNumSubSpecies, int inputNumTimeSteps, float* inputArray2)
{
	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tId < numTrajectories)
	{
		int beginIndex = tId*numSubElements;
		int endIndex = beginIndex + numSubElements - 1;

		float z2 = curand_uniform(&globalStateArray[tId]);
		float propensitySum = inputArray[endIndex];
		float tau = log10(propensitySum) / z2;
		float findMe = propensitySum*z2;
		int foundReactionIndex = findTarget(inputArray, beginIndex, endIndex, findMe);
		
		fireReactionCUDAv2(inputReactionMatrix, inputReactionMatrixWidth, inputSpeciesMatrix, foundReactionIndex, numSubElements, inputNumSubSpecies);

		inputReactionFiredMatrix[tId*inputNumTimeSteps + inputCurrentTimeStep * 2 + 0] = tau; inputReactionFiredMatrix[tId*inputNumTimeSteps + inputCurrentTimeStep * 2 + 1] = foundReactionIndex;

	}
}

tauReactantIndex findReactionToFire(float *inputSummedPropensityArray, int inputNumReactions)
{
	float propensitySum = inputSummedPropensityArray[inputNumReactions - 1];
	float z2 = (float)rand() / float(RAND_MAX);
	float tau = log10(propensitySum) / z2;
	float findMe = propensitySum*z2;

	float *p = std::upper_bound(inputSummedPropensityArray, inputSummedPropensityArray + inputNumReactions - 1, findMe);
	int reactionIndex = p - inputSummedPropensityArray;
	tauReactantIndex returnMe = { tau, reactionIndex };
	return returnMe;
}

int comparator(const void *p, const void*q)
{
	const int *leftArray = *(const int**)p;
	const int *rightArray = *(const int**)q;

	int leftValue = leftArray[0];
	int rightValue = rightArray[0];

	return leftValue - rightValue;
}

void runCPUSimulation(float* inputKeff, int* inputReactionMatrix, int* inputSpecies, int* inputCalcSpecies, int inputNumReactions, int inputNumTimeSteps, int inputNumSpecies, float* inputPropensityArray, float* inputSummedPropensityArray, float* inputReactantFiredMatrix)
{
	for (int i = 0; i < inputNumTimeSteps; ++i)
	{
		calculatePropensities(inputPropensityArray, inputCalcSpecies, inputKeff, inputReactionMatrix, 9, inputNumReactions);
		sumPropensities(inputPropensityArray, inputSummedPropensityArray, inputNumReactions);
		tauReactantIndex tauReactantObject = findReactionToFire(inputSummedPropensityArray, inputNumReactions);
		inputReactantFiredMatrix[i * 2 + 0] = tauReactantObject.tau; inputReactantFiredMatrix[i * 2 + 1] = tauReactantObject.reactantIndex;
		fireReaction(inputReactionMatrix, 9, inputCalcSpecies, tauReactantObject.reactantIndex);
	}

}

void runGPUSimulationv3(float* inputKeff_CUDA, int* inputReactionMatrix_CUDA, int* inputSpecies_CUDA, int* inputCalcSpecies_CUDA, int inputNumReactions, int inputNumTimeSteps, int inputNumSpecies, float* inputPropensityArray_CUDA, float* inputSummedPropensityArray_CUDA, float* inputReactantFiredMatrix_CUDA, float* inputReactantFiredMatrix_HOST, int inputNumTrajectories, curandState* globalStateArray)
{
	int threadsPerBlock = 32;

	for (int j = 0; j < inputNumTimeSteps; ++j)
	{
		calculatePropensitiesCUDAv2 <<<(inputNumTrajectories*inputNumReactions + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >>>(inputPropensityArray_CUDA, inputSpecies_CUDA, inputKeff_CUDA, inputReactionMatrix_CUDA, 9, inputNumReactions, inputNumReactions*inputNumTrajectories, inputNumSpecies);
		scan <<<inputNumTrajectories, inputNumReactions, inputNumReactions*sizeof(float) >>>(inputPropensityArray_CUDA, inputNumReactions);
		findTargets <<<(inputNumTrajectories + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >>>(inputPropensityArray_CUDA, inputNumReactions, inputNumTrajectories, globalStateArray, inputReactionMatrix_CUDA, 9, inputSpecies_CUDA, j, inputReactantFiredMatrix_CUDA, inputNumSpecies, inputNumTimeSteps, inputPropensityArray_CUDA);
	}

	gpuErrchk(cudaMemcpy(inputReactantFiredMatrix_HOST, inputReactantFiredMatrix_CUDA, inputNumTrajectories* inputNumTimeSteps * 2 * sizeof(float), cudaMemcpyDeviceToHost));

}

__global__ void warmUp()
{
}

int * flatten2DArray(int** input2DArray, int inputSizeX, int inputSizeY)
{
	int * returnArray = get2DIntArray(inputSizeX, inputSizeY);
	for (int i = 0; i < inputSizeX; ++i)
	{
		for (int j = 0; j < inputSizeY; ++j)
		{
			returnArray[i*inputSizeY + j] = input2DArray[i][j];
		}
	}
	return returnArray;
}

void printTimings(bool inputReadFile, int inputNumRandomReactions, int inputNumRandSpecies, int inputNumTimeSteps, int inputNumSimulations)
{
	clock_t begin_CPU, end_CPU, begin_GPU, end_GPU;
	float time_spent_GPU, time_spent_CPU;

	float *kEff;
	int *reactionMatrix;
	int *species;
	inputArrays inputArraysRead;

	int numSimulations = inputNumSimulations;
	int numTimeSteps = inputNumTimeSteps;

	int numReactions;
	int numSpecies;

	if (inputReadFile == true)
	{
		inputArraysRead = readInputFiles();
		numReactions = inputArraysRead.numReactions;
		numSpecies = inputArraysRead.numSpecies;
		reactionMatrix = inputArraysRead.reactionMatrix;
		species = inputArraysRead.speciesArray;
		kEff = inputArraysRead.parameterArray;
	}
	else
	{
		numReactions = inputNumRandomReactions;
		numSpecies = inputNumRandSpecies;
		species = getRandomIntArray(numSpecies, 100);
		kEff = getRandomFloatArray(numReactions);
		int **reactionMatrixOLD = getRandom2DIntArrayOLD(numReactions, 9, numSpecies);
		qsort(reactionMatrixOLD, numReactions, sizeof(int), comparator); //Sort the array to make branch prediction work
		reactionMatrix = flatten2DArray(reactionMatrixOLD, numReactions, 9);
		free2DArray(reactionMatrixOLD, numSpecies);
	}

	printf("readFile:%d numReactions:%i numSpecies:%i numTimeSteps:%i numSimulations:%i\n", inputReadFile, numReactions, numSpecies, numTimeSteps, numSimulations);

	//These guys will always be changing
	int* calcSpecies = (int *)malloc(sizeof(int)*numSpecies);
	std::copy(species, species + numSpecies, calcSpecies);
	float *propensityArray = (float *)malloc(sizeof(float)*numReactions); //initially empty
	float *summedPropensityArray = (float *)malloc(sizeof(float)*numReactions); //initially empty
	//OUTPUT
	float *reactantFiredMatrix = (float *)malloc(numTimeSteps * 2 * sizeof(float)); //column1=time,column2=reactionFired

	//INPUTS SPECIFICALLY FOR GPU SIMULATION (some of the CPU inputs are reused)
	int* species_HOST = (int *)malloc(sizeof(int)*numSpecies*numSimulations);
	int* calcSpecies_HOST = (int *)malloc(sizeof(int)*numSpecies*numSimulations);
	float *propensityArray_HOST = (float *)malloc(sizeof(float)*numReactions*numSimulations);
	float *summedPropensityArray_HOST = (float *)malloc(sizeof(float)*numReactions*numSimulations);
	float* reactantFiredMatrix_HOST = (float *)malloc(numSimulations*numTimeSteps * 2 * sizeof(float));
	for (int l = 0; l < numSimulations; l++)
	{
		for (int k = 0; k < numReactions; k++)
		{
			propensityArray_HOST[l*numReactions + k] = propensityArray[k];
		}
		for (int m = 0; m < numSpecies; m++)
		{
			species_HOST[l*numSpecies + m] = species[m];
			calcSpecies_HOST[l*numSpecies + m] = species[m];
		}
	}

	//CUDA Variable Versions
	float *kEffCUDA;
	int *reactionMatrixCUDA;
	int *speciesCUDA;
	int *calcSpeciesCUDA;
	float *propensityArrayCUDA;
	float *summedPropensityArrayCUDA;
	float *reactantFiredMatrixCUDA;

	//Make Device Pointers
	gpuErrchk(cudaMalloc(&reactionMatrixCUDA, numReactions * 9 * sizeof(int)));
	gpuErrchk(cudaMalloc(&kEffCUDA, numReactions*sizeof(float)));
	gpuErrchk(cudaMalloc(&speciesCUDA, numSimulations*numSpecies*sizeof(int)));
	gpuErrchk(cudaMalloc(&calcSpeciesCUDA, numSimulations*numSpecies*sizeof(int)));
	gpuErrchk(cudaMalloc(&propensityArrayCUDA, numSimulations*numReactions*sizeof(float)));
	gpuErrchk(cudaMalloc(&summedPropensityArrayCUDA, numSimulations*numReactions*sizeof(float)));
	gpuErrchk(cudaMalloc(&reactantFiredMatrixCUDA, numSimulations*numTimeSteps * 2 * sizeof(float)));

	//Copy Data to Device
	gpuErrchk(cudaMemcpy(reactionMatrixCUDA, reactionMatrix, numReactions * 9 * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(kEffCUDA, kEff, numReactions*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(speciesCUDA, species_HOST, numSimulations*numSpecies*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(calcSpeciesCUDA, calcSpecies_HOST, numSimulations*numSpecies*sizeof(int), cudaMemcpyHostToDevice));

	printf("Starting!\n");
	//GPU Timing
	warmUp << <1, 1 >> >();
	curandState* globalStateArrayInput;
	gpuErrchk(cudaMalloc(&globalStateArrayInput, numSimulations * sizeof(curandState)));
	int threadsPerBlock = 32;
	initStates << <(numSimulations + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(globalStateArrayInput, numSimulations);

	begin_GPU = clock();
	runGPUSimulationv3(kEffCUDA, reactionMatrixCUDA, speciesCUDA, calcSpeciesCUDA, numReactions, numTimeSteps, numSpecies, propensityArrayCUDA, summedPropensityArrayCUDA, reactantFiredMatrixCUDA, reactantFiredMatrix_HOST, numSimulations, globalStateArrayInput);

	cudaDeviceSynchronize();
	end_GPU = clock();

	printf("Ending!\n");

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}

	time_spent_GPU = (float)(end_GPU - begin_GPU) / CLOCKS_PER_SEC;

	float avg_GPU = (time_spent_GPU) / numSimulations;
	printf("Avg. GPU Simulation Time: %.17g [sim/sec]\n", avg_GPU);
	cudaFree(reactionMatrixCUDA); cudaFree(kEffCUDA); cudaFree(speciesCUDA); cudaFree(calcSpeciesCUDA); cudaFree(propensityArrayCUDA); cudaFree(summedPropensityArrayCUDA); cudaFree(reactantFiredMatrixCUDA); cudaFree(globalStateArrayInput);
	free(species_HOST); free(calcSpecies_HOST); free(propensityArray_HOST); free(summedPropensityArray_HOST); free(reactantFiredMatrix_HOST);
	cudaDeviceSynchronize();
	cudaDeviceReset();

	////CPU Timing
	//begin_CPU = clock();

	//for (int j = 0; j < numSimulations; ++j)
	//{
	//	runCPUSimulation(kEff, reactionMatrix, species, calcSpecies, numReactions, numTimeSteps, numSpecies, propensityArray, summedPropensityArray, reactantFiredMatrix);
	//}

	//end_CPU = clock();

	////Clean-up 
	free(kEff); free(species); free(calcSpecies);  free(reactionMatrix); free(propensityArray); free(summedPropensityArray); free(reactantFiredMatrix);

	//time_spent_CPU = (float)(end_CPU - begin_CPU) / CLOCKS_PER_SEC;
	//float avg_CPU = time_spent_CPU / numSimulations;
	//printf("Avg. CPU Simulation Time: %.17g [sim/sec]\n", avg_CPU);
	//printf("CPU/GPU Diff:%.17g\n", avg_CPU / avg_GPU);
}

int main(int argc, char** argv)
{
	printTimings(true, 1024, 1024, 10000, 1000);

	int numSpeciesReactions = 1;
	for (int i = 1; i <= 11; i++)
	{
		numSpeciesReactions *= 2;
		printTimings(false, numSpeciesReactions / 2, numSpeciesReactions / 2, 10000, 1000);
	}

	return 0;
}