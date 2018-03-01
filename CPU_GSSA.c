//Gillespie's Direct Stochastic Simulation Algorithm Program
//Single-Core CPU Simulation Code
//Final Project for BIOEN 6760, Modeling and Analysis of Biological Networks
//Trevor James Tanner
//Copyright 2013-2015

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

//Binary Search Tree - Upper Bound Search
int findTarget(float* inputArray, int startingIndex, int endingIndex, float targetValue)
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

void calculatePropensities(float* inputPropensityArray, int* inputSpeciesArray, float* inputKeffArray, int* inputReactantMatrix, int inputReactantMatrixWidth, int inputNumReactants, int oneStart, int twoStart, int threeStart)
{
	int numOnes = twoStart - oneStart;
	int numTwos = threeStart - twoStart;
	int numThrees = inputNumReactants - threeStart;
	for (int i = 0; i < numOnes; i++)
	{
		inputPropensityArray[i] = inputKeffArray[i] * inputSpeciesArray[inputReactantMatrix[i*inputReactantMatrixWidth + 1]];
	}
	for (int i = numOnes; i < (numOnes + numTwos); i++)
	{
		inputPropensityArray[i] = inputKeffArray[i] * inputSpeciesArray[inputReactantMatrix[i*inputReactantMatrixWidth + 1]] * inputSpeciesArray[inputReactantMatrix[i*inputReactantMatrixWidth + 2]];
	}
	for (int i = numOnes + numTwos; i < (numOnes + numTwos + numThrees); i++)
	{
		inputPropensityArray[i] = inputKeffArray[i] * inputSpeciesArray[inputReactantMatrix[i*inputReactantMatrixWidth + 1]] * (inputSpeciesArray[inputReactantMatrix[i*inputReactantMatrixWidth + 1]] - 1) / 2;
	}

	//OLD CODE - new code unrolls for the loops so Intel compiler can auto-vectorize
	//for (int i = 0; i < inputNumReactants; i++)
	//{
	//	int reactantType = inputReactantMatrix[i*inputReactantMatrixWidth + 0];
	//	if (reactantType == 0)
	//	{
	//		inputPropensityArray[i] = inputKeffArray[i] * inputSpeciesArray[inputReactantMatrix[i*inputReactantMatrixWidth + 1]];
	//	}
	//	else if (reactantType == 1)
	//	{
	//		inputPropensityArray[i] = inputKeffArray[i] * inputSpeciesArray[inputReactantMatrix[i*inputReactantMatrixWidth + 1]] * inputSpeciesArray[inputReactantMatrix[i*inputReactantMatrixWidth + 2]];
	//	}
	//	else
	//	{
	//		inputPropensityArray[i] = inputKeffArray[i] * inputSpeciesArray[inputReactantMatrix[i*inputReactantMatrixWidth + 1]] * (inputSpeciesArray[inputReactantMatrix[i*inputReactantMatrixWidth + 1]] - 1) / 2;
	//	}
	//}
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
	inputArrays returnInputArrays = { speciesArray, parameterArray, reactionMatrixArray, numSpecies, numReactions };

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

void runCPUSimulation(float* inputKeff, int* inputReactionMatrix, int* inputSpecies, int* inputCalcSpecies, int inputNumReactions, int inputNumTimeSteps, int inputNumSpecies, float* inputPropensityArray, float* inputSummedPropensityArray, float* inputReactantFiredMatrix, int inputOneIndex, int inputTwoIndex, int inputThreeIndex)
{
	for (int i = 0; i < inputNumTimeSteps; ++i)
	{
		calculatePropensities(inputPropensityArray, inputCalcSpecies, inputKeff, inputReactionMatrix, 9, inputNumReactions, inputOneIndex, inputTwoIndex,inputThreeIndex);
		sumPropensities(inputPropensityArray, inputSummedPropensityArray, inputNumReactions);
		tauReactantIndex tauReactantObject = findReactionToFire(inputSummedPropensityArray, inputNumReactions);
		inputReactantFiredMatrix[i * 2 + 0] = tauReactantObject.tau; inputReactantFiredMatrix[i * 2 + 1] = tauReactantObject.reactantIndex;
		fireReaction(inputReactionMatrix, 9, inputCalcSpecies, tauReactantObject.reactantIndex);
	}

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
	clock_t begin_CPU, end_CPU;
	float time_spent_CPU;

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

	int* reactionTypes = (int*)malloc(sizeof(int)*numReactions);
	for (int i = 0; i < numReactions; i++)
	{
		reactionTypes[i] = reactionMatrix[i * 9];
		//printf("%i\n", reactionTypes[i]);
	}
	int oneStart = (std::lower_bound(reactionTypes, reactionTypes + numReactions, 1) - reactionTypes);
	int twoStart = (std::lower_bound(reactionTypes, reactionTypes + numReactions, 2) - reactionTypes);
	int threeStart = (std::lower_bound(reactionTypes, reactionTypes + numReactions, 3) - reactionTypes);
	//int fourIndex = (std::lower_bound(reactionTypes, reactionTypes + numReactions, 4) - reactionTypes);
	//printf("1:%i 2:%i 3:%i 4:%i", oneIndex,twoIndex,threeIndex,fourIndex);

	//These guys will always be changing
	int* calcSpecies = (int *)malloc(sizeof(int)*numSpecies);
	std::copy(species, species + numSpecies, calcSpecies);
	float *propensityArray = (float *)malloc(sizeof(float)*numReactions); //initially empty
	float *summedPropensityArray = (float *)malloc(sizeof(float)*numReactions); //initially empty
	//OUTPUT
	float *reactantFiredMatrix = (float *)malloc(numTimeSteps * 2 * sizeof(float)); //column1=time,column2=reactionFired

	//CPU Timing
	begin_CPU = clock();

	for (int j = 0; j < numSimulations; ++j)
	{
		runCPUSimulation(kEff, reactionMatrix, species, calcSpecies, numReactions, numTimeSteps, numSpecies, propensityArray, summedPropensityArray, reactantFiredMatrix, oneStart,twoStart,threeStart);
	}

	end_CPU = clock();

	//Clean-up 
	free(kEff); free(species); free(calcSpecies);  free(reactionMatrix); free(propensityArray); free(summedPropensityArray); free(reactantFiredMatrix);

	time_spent_CPU = (float)(end_CPU - begin_CPU) / CLOCKS_PER_SEC;
	float avg_CPU = time_spent_CPU / numSimulations;
	printf("Avg. CPU Simulation Time: %.17g [sim/sec]\n", avg_CPU);
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
