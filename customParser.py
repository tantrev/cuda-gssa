#Gillespie's Direct Stochastic Simulation Algorithm Program
#Parser to convert .PSC files to arrays/matrices (speciesArray, parameterArray, and reactionMatrix) suitable for the main simulation code
#Note: the example .psc file was generated from a SBML .xml file using StochPy
#Final Project for BIOEN 6760, Modeling and Analysis of Biological Networks
#Trevor James Tanner
#Copyright 2013-2015

import re
import numpy as np
import itertools
import pandas as pd

filer = "BIOMD0000000504.xml.psc"

data = open(filer)

def convertReactants(reactants):
    newReactantList = list()
    for reactant in reactants:
        if '>' not in reactant and ('+' not in reactant and '-' not in reactant):
            if '{' in reactant:
                m=re.match("{[0-9]*.[0-9]*}",reactant)
                num = float(m.group()[1:-1])
                reactant = reactant[m.span()[1]:]
                newReactantList.append((num,reactant))
            else:
                newReactantList.append((1,reactant))
        else:
            newReactantList.append(reactant)
    return newReactantList
                
                
def convertMe(line, reactants):
    reactants = convertReactants(reactants)
    flatReactants = list(itertools.chain(*reactants))
    line = line.strip()
    numMultiply = len(re.findall("\*",line))
    equalsLocation = 0
    multiplyIter = re.finditer("\*",line)
    if(numMultiply==1):
        for a in multiplyIter:
            multiplyLocation = a.start()
            cVal = line[equalsLocation:multiplyLocation]
            specieName = line[multiplyLocation+1:len(line)]
            if 'Sink' in flatReactants:
                return (1,cVal,-reactants[0][0],reactants[0][1],0,0,0,reactants[2][1],0,0)
            elif 'Source' in flatReactants:
                return (1,cVal,-0,reactants[0][1], 0,0,reactants[2][0],reactants[2][1],0,0)
            elif '$pool' in flatReactants:
                return (1,cVal,-0,reactants[0][1], 0,0,reactants[2][0],reactants[2][1],0,0)
            elif len(reactants)==5:
                return (1,cVal,-reactants[0][0],reactants[0][1],0,0, reactants[2][0],reactants[2][1],reactants[4][0],reactants[4][1])
            else:
                return (1,cVal,-reactants[0][0],reactants[0][1],0,0, reactants[2][0],reactants[2][1],0,0)
    elif(numMultiply==2):
        for i,a in enumerate(multiplyIter):
            if i==0:
                multiplyLocation1 = a.start()
                cVal = line[equalsLocation:multiplyLocation1]
            elif i==1:
                multiplyLocation2 = a.start()
                specie1 = line[multiplyLocation1:multiplyLocation2][1:]
                specie2 = line[multiplyLocation2:len(line)][1:]
                if len(reactants)==3:
                    if specie1 not in flatReactants:
                        return (2,cVal,-reactants[0][0],reactants[0][1],0,specie1,reactants[2][0],reactants[2][1],0,0)
                    elif specie2 not in flatReactants:
                        return (2,cVal,-reactants[0][0],reactants[0][1],0,specie2,reactants[2][0],reactants[2][1],0,0)
                reactionArrowIndex = reactants.index('>')
                if reactionArrowIndex==3:
                    return (2,cVal,-reactants[0][0],reactants[0][1],-reactants[2][0],reactants[2][1],reactants[4][0],reactants[4][1],0,0)
                elif reactionArrowIndex==1:
                    if specie1 not in flatReactants:
                        return (2,cVal,-reactants[0][0],reactants[0][1],0,specie1,reactants[2][0],reactants[2][1],reactants[4][0],reactants[4][1])
                    elif specie2 not in flatReactants:
                        return (2,cVal,-reactants[0][0],reactants[0][1],0,specie2,reactants[2][0],reactants[2][1],reactants[4][0],reactants[4][1])
    elif(numMultiply==3):
        for i,a in enumerate(multiplyIter):
            if i==0:
                multiplyLocation1 = a.start()
                cVal = line[equalsLocation:multiplyLocation1]
            elif i==1:
                multiplyLocation2 = a.start()
                minusOneLocation = line.find("-1.0")
                specie1 = line[multiplyLocation1:multiplyLocation2][1:]
                convertReactants(reactants)
                return (3,cVal,-reactants[0][0],reactants[0][1],0,0,reactants[2][0],reactants[2][1],0,0)

reactionList = list()
specieDict = dict()
parameterDict = dict()

reactionFlag = bool()
fixedSpeciesFlag = bool()
variableSpeciesFlag = bool()
parameterFlag = bool()
targetLine1 = int()
targetLine2 = int()
subLine = str()

for i,line in enumerate(data):
    if "# Reactions" in line:
        reactionFlag = True
    if "# Fixed species" in line:
        fixedSpeciesFlag = True
    if "# Variable species" in line:
        variableSpeciesFlag = True
    if "# Parameters" in line:
        parameterFlag = True
        
    if reactionFlag==True and fixedSpeciesFlag==False:
        if ":" in line and "#" not in line:
            targetLine1 = i+1
            targetLine2 = i+2
        elif i==targetLine1:
            subLine = line.strip()
        elif i==targetLine2:
            reactionList.append(convertMe(line,subLine.split(' ')))
    elif variableSpeciesFlag==True and parameterFlag==False:
        if "=" in line:
            subLine = line.strip()
            equalsLocation = subLine.find("=")
            specie = subLine[0:equalsLocation-1].split('@')[0]
            specieQuantity = subLine[equalsLocation+2:len(subLine)]
            specieDict[specie] = float(specieQuantity)
    elif parameterFlag==True:
        if "=" in line:
            subLine = line.strip()
            equalsLocation = subLine.find("=")
            parameter = subLine[0:equalsLocation-1]
            parameterValue = subLine[equalsLocation+2:len(subLine)]
            parameterDict[parameter] = float(parameterValue)

specieDict['Source'] = 1;
specieDict['Sink'] = 1;
specieDict['$pool'] = 1;

def findIndex(inputReactant,inputReactantDict):
    if(inputReactant == 0):
        return 0
    else:
        return inputReactantDict.keys().index(inputReactant)

reactionMatrixList = list()
for reaction in reactionList:
    reactionMatrixList.append((reaction[0],findIndex(reaction[1],parameterDict),reaction[2],findIndex(reaction[3],specieDict),reaction[4],findIndex(reaction[5],specieDict),reaction[6],findIndex(reaction[7],specieDict),reaction[8],findIndex(reaction[9],specieDict)))

reactionMatrix = np.array(reactionMatrixList,dtype='int32')
reactionMatrix = reactionMatrix[reactionMatrix[:,0].argsort()] #Sort the arrays by reaction type to minimize branch divergence / warp divergence when calculating propensities

#Make Parameter Array
parameterList = list()
parameterIndices = reactionMatrix[:,1]
for subParameterIndex in parameterIndices:
    parameterList.append(parameterDict.values()[int(subParameterIndex)])
parameterArray = np.array(parameterList,dtype='float32')

#Make Specices Array
speciesArray = np.array(specieDict.values(),dtype='int32')

#Make Reaction Matrix
reactionMatrix = np.delete(reactionMatrix,1,axis=1)
reactionDF = pd.DataFrame(reactionMatrix)
reactionDF = reactionDF[[0,2,4,6,8,1,3,5,7]] #Reshuffle the columns so reactant/product indices first, followed by their respective reaction deltas
reactionDF = reactionDF.sort(columns=[0,2,6,4,8]) #Simple sorting to maximize coalesced reads
reactionMatrix = reactionDF.values

np.savetxt('reactionMatrix.txt',reactionMatrix,fmt='%i',header="%i rows" % reactionMatrix.shape[0])
np.savetxt('speciesArray.txt',speciesArray,fmt='%i',header="%i rows" % speciesArray.shape[0])
np.savetxt('parameterArray.txt',parameterArray,header="%i rows" % parameterArray.shape[0])
