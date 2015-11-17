# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 13:23:45 2015

@author: jordan
"""

#Import whatever modules will be used
import os
import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.table import Column as Column
from astropy.io import fits, ascii
from scipy import stats
from pyPol import Image

#Setup the path delimeter for this operating system
delim = os.path.sep

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================
# This is the location of the raw data for the observing run
reducedPath = '/home/jordan/ThesisData/PRISM_Data/Reduced_data/'

#Loop through each night and build a list of all the files in observing run
fileList = []
for file in os.listdir(reducedPath):
    # Check that the file is not actually a directory
    filePath = os.path.join(reducedPath, file)
    if not os.path.isdir(filePath):
        fileList.extend([os.path.join(reducedPath, file)])

#Sort the fileList
fileNums = [''.join((file.split(delim).pop().split('.'))[0:2]) for file in fileList]
sortInds = np.argsort(np.array(fileNums, dtype = np.int))
fileList = [fileList[ind] for ind in sortInds]

groupIndexDir = reducedPath + delim + 'GroupIndices'
if (not os.path.isdir(groupIndexDir)):
    os.mkdir(groupIndexDir, 0o755)


#==============================================================================
# ***************************** INDEX *****************************************
# Build an index of the file type and binning, and write it to disk
#==============================================================================
# Check if a file index already exists... if it does then just read it in
indexFile = 'fileIndex.dat'
if not os.path.isfile(indexFile):
    # Loop through each night and test for image type
    print('\nCategorizing files by groups.\n')
    startTime = os.times().elapsed
    
    # Begin by initalizing some arrays to store the image classifications
    obsType  = []
    name     = []
    binType  = []
    polAng   = []
    waveBand = []
    fileCounter = 0
    percentage  = 0
    
    #Loop through each file in the fileList variable
    for file in fileList:
        # Read in the image
        tmpImg = Image(file)
        
        # Classify each file type and binning
        tmpName = tmpImg.header['OBJECT']
        if len(tmpGroup) < 1:
            tmpName = 'blank'
        name.append(tmpName)
        polAng.append(tmpImg.header['POLPOS'])
        waveBand.append(tmpImg.header['FILTNME3'])
        
        # Test the binning of this file
        binTest  = tmpImg.header['CRDELT*']
        if binTest[0] == binTest[1]:
            binType.append(int(binTest[0]))
    
        # Count the files completed and print update progress message
        fileCounter += 1
        percentage1  = np.floor(fileCounter/len(fileList)*100)
        if percentage1 != percentage:
            print('completed {0:3g}%'.format(percentage1), end="\r")
        percentage = percentage1
             
    endTime = os.times().elapsed
    print('\nFile processing completed in {0:g} seconds'.format(endTime -startTime))
    
    # Write the file index to disk
    fileIndex = Table([fileList, name, waveBand, polAng, binType],
                      names = ['Filename', 'Name', 'Waveband', 'Polaroid Angle', 'Binning'])
    ascii.write(fileIndex, indexFile)
else:
    # Read the fileIndex back in as an astropy Table
    print('\nReading file index from disk')
    fileIndex = ascii.read(indexFile)#, guess=False, delimiter=' ')

# Now that the files have been indexed, indetify the number of named groups,
# and prompt the user to fill in the details from the logs.
fileIndexGroups = fileIndex.group_by(['Name'])
numGroups       = len(fileIndexGroups.groups)
print('\nIdentified {0} unique groups'.format(numGroups))

# Grab the start-file for each group
groups        = fileIndexGroups.groups
startFiles    = fileIndexGroups[groups.indices[0:numGroups]]['Filename']

# Sort the groups so that we can move through them "chronologically"
groupSortInds = np.argsort(startFiles)

# Loop through the group sort Inds
for groupInd in groupSortInds:
    # Grab the curent group
    thisGroup = groups[np.int(groupInd)]
    
    # Grab the current target information and check that it is unique
    thisGroupName = np.unique(thisGroup['Name'].data)
    if len(thisGroupName) == 1:
        thisGroupName = str(thisGroupName[0])
    else:
        print('Error: There are multiple OBJECT values in this parent group...')
        pdb.set_trace()
    
    thisWaveband = np.unique(thisGroup['Waveband'].data)
    if len(thisWaveband) == 1:
        thisWaveband = str(thisWaveband[0])
    else:
        print('Error: There are multiple wavebands in this parent group...')
        pdb.set_trace()
    
    # Print the group information to the user
    numImgs = len(thisGroup)
    print('\nProcessing {0} images for parent group "{1}"'.format(numImgs, thisGroupName))
    print('\tFirst file : {0}'.format(os.path.basename(thisGroup['Filename'][0])))
    print('\tLast file  : {0}'.format(os.path.basename(thisGroup['Filename'][numImgs-1])))
    print('\tWaveband   : {0}'.format(thisWaveband))

    keepGroupEntered = False
    while not keepGroupEntered:
        # Ask the user how many sub-groups there are
        keepGroup = input('\n\tShould this group be kept and indexed? (Y/N) ')
        
        # Try to convert response to an integer and catch failures
        try:
            keepGroup = (keepGroup.upper())
            keepGroupEntered = ((keepGroup == 'Y') or (keepGroup == 'YES') or
                                (keepGroup == 'N') or (keepGroup == 'NO'))
            keepGroup = keepGroup[0]
        except:
            print('\tPlease type "Y[es]" or "N[o]"')

    # Proceed if the user tyed "Yes"
    if keepGroup == 'Y':
        print('\n\tCheck the logs:')
        
        # Ask the user to confirm which type of dithering was used
        ditherEntered = False
        while not ditherEntered:
            # Ask the user which type of dithering was used
            ditherType = input('\tDithering type: [1: ABBA, 2: HEX]\n\t')
                            
            # Try to convert response to an integer and catch failures
            try:
                ditherType    = np.int(ditherType)
                ditherEntered = (ditherType == 1) or (ditherType == 2)
            except:
                print('failed to convert input into integer')
        
        # Now save the dither type as a string
        ditherType = ['ABBA', 'HEX'][ditherType-1]
        
        # Ask the user to confirm how many sub-groups are in this parent-group
        numSubGroupsEntered = False
        while not numSubGroupsEntered:
            print('\n\tHint for ABBA dithering:')
            print('\t{0}/(4 polAng * 4 points/polAng) = {1}'.format(numImgs, numImgs/16))
            print('\n\tCheck the logs:')

            # Ask the user how many sub-groups there are
            numSubGroups = input('\tHow many sub-groups are in this parent-group? ')
            
            # Try to convert response to an integer and catch failures
            try:
                numSubGroups = np.int(numSubGroups)
                numSubGroupsEntered = True
            except:
                print('failed to convert input into integer')
        
        # Compute the number of images per sub-group
        numImgsPerSubGroup = float(numImgs)/float(numSubGroups)
        
        # Check that the number of images per sub-group is an integer
        if numImgsPerSubGroup.is_integer():
            # If so, then write a single "index" for each sub-group
            subGroupInds = np.arange(0,numImgs+1,numImgsPerSubGroup)
            for iSub in range(numSubGroups):
                thisSubGroupTable = thisGroup[subGroupInds[iSub]:subGroupInds[iSub+1]]
                subName           = '_'.join([thisGroupName, chr(97+iSub)])
                
                # Add a usage flag column
                thisSubGroupTable.add_column(
                    Column(name='Use',
                           data=np.ones((numImgsPerSubGroup), dtype=np.int)),
                           index=0)
                
                # Add a dither type column
                thisSubGroupTable.add_column(
                    Column(name='Dither',
                           data=np.repeat([ditherType], numImgsPerSubGroup)),
                           index=6)
                
                # Write the file to disk
                indexFile = groupIndexDir + delim + subName + '.dat'
                ascii.write(thisSubGroupTable, indexFile)
                pdb.set_trace()
                
    else:
        print('Skipping to the next group')