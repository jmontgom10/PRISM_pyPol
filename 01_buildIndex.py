# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 13:23:45 2015

@author: jordan
"""

#Import whatever modules will be used
import os
import sys
import time
import pdb
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.table import Column as Column
from astropy.io import fits, ascii
from scipy import stats

# Add the AstroImage class
sys.path.append("C:\\Users\\Jordan\\Libraries\\python\\AstroImage")
from AstroImage import AstroImage

################################################################################
# Define a recursive file search which takes a parent directory and returns all
# the FILES (not DIRECTORIES) beneath that node.
def recursive_file_search(parentDir, exten='', fileList=[]):
    # Query the elements in the directory
    subNodes = os.listdir(parentDir)

    # Loop through the nodes...
    for node in subNodes:
        # If this node is a directory,
        thisPath = os.path.join(parentDir, node)
        if os.path.isdir(thisPath):
            # then drop down recurse the function
            recursive_file_search(thisPath, exten, fileList)
        else:
            # otherwise test the extension,
            # and append the node to the fileList
            if len(exten) > 0:
                # If an extension was defined,
                # then test if this file is the right extension
                exten1 = (exten[::-1]).upper()
                if (thisPath[::-1][0:len(exten1)]).upper() == exten1:
                    fileList.append(thisPath)
            else:
                fileList.append(thisPath)

    # Return the final list to the user
    return fileList
################################################################################

#Setup the path delimeter for this operating system
delim = os.path.sep

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================
# This is the location of all pyBDP data (index, calibration images, reduced...)
pyBDP_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyBDP_data'

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data'

# This is the location of the pyBDP processed Data
pyBDP_reducedDir = os.path.join(pyBDP_data, 'pyBDP_reduced_images')

fileList = recursive_file_search(pyBDP_reducedDir, exten='.fits')

#Sort the fileList
fileNums = [''.join((file.split(delim).pop().split('.'))[0:2]) for file in fileList]
fileNums = [file.split('_')[0] for file in fileNums]
sortInds = np.argsort(np.array(fileNums, dtype = np.int64))
fileList = [fileList[ind] for ind in sortInds]

#==============================================================================
# ***************************** INDEX *****************************************
# Build an index of the file type and binning, and write it to disk
#==============================================================================
# Check if a file index already exists... if it does then just read it in
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')

# Loop through each night and test for image type
print('\nCategorizing files by groups.\n')
startTime = time.time()

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
    tmpImg = AstroImage(file)

    # Classify each file type and binning
    tmpName = tmpImg.header['OBJECT']
    if len(tmpName) < 1:
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

endTime  = time.time()
numFiles = len(fileList)
print(('\n{0} File processing completed in {1:g} seconds'.
       format(numFiles, (endTime -startTime))))

# Write the file index to disk
fileIndex = Table([fileList, name, waveBand, polAng, binType],
                  names = ['Filename', 'Name', 'Waveband', 'Polaroid Angle', 'Binning'])
fileIndex.add_column(Column(name='Use',
                            data=np.ones((numFiles)),
                            dtype=np.int),
                            index=0)

# Group by "Name"
groupFileIndex = fileIndex.group_by('Name')

# Grab the file-number orderd indices for the groupFileIndex
fileIndices = np.argsort(groupFileIndex['Filename'])

# Loop through each "Name" and assign it a "Target" value
targetList = []
ditherList = []
for group in groupFileIndex.groups:
    # Select this groups properties
    thisName = np.unique(group['Name'])

    # Test if the group name truely is unique
    if len(thisName) == 1:
        thisName = thisName[0]
    else:
        print('There is more than one name in this group!')
        pdb.set_trace()

    # Count the number of elements in this group
    groupLen = len(group)

    # Add the "Target" column to the fileIndex
    thisTarget = input('\nEnter the target for group "{0}": '.format(thisName))
    thisTarget = [thisTarget]*groupLen

    # Ask the user to supply the dither pattern for this group
    thisDitherEntered = False
    while not thisDitherEntered:
        # Have the user select option 1 or 2
        print('\nEnter the dither patttern for group "{0}": '.format(thisName))
        thisDither = input('[1: ABBA, 2: HEX]')

        # Test if the numbers 1 or 2 were entered
        try:
            thisDither = np.int(thisDither)
            if (thisDither == 1) or (thisDither == 2):
                # If so, then reassign as a string
                thisDither = ['ABBA', 'HEX'][(thisDither-1)]
                thisDitherEntered = True
        except:
            print('Response not recognized')

    # Create a list of "thisDither" entries
    thisDither = [thisDither]*groupLen

    # Add these elements to the target list
    targetList.extend(thisTarget)
    ditherList.extend(thisDither)

pdb.set_trace()
# Add the "Target" and "Dither columns"
groupFileIndex.add_column(Column(name='Target',
                            data=np.array(targetList)),
                            index = 2)
groupFileIndex.add_column(Column(name='Dither',
                            data=np.array(ditherList)),
                            index = 7)

# Re-sort by file-number
fileSortInds = np.argsort(groupFileIndex['Filename'])
fileIndex1   = groupFileIndex[fileSortInds]

# Write file to disk
fileIndex1.write(indexFile, format='csv')
