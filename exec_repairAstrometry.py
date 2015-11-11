# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:56:02 2015

@author: jordan
"""

import os
import numpy as np
from astropy.io import ascii
from astropy.table import Column as Column
import astropy.coordinates as coord
import astropy.units as u
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib import pyplot as plt
import pdb
from pyPol import Image

#******************************************************************************
# Write a quick function to provide list string searching...
def str_list_contains(inList, searchStr):
    """This function searches the elements of a list of strings for matches.
    
    parameters:
    inList    -- a list containing ONLY strings
    searchStr -- a string to search for...
    """
    
    # Check that the searchStr parameter is a string
    if not isinstance(searchStr, str):
        print('The searhcStr parameter must be a string')
        return None
    
    outList = []
    for el in inList:
        # Check that this element is also a string
        if not isinstance(el, str):
            print('All elements of the inList parameter must be pure strings')
            return None
        
        outList.append(el.__contains__(searchStr))
    
    return outList
#******************************************************************************

# This script will run the image averaging step of the pyPol reduction

#******************************************************************************
# First the user must identify the names of the targets to be batched
#******************************************************************************
targets = ['M104', 'M78', 'M82', 'NGC2023', 'NGC7023', 'NGC_1977']

#Setup the path delimeter for this operating system
delim = os.path.sep

# Grab all the *.fits files in the reduced science data directory
reducedDir = '/home/jordan/ThesisData/PRISM_Data/Reduced_data'
fileList   = []
for file in os.listdir(reducedDir):
    filePath = os.path.join(reducedDir, file)
    fileTest = os.path.isfile(filePath)
    extTest  = (os.path.splitext(filePath)[1] == '.fits')
    if fileTest and extTest:
        fileList.extend([os.path.join(reducedDir, file)])

# Setup new directory for plate scale histograms data
astroCheckDir = reducedDir + delim + 'astrometryCheck'
if (not os.path.isdir(astroCheckDir)):
    os.mkdir(astroCheckDir, 0o755)

failedAstroFile = astroCheckDir + delim + 'failedAstro.dat'


# Sort the fileList
fileNums = [''.join((file.split(delim).pop().split('.'))[0:2]) for file in fileList]
fileNums = [num.split('_')[0] for num in fileNums]
sortInds = np.argsort(np.array(fileNums, dtype = np.int))
fileList = [fileList[ind] for ind in sortInds]

# Setup new directory for polarimetry data
polarimetryDir = reducedDir + delim + 'Polarimetry'
if (not os.path.isdir(polarimetryDir)):
    os.mkdir(polarimetryDir, 0o755)

# Read the fileIndex back in as an astropy Table
print('\nReading file index from disk')
indexFile = 'fileIndex.dat'
fileIndex = ascii.read(indexFile)

# Determine which parts of the Fileindex pertain to science images
keepFiles = []
for file in fileIndex['Filename']:
    Filename = file.split(delim)
    Filename.reverse()
    Filename = reducedDir + delim + Filename[0]
    keepFiles.append(Filename in fileList)

# Update the fileIndex with the paths to reduced files
fileIndex = fileIndex[np.where(keepFiles)]
fileIndex['Filename'] = fileList

# Read the groupDither back in as an astropy Table
print('\nReading dither index from disk')
groupDither = ascii.read('groupDither.dat')


# Use the groupDither information to add a "Dither" column to the fileIndex
nullStr    = 'ThisDitherIsNotRecorded'
ditherList = np.array([nullStr]*len(fileIndex))
for group, dither in groupDither:
    groupInds = np.where(str_list_contains(fileIndex['Group'].data, group))
    ditherList[groupInds] = dither

# Add a "Dither" column to the fileIndex
fileIndex.add_column(Column(name='Dither', data=ditherList), index=5)

# Prepare to add a 'Target' Column to the fileIndex
nullStr   = "ThisIsNotATarget"
groupList = []
groupList.extend(fileIndex['Group'].data)
targetList = np.array([nullStr]*len(groupList))

# Loop through each of the targets and identify
# which groups are assigned to each target.
for target in targets:
    targetInds = np.where(str_list_contains(groupList, target))
    targetList[targetInds] = target

# Add a "Target" column to the fileIndex
fileIndex.add_column(Column(name='Target', data=targetList), index=2)

# Remove non-target elements of the fileIndex
keepFiles = [not i for i in str_list_contains(targetList, nullStr)]
fileIndex = fileIndex[np.where(keepFiles)]

# Group the fileIndex by...
# 1. Target
# 2. Waveband
# 3. Dither (pattern)
# 4. Binning
fileIndexByTarget = fileIndex.group_by(['Target', 'Waveband', 'Dither', 'Binning'])
for group in fileIndexByTarget.groups:
    # Grab the current target information
    thisTarget   = str(np.unique(group['Target'].data)[0])
    thisWaveband = str(np.unique(group['Waveband'].data)[0])
    thisDither   = str(np.unique(group['Dither'].data)[0])
    thisBinning  = str(np.unique(group['Binning'].data)[0])
    
    numImgs      = len(group)
    print('\nProcessing {0} images for'.format(numImgs))
    print('\tTarget        : {0}'.format(thisTarget))
    print('\tWaveband      : {0}'.format(thisWaveband))
    print('\tDither        : {0}'.format(thisDither))
    print('\tBinning       : {0}'.format(thisBinning))
    
    # Get plate scales for all the images of this target.
    # Load the plate scales into a list
    targetBinningList    = []
    targetPC_matrixList  = []
    targetPlateScaleList = []
    
    # Loop through the files
    for file in group['Filename']:
        # Load the image header
        tmpImage  = Image(file)
        tmpHeader = tmpImage.header
        
        # Grab the WCS
        tmpWCS    = WCS(tmpHeader)
        
        # Grab the PC macrices and pixel scales and store them in the list
        targetBinningList.append(tmpImage.binning)
        targetPC_matrixList.append(tmpWCS.wcs.get_pc())
        targetPlateScaleList.append(proj_plane_pixel_scales(tmpWCS))
    
    # Histogram the plate scales along each axis
    targetBinning     = (np.tile(np.array(targetBinningList), (2,1))).T
    targetPC_matrices = np.array(targetPC_matrixList)
    targetPlateScales = np.array(targetPlateScaleList) * 3600.0 / targetBinning
    
    # Check for significant deviates
    deviation    = np.abs(targetPlateScales - np.median(targetPlateScales))
    badAstroBool = np.logical_or((deviation[:,0] > 0.05),
                                 (deviation[:,1] > 0.05))
    badAstroInds = np.where(badAstroBool)[0]
    # If there were bad astrometries, then record them
    if len(badAstroInds) > 0:
        # Determine the good astrometry values
        goodAstroInds = np.where(np.logical_not(badAstroBool))[0]
        goodAstroVals = targetPC_matrices[goodAstroInds]
        goodAstroVals = np.mean(goodAstroVals, axis=0)
        
        # Read in a good image to get a good WCS
        goodImg = Image(group['Filename'][goodAstroInds[0]])
        goodWCS = WCS(goodImg.header)
        
        # Loop through all the bad files and "fix them"
        for badInd in badAstroInds:
            # Build a quick header from the WCS object
            thisFile = group['Filename'][badInd]
            thisImg  = Image(thisFile)
            
            # Grab the approximate pointing from the header
            thisRA  = coord.Angle(thisImg.header['TELRA'], unit=u.hour)
            thisRA.degree
            thisDec = coord.Angle(thisImg.header['TELDEC'], unit=u.degree)
            
           # Update the image header to contain the astrometry info
            thisImg.header['CTYPE1']  = 'RA---TAN-SIP'
            thisImg.header['CTYPE2']  = 'DEC--TAN-SIP'
            thisImg.header['CRPIX1']  = thisImg.arr.shape[1]//2
            thisImg.header['CRPIX2']  = thisImg.arr.shape[0]//2
            thisImg.header['CRVAL1']  = thisRA.deg
            thisImg.header['CRVAL2']  = thisDec.deg
            thisImg.header['PC1_1']   = goodAstroVals[0,0]
            thisImg.header['PC1_2']   = goodAstroVals[0,1]
            thisImg.header['PC2_1']   = goodAstroVals[1,0]
            thisImg.header['PC2_2']   = goodAstroVals[1,1]
            thisImg.header['CDELT1']  = 1.0
            thisImg.header['CDELT2']  = 1.0
            thisImg.header['CUNIT1']  = 'deg'
            thisImg.header['CUNIT2']  = 'deg'
            thisImg.header['LONPOLE'] = 180.0
            thisImg.header['LATPOLE'] = -2.2    #Does this need to be calculated somehow?
            thisImg.header['RADESYS'] = 'FK5'
            
            # Insert repaired pc matrix and plate scale
            tmpWCS                    = WCS(thisImg.header)
            targetPC_matrices[badInd] = tmpWCS.wcs.get_pc()
            targetPlateScales[badInd] = proj_plane_pixel_scales(tmpWCS) * 3600 / thisImg.binning

            # Write updated file properties to disk
            thisImg.write()
            
            thisFile = os.path.basename(group['Filename'][badInd])
            print('Logging bad file: ' + thisFile)
            os.system('echo "' + thisFile +
                      '" >> ' + failedAstroFile)
    
    # Prep for histogram plotting
    minPS    = np.min(targetPlateScales)
    maxPS    = np.max(targetPlateScales)
    binWidth = 0.0005
    numBins  = np.int(np.ceil((maxPS - minPS)/binWidth)) + 2
    binEdges = np.linspace(minPS-binWidth, maxPS+binWidth, numBins)
    
    n1, bins1, patches1 = plt.hist(targetPlateScales[:,0], binEdges, histtype='stepfilled',
                                rwidth=1, facecolor='red', alpha=0.7)
    n2, bins2, patches2 = plt.hist(targetPlateScales[:,1], binEdges, histtype='stepfilled',
                                rwidth=1, facecolor='blue', alpha=0.4)
    
    filename = (astroCheckDir + delim + 
                '_'.join([thisTarget, thisWaveband, thisBinning]) + '.png')
    plt.savefig(filename)
    plt.clf()