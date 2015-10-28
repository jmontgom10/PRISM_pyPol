# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 17:03:39 2015

@author: jordan
"""

import os
import numpy as np
from astropy.io import fits, ascii
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from photutils import detect_sources, detect_threshold, Background
import pdb
from pyPol import Image

# This script will run the astrometry step of the pyPol reduction

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

# Sort the fileList
fileNums = [''.join((file.split(delim).pop().split('.'))[0:2]) for file in fileList]
fileNums = [num.split('_')[0] for num in fileNums]
sortInds = np.argsort(np.array(fileNums, dtype = np.int))
fileList = [fileList[ind] for ind in sortInds]

# Read the fileIndex back in as an astropy Table
print('\nReading file index from disk')
indexFile = 'fileIndex.dat'
fileIndex = ascii.read(indexFile)

# Read the groupDither back in as an astropy Table
print('\nReading dither index from disk')
groupDither = ascii.read('groupDither.dat')

# Determine which parts of the Fileindex pertain to science images
keepFiles = []
for file in fileIndex['Filename']:
    Filename = file.split(delim)
    Filename.reverse()
    Filename = reducedDir + delim + Filename[0]
    keepFiles.append(Filename in fileList)

# Update the fileIndex
fileIndex = fileIndex[np.where(keepFiles)]
fileIndex['Filename'] = fileList

# Group the fileIndex by 'Target' and 'Polaroid Angle'
fileIndexByTarget = fileIndex.group_by(['Target', 'Polaroid Angle'])

# Loop through each group
groupKeys = fileIndexByTarget.groups.keys
for group in fileIndexByTarget.groups:
    # Find the group index
    groupInd   = ((np.where(groupDither['Target'] == np.unique(group['Target'])))[0])[0]
    
    print('Processing group {0} with polaroid angle {1}'.format(
        groupKeys['Target'][groupInd], groupKeys['Polaroid Angle'][groupInd]))
        
    # Read in the files of this group
    imgList    = []
    for file in group['Filename']:
        imgList.append(Image(file))
        numImgs = len(imgList)
    
    # Process images in this group according to dither type
    ditherType = groupDither[groupInd]['Dither']
    if ditherType == "ABBA UQ":
        # Test if ABBA pattern is there
        if (numImgs % 4) != 0:
            print('The ABBA pattern is not there...')
            pdb.set_trace()
        A1 = imgList[0:len(imgList):4]
        B1 = imgList[1:len(imgList):4]
        B2 = imgList[2:len(imgList):4]
        A2 = imgList[3:len(imgList):4]
        
        # Loop through each of the repeats for this group
        # *********************************************************************
        # ACTUALLY SHOULD BG-SUBTRACT **AND** COMPUTE U/Q IMAGES
        # *********************************************************************
        numDithers = int(numImgs / 4)
        # Loop through each of the repeats for this target
        for iDith in range(numDithers):
            # Compute subtracted images for each repeat
            A1sub     = A1[iDith]
            B1bkg     = Background(B1[iDith].arr, (100, 100), filter_shape=(3, 3),
                                   method='median')
            # Estimate a 2D background image
            # Use the background image to set a 3-sigma detection threshold
            threshold = B1bkg.background + (1.0 * B1bkg.background_rms)
            
            # Build a mask for any sources above the 3-sigma threshold
            sigma  = 2.0 * gaussian_fwhm_to_sigma    # FWHM = 2.
            kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
            segm   = detect_sources(B1[iDith].arr, threshold, 
                                  npixels=5, filter_kernel=kernel)
            # Build the actual mask and include a step to capture negative
            # saturation values
            mask   = np.logical_or((segm > 0),
                     (np.abs((B1[iDith].arr -
                      B1bkg.background)/B1bkg.background_rms) > 7.0))

            # Estimate a 2D background with a finer meshing
            B1bkg     = Background(B1[iDith].arr, (50, 50), filter_shape=(3, 3),
                                   method='median', mask=mask)
            # Perform the actual background subtraction
            A1sub.arr = A1sub.arr - B1bkg.background
            
            pdb.set_trace()
            
            A2sub     = A2[iDith]
            B2bkg     = Background(B2[iDith].arr, (100, 100), filter_shape=(3, 3),
                                   method='median')
            A2sub.arr = A2sub.arr - B2bkg.background
            pdb.set_trace()
            
            # Write the subtracted files to disk
            A1Filename = A1sub.filename
            A1sub.write('test.fits')
            
        pdb.set_trace()
        