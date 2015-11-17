# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 17:03:39 2015

@author: jordan
"""

import os
import numpy as np
from astropy.io import ascii
from astropy.table import Column as Column
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from photutils import detect_sources, Background
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
# 4. Polaroid Angle
fileIndexByTarget = fileIndex.group_by(['Target', 'Waveband', 'Dither', 'Polaroid Angle'])
#fileIndexByTarget = fileIndex.group_by(['Polaroid Angle', 'Target'])

# Loop through each group
groupKeys = fileIndexByTarget.groups.keys
for group in fileIndexByTarget.groups:
    # Grab the current target information
    thisTarget   = str(np.unique(group['Target'].data)[0])
    thisWaveband = str(np.unique(group['Waveband'].data)[0])
    thisPolAng   = str(np.unique(group['Polaroid Angle'].data)[0])
    
    # Test if this target-waveband-polAng combo was previously processed
    outFile = (polarimetryDir + delim +
           '_'.join([thisTarget, thisWaveband, thisPolAng]) + '.fits')
    if os.path.isfile(outFile):
        print('File ' + '_'.join([thisTarget, thisWaveband, thisPolAng]) +
              ' already exists... skipping to next group')
        continue
    
    #**************************************************************************
    # NGC2023 IS TEMPORARILY OUT OF ORDER
    #**************************************************************************
    if thisTarget == 'NGC2023':
        print('Skipping target NGC2023')
        continue
    
    numImgs      = len(group)
    print('\nProcessing {0} images for'.format(numImgs))
    print('\tTarget        : {0}'.format(thisTarget))
    print('\tWaveband      : {0}'.format(thisWaveband))
    print('\tPolaroid Angle: {0}'.format(thisPolAng))
    #
    # This will also require me to TEST that the (AB)BA pair is in order
    # and that the AB(BA) pair is also in the correct order
    #
    # I need to include a step (somewhere) that parses the dither pattern
    # for each "group"... NO CAN DO
    # *** MY ABBA DITHERING IS DETERMINED BY HAND FROM THE LOGS ***
    
    # Read in the files of this group
    imgList    = []
    for file in group['Filename']:
        tmpImg = Image(file)
        polPos = str(tmpImg.header['POLPOS'])
        
        # Check that the polPos value is correct
        if polPos != thisPolAng:
            print('Image polaroid angle does not match expected value')
            pdb.set_trace()
        
        # Only read in 2x2 binned images
        if tmpImg.binning() == 2:
            imgList.append(tmpImg)
    
    del tmpImg
    #
    #*************************************************************************
    # This is where I am... 
    # I should assume that all these are ABBA dithers because I will cull
    # the groups that were not dithered as ABBA ...
    #**************************************************************************
    #
    # Process images in this group according to dither type
    ditherType = np.unique(group['Dither'].data)
    if ditherType == "ABBA UQ":
        # Test if ABBA pattern is there
        if (numImgs % 4) != 0:
            print('The ABBA pattern is not there...')
            pdb.set_trace()
        A1 = imgList[0:len(imgList):4]
        B1 = imgList[1:len(imgList):4]
        B2 = imgList[2:len(imgList):4]
        A2 = imgList[3:len(imgList):4]
        
#        testList = A1.copy()
#        testList.extend(A2)
#        testOutput = Image.align_stack(testList)

        # Initalize lists to store the background subtracted science images
        sciImgList = []
        
        # Loop through each of the dither pattern repeats for this group
        numDithers = int(numImgs / 4)
        print('\n\tIdentified {0} ABBA dithers for this target'.format(numDithers))
        for iDith in range(numDithers):
            print('\tProcessing dither {0}'.format(iDith+1))
            # Compute subtracted images for each repeat
            # *****************************************************************
            # first handle (AB)BA...
            # *****************************************************************
            A1sub = A1[iDith]
            
            # Use the background image to set a 3-sigma detection threshold
            B1bkg     = Background(B1[iDith].arr, (100, 100), filter_shape=(3, 3),
                                   method='median')
            threshold = B1bkg.background + 3.0*B1bkg.background_rms
            
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
            
            # Estimate a 2D background image masking possible sources
            B1bkg = Background(B1[iDith].arr, (100, 100), filter_shape=(3, 3),
                                   method='median', mask=mask)

            # Perform the actual background subtraction
            A1sub.arr = A1sub.arr - B1bkg.background

            # *****************************************************************
            # ...then handle AB(BA)
            # *****************************************************************
            A2sub     = A2[iDith]
            
            # Use the background image to set a 3-sigma detection threshold
            B2bkg     = Background(B2[iDith].arr, (100, 100), filter_shape=(3, 3),
                                   method='median')
            threshold = B2bkg.background + 3.0*B2bkg.background_rms
            
            # Build a mask for any sources above the 3-sigma threshold
            sigma  = 2.0 * gaussian_fwhm_to_sigma    # FWHM = 2.
            kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
            segm   = detect_sources(B2[iDith].arr, threshold, 
                                  npixels=5, filter_kernel=kernel)
            # Build the actual mask and include a step to capture negative
            # saturation values
            mask   = np.logical_or((segm > 0),
                     (np.abs((B2[iDith].arr -
                      B2bkg.background)/B2bkg.background_rms) > 7.0))
            
            # Estimate a 2D background image masking possible sources
            B2bkg = Background(B2[iDith].arr, (100, 100), filter_shape=(3, 3),
                                   method='median', mask=mask)
            
            # Perform the actual background subtraction
            A2sub.arr = A2sub.arr - B2bkg.background
            
            # Append the background subtracted images to the image list
            sciImgList.extend([A1sub, A2sub])
        
        # Now that all the backgrounds have been subtracted,
        # let's align the images and compute an average image
        sciImgList    = Image.align_stack(sciImgList, padding=np.NaN)
        avgImg        = Image()
        avgImg.header = sciImgList[0].header
        avgImg.arr    = Image.stacked_average(sciImgList)
        
        # Clear out the old astrometry
        del avgImg.header['WCSAXES']
        del avgImg.header['PC*']
        del avgImg.header['CDELT*']
        del avgImg.header['CUNIT*']
        del avgImg.header['*POLE']
        avgImg.header['CRPIX*'] = 1.0
        avgImg.header['CRVAL*'] = 1.0
        avgImg.header['CTYPE*'] = 'Linear Binned ADC Pixels'
        avgImg.header['NAXIS1'] = avgImg.arr.shape[1]
        avgImg.header['NAXIS2'] = avgImg.arr.shape[0]

 
        # I can only "redo the astrometry" if the file is written to disk
        avgImg.filename = 'tmp.fits'
        avgImg.write()
       
        # Solve the stacked image astrometry
        success  = avgImg.astrometry()
        
        # Clean up temporary files
        if os.path.isfile('none'):
            os.system('rm none')
        if os.path.isfile('tmp.fits'):
            os.system('rm tmp.fits')
        
        # With successful astrometry, save result to disk
        if success:
            avgImg.write(outFile)
        else:
            print('astrometry failed?!')
            pdb.set_trace()
    else:
        print('Hex-dither support not yet enabled')
