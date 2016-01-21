# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:59:25 2015

@author: jordan
"""

import os
import sys
import numpy as np
from astropy.io import ascii
from astropy.table import Table as Table
from astropy.table import Column as Column
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from photutils import detect_sources, Background
import pdb

# Add the AstroImage class
sys.path.append("C:\\Users\\Jordan\\Libraries\\python\\AstroImage")
from AstroImage import AstroImage

# This script will compute the photometry of polarization standard stars
# and output a file containing the polarization position angle
# additive correction and the polarization efficiency of the PRISM instrument.

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data'

# This is the location of the previously generated masks (step 4)
maskDir = os.path.join(pyPol_data, 'Masks')

# Setup new directory for polarimetry data
polarimetryDir = os.path.join(pyPol_data, 'Polarimetry')
if (not os.path.isdir(polarimetryDir)):
    os.mkdir(polarimetryDir, 0o755)

polAngDir = os.path.join(polarimetryDir, 'polAngImgs')
if (not os.path.isdir(polAngDir)):
    os.mkdir(polAngDir, 0o755)

# Read in the indexFile data and select the filenames
print('\nReading file index from disk')
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='csv')

# The user can speed up the process by defining the "Target" values from
# the fileIndex to be considered for masking.
# Masks can onlybe produced for targets in this list.
targets = ['Orion_Cal', 'Taurus_Cal']

# Determine which parts of the fileIndex pertain to science images
# TODO make this look only for the specified targets (cf. 04_buildMasks)

# Determine which parts of the fileIndex pertain to science images
useFiles = np.logical_and(fileIndex['Use'] == 1,
                          fileIndex['Dither'] == 'HEX')

# Further restrict the selection to only include the selected targets
targetFiles = np.array([False]*len(fileIndex), dtype=bool)
for target in targets:
    targetFiles = np.logical_or(targetFiles,
                                fileIndex['Target'] == target)

# Cull the fileIndex to ONLY include the specified targets
fileIndex = fileIndex[np.where(np.logical_and(useFiles, targetFiles))]

# Group the fileIndex by...
# 1. Target
# 2. Waveband
# 3. Dither (pattern)
# 4. Polaroid Angle
fileIndexByTarget = fileIndex.group_by(['Target', 'Night', 'Waveband', 'Dither', 'Polaroid Angle'])


for group in fileIndexByTarget.groups:
    # Grab the current target information
    thisTarget   = str(np.unique(group['Target'].data)[0])
    thisNight    = str(np.unique(group['Night'].data)[0])
    thisWaveband = str(np.unique(group['Waveband'].data)[0])
    thisPolAng   = str(np.unique(group['Polaroid Angle'].data)[0])

    # Test if this target-waveband-polAng combo was previously processed
    outFile = os.path.join(polAngDir,
        '_'.join([thisTarget, thisNight, thisWaveband, thisPolAng]) + '.fits')

    if os.path.isfile(outFile):
        print('File ' + '_'.join([thisTarget, thisWaveband, thisPolAng]) +
              ' already exists... skipping to next group')
        continue

    numImgs      = len(group)
    print('\nProcessing {0} images for'.format(numImgs))
    print('\tTarget         : {0}'.format(thisTarget))
    print('\tNight          : {0}'.format(thisNight))
    print('\tWaveband       : {0}'.format(thisWaveband))
    print('\tPolaroid Angle : {0}'.format(thisPolAng))

    # Test if numImgs matches the ABBA pattern
    if (numImgs % 6) != 0:
        print('The HEX dither pattern is not there...')
        pdb.set_trace()

    # Read in the files of this group
    imgList    = []
    for file1 in group['Filename']:
        tmpImg  = AstroImage(file1)

        # Test if this file has an associated mask
        maskFile = os.path.join(maskDir,os.path.basename(file1))
        if os.path.isfile(maskFile):
            # Read in any associated mask and add it to the arr attribute
            tmpMask = AstroImage(maskFile)

            # Mask the image by setting masked values to "np.NaN"
            tmpImg.arr[np.where(tmpMask.arr == 1)] = np.NaN
#            tmpImg.arr = np.ma.array(tmpImg.arr, mask=tmpMask.arr, copy=True)

        # Grab the polaroid angle position from the header
        polPos = str(tmpImg.header['POLPOS'])

        # Check that the polPos value is correct
        if polPos != thisPolAng:
            print('Image polaroid angle does not match expected value')
            pdb.set_trace()

        # If everything seems ok, then add this to the imgList
        imgList.append(tmpImg)

    # Cleanup temporary variables
    del tmpImg

    # Now align and combine these images into a single average image
    imgList       = AstroImage.align_stack(imgList, padding=np.NaN)
    avgImg        = AstroImage()
    avgImg.header = imgList[0].header
    avgImg.arr    = AstroImage.stacked_average(imgList)

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
        print('astrometry succeded')
        avgImg.write(outFile)
    else:
        print('astrometry failed?!')
        pdb.set_trace()

print('Done with this script!')
