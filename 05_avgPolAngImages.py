# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 17:03:39 2015

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

# This script will run the image averaging step of the pyPol reduction

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

# Determine which parts of the fileIndex pertain to science images
useFiles = np.where(fileIndex['Use'] == 1)

# Cull the file index to only include files selected for use
fileIndex = fileIndex[useFiles]

# Group the fileIndex by...
# 1. Target
# 2. Waveband
# 3. Dither (pattern)
# 4. Polaroid Angle
fileIndexByTarget = fileIndex.group_by(['Target', 'Waveband', 'Dither', 'Polaroid Angle'])
#fileIndexByTarget = fileIndex.group_by(['Polaroid Angle', 'Target'])

# for group in fileIndexByTarget.groups:
#     thisTarget   = str(np.unique(group['Target'].data)[0])
#     thisWaveband = str(np.unique(group['Waveband'].data)[0])
#     thisPolAng   = str(np.unique(group['Polaroid Angle'].data)[0])
#     if thisTarget == 'NGC2023' and thisWaveband == 'V':
#         pdb.set_trace()

# Loop through each group
groupKeys = fileIndexByTarget.groups.keys
for group in fileIndexByTarget.groups:
    # Grab the current target information
    thisTarget   = str(np.unique(group['Target'].data)[0])
    thisWaveband = str(np.unique(group['Waveband'].data)[0])
    thisPolAng   = str(np.unique(group['Polaroid Angle'].data)[0])

    # Test if this target-waveband-polAng combo was previously processed
    outFile = os.path.join(polAngDir,
        '_'.join([thisTarget, thisWaveband, thisPolAng]) + '.fits')

    if os.path.isfile(outFile):
        print('File ' + '_'.join([thisTarget, thisWaveband, thisPolAng]) +
              ' already exists... skipping to next group')
        continue

    numImgs      = len(group)
    print('\nProcessing {0} images for'.format(numImgs))
    print('\tTarget         : {0}'.format(thisTarget))
    print('\tWaveband       : {0}'.format(thisWaveband))
    print('\tPolaroid Angle : {0}'.format(thisPolAng))

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

    # Process images in this group according to dither type
    ditherType = np.unique(group['Dither'].data)
    if ditherType == "ABBA":
        # Test if ABBA pattern is there
        if (numImgs % 4) != 0:
            # No ABBA pattern, so stop executing
            print('No (ABBA) pattern present')
            print('Could not guess which images are "on target"')
            pdb.set_trace()
        else:
            # Proper number of images for an ABB pattern,
            # so break the images up into
            #"on-target" / "off-target" image lists
            Aimgs = []
            Bimgs = []
            for i, img in enumerate(imgList):
                pointing = 'ABBA'[i % 4]
                if pointing == 'A':
                    Aimgs.append(img)
                if pointing == 'B':
                    Bimgs.append(img)

            # Loop through each pair and bild a bg-subtracted list
            sciImgList = []
            for Aimg, Bimg in zip(Aimgs, Bimgs):
                # Estimate the background for this pair
                B1bkg     = Background(Bimg.arr, (100, 100), filter_shape=(3, 3),
                                       method='median')
                threshold = B1bkg.background + 3.0*B1bkg.background_rms

                # Build a mask for any sources above the 3-sigma threshold
                sigma  = 2.0 * gaussian_fwhm_to_sigma    # FWHM = 2.
                kernel = Gaussian2DKernel(sigma, x_size=6, y_size=6)
                segm   = detect_sources(Bimg.arr, threshold,
                                      npixels=5, filter_kernel=kernel)

                # Build the actual mask and include a step to capture negative
                # saturation values
                mask   = np.logical_or((segm.data > 0),
                         (np.abs((Bimg.arr -
                          B1bkg.background)/B1bkg.background_rms) > 7.0))

                # Estimate a 2D background image masking possible sources
                B1bkg = Background(Bimg.arr, (100, 100), filter_shape=(3, 3),
                                       method='median', mask=mask)

                # Perform the actual background subtraction
                Aimg.arr  = Aimg.arr - B1bkg.background

                # Append the background subtracted image to the list
                sciImgList.append(Aimg.copy())

            # Now that all the backgrounds have been subtracted,
            # let's align the images and compute an average image
            # TODO Should I use "cross-correlation" alignment?
            sciImgList    = AstroImage.align_stack(sciImgList, padding=np.NaN)
            avgImg        = AstroImage()
            avgImg.header = sciImgList[0].header
            avgImg.arr    = AstroImage.stacked_average(sciImgList)

    else:
        # Handle HEX-DITHERS images
        # (mostly for calibration images as of now)

        # Test if numImgs matches the ABBA pattern
        if (numImgs % 6) != 0:
            print('The HEX dither pattern is not there...')
            pdb.set_trace()

        # Now align and combine these images into a single average image
        imgList       = AstroImage.align_stack(imgList, padding=np.NaN)
        avgImg        = AstroImage()
        avgImg.header = imgList[0].header
        avgImg.arr    = AstroImage.stacked_average(imgList)

    # Now that an average image has been computed (for either dither pattern),
    # let's completely re-solve the astrometry of the newly created image.
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
