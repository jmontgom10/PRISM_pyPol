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
import image_tools

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

# Setup PRISM detector properties
read_noise = 13.0 # electrons
effective_gain = 3.3 # electrons/ADU

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

# Loop through each group
groupKeys = fileIndexByTarget.groups.keys
for group in fileIndexByTarget.groups:
    # Grab the current target information
    thisTarget   = str(np.unique(group['Target'].data)[0])
    thisWaveband = str(np.unique(group['Waveband'].data)[0])
    thisPolAng   = str(np.unique(group['Polaroid Angle'].data)[0])

    if thisTarget != 'NGC7023' or thisWaveband != 'V': continue

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
            for pairNum, ABimg in enumerate(zip(Aimgs, Bimgs)):
                Aimg = ABimg[0]
                Bimg = ABimg[1]
                print('\t\tProcessing on-off pair {0}'.format(pairNum+1))

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

                # Catch any non-finite values and "mask" them with -1e6 value
                nanInds = np.where(np.logical_not(np.isfinite(Aimg.arr)))
                Aimg.arr[nanInds] = -1e6

                # Catch saturated negative values
                badInds = np.where(Aimg.arr < -3*read_noise/effective_gain)
                Aimg.arr[badInds] = np.NaN

                # Perform the actual background subtraction
                Aimg.arr  = Aimg.arr - B1bkg.background

                # Append the background subtracted image to the list
                sciImgList.append(Aimg.copy())

            # Now that all the backgrounds have been subtracted,
            # let's align the images and compute an average image
            # TODO Should I use "cross-correlation" alignment?
            sciImgList = image_tools.align_images(sciImgList, padding=-1e6)
            avgImg     = image_tools.combine_images(sciImgList, output = 'MEAN',
                effective_gain = effective_gain,
                read_noise = read_noise)

    else:
        # Handle HEX-DITHERS images
        # (mostly for calibration images as of now)

        # Test if numImgs matches the ABBA pattern
        if (numImgs % 6) != 0:
            print('The HEX dither pattern is not there...')
            pdb.set_trace()

        # Now align and combine these images into a single average image
        imgList = image_tools.align_images(imgList, padding=-1e6)

        # Loop through and catch any pixels which need masking...
        for img in imgList:
            # Catch any non-finite values and "mask" them with -1e6 value
            nanInds = np.where(np.logical_not(np.isfinite(img.arr)))
            img.arr[nanInds] = -1e6

            # Catch saturated negative values
            badInds = np.where(img.arr < -3*read_noise/effective_gain)
            img.arr[badInds] = np.NaN

        avgImg  = image_tools.combine_images(imgList, output = 'MEAN',
            effective_gain = effective_gain,
            read_noise = read_noise)

    # It is only possible to "redo the astrometry" if the file on the disk
    # First make a copy of the average image
    tmpImg = avgImg.copy()
    # Replace NaNs with something finite and name the file "tmp.fits"
    tmpImg.arr = np.nan_to_num(tmpImg.arr)
    tmpImg.filename = 'tmp.fits'

    # Delete the sigma attribute
    if hasattr(tmpImg, 'sigma'):
        del tmpImg.sigma

    # Record the temporary file to disk for performing astrometry
    tmpImg.write()

    # Solve the stacked image astrometry
    avgImg1, success = image_tools.astrometry(tmpImg)

    # Re-assign the uncertainty array to the output image
    if hasattr(avgImg, 'sigma'):
        avgImg1.sigma = avgImg.sigma

    # With successful astrometry, save result to disk
    if success:
        print('astrometry succeded')

        # Clean up temporary files
        # TODO update to by system independent
        # (use subprocess module and "del" command for Windows)
        # See AstroImage.astrometry for example

        del tmpImg

        if os.path.isfile('none'):
            os.system('rm none')
        if os.path.isfile('tmp.fits'):
            os.system('rm tmp.fits')

        # Save the file to disk
        avgImg1.write(outFile)
    else:
        print('astrometry failed?!')
        pdb.set_trace()

print('\nDone computing average images!')
