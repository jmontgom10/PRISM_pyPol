# -*- coding: utf-8 -*-
"""
Combines all the images for a given (TARGET, FILTER, POLPOS) combination to
produce a single, average image.

Estimates the sky background level of the on-target position at the time of the
on-target observation using a bracketing pair of off-target observations through
the same POLPOS polaroid rotation value. Subtracts this background level from
each on-target image to produce background free images. Applies an airmass
correction to each image, and combines these final image to produce a background
free, airmass corrected, average image.
"""

# Imports
import os
import sys
import copy
from datetime import datetime
import warnings

# Import scipy/numpy packages
import numpy as np

# Import astropy packages
from astropy.table import Table
import astropy.units as u
from astropy.convolution import Gaussian2DKernel
from astropy.modeling import models, fitting
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from photutils import detect_sources, SigmaClip, Background2D, MedianBackground

# Import plotting utilities
from matplotlib import pyplot as plt

# Add the AstroImage class
import astroimage as ai

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================

# This is a list of targets for which to process each subgroup (observational
# group... never spanning multiple nights, etc...) instead of combining into a
# single "metagroup" for all observations of that target. The default behavior
# is to go ahead and combine everything into a single, large "metagroup". The
# calibration data should probably not be processed as a metagroup though.
processSubGroupList = ['Taurus_Cal', 'Orion_Cal', 'Cyg_OB2']
processSubGroupList = [t.upper() for t in processSubGroupList]

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data\\201612'

# This is the location of the previously generated masks (step 4)
maskDir = os.path.join(pyPol_data, 'Masks')

# Setup new directory for polarimetry data
polarimetryDir = os.path.join(pyPol_data, 'Polarimetry')
if (not os.path.isdir(polarimetryDir)):
    os.mkdir(polarimetryDir, 0o755)

polAngDir = os.path.join(polarimetryDir, 'polAngImgs')
if (not os.path.isdir(polAngDir)):
    os.mkdir(polAngDir, 0o755)

bkgPlotDir = os.path.join(polAngDir, 'bkgPlots')
if (not os.path.isdir(bkgPlotDir)):
    os.mkdir(bkgPlotDir, 0o755)

# # Setup PRISM detector properties
# read_noise = 13.0 # electrons
# effective_gain = 3.3 # electrons/ADU

#########
### Establish the atmospheric extinction (magnitudes/airmass)
#########
# Following table from Hu (2011)
# Data from Gaomeigu Observational Station
# Passband | K'(lambda) [mag/airmass] | K'' [mag/(color*airmass)]
# U	         0.560 +/- 0.023            0.061 +/- 0.004
# B          0.336 +/- 0.021            0.012 +/- 0.003
# V          0.198 +/- 0.024           -0.015 +/- 0.004
# R          0.142 +/- 0.021           -0.067 +/- 0.005
# I          0.093 +/- 0.020            0.023 +/- 0.006


# Following table from Schmude (1994)
# Data from Texas A & M University Observatory
# Passband | K(lambda) [mag/airmass] | dispersion on K(lambda)
# U	         0.60 +/- 0.05             0.120
# B          0.40 +/- 0.06             0.165
# V          0.26 +/- 0.03             0.084
# R          0.19 +/- 0.03             0.068
# I          0.16 +/- 0.02             0.055

kappa = dict(zip(['U',    'B',    'V',    'R'   ],
                 [0.60,   0.40,   0.26,   0.19  ]))


# Read in the indexFile data and select the filenames
print('\nReading file index from disk')
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='ascii.csv')

# Determine which parts of the fileIndex pertain to HEX dither science images
useFiles    = np.logical_and(
    fileIndex['USE'] == 1,
    fileIndex['DITHER_TYPE'] == 'HEX'
)
useFileRows = np.where(useFiles)

# Cull the file index to only include files selected for use
fileIndex = fileIndex[useFileRows]

# Define an approximate pixel scale
pixScale  = 0.39*(u.arcsec/u.pixel)


# TODO: implement a FWHM seeing cut... not yet working because PSF getter seems
# to be malfunctioning in step 2
#
#
# # Loop through each unique GROUP_ID and test for bad seeing conditions.
# groupByID = fileIndex.group_by(['GROUP_ID'])
# for subGroup in groupByID.groups:
#     # Grab the FWHM values for this subGroup
#     thisFWHMs = subGroup['FWHM']*u.pixel
#
#     # Grab the median and standard deviation of the seeing for this subgroup
#     medianSeeing = np.median(thisFWHMs)
#     stdSeeing    = np.std(thisFWHMs)
#
#     # Find bad FWHM values
#     badFWHMs = np.logical_not(np.isfinite(subGroup['FWHM']))
#     badFWHMs = np.logical_or(
#         badFWHMs,
#         thisFWHMs <= 0
#     )
#     badFWHM = np.logical_and(
#         badFWHM,
#         thisFWHMs > 2.0*u.arcsec
#     )
#     import pdb; pdb.set_trace()

# Group the fileIndex by...
# 1. Target
# 2. Waveband
fileIndexByTarget = fileIndex.group_by(['TARGET', 'FILTER'])

# Loop through each group
for group in fileIndexByTarget.groups:
    # Grab the current group information
    thisTarget     = str(np.unique(group['TARGET'].data)[0])
    thisFilter     = str(np.unique(group['FILTER'].data)[0])

    # Update the user on processing status
    print('\nProcessing images for')
    print('Target : {0}'.format(thisTarget))
    print('Filter : {0}'.format(thisFilter))

    # Grab the atmospheric extinction coefficient for this wavelength
    thisKappa = kappa[thisFilter]

    # Further divide this group by its constituent POLPOS values
    indexByPolAng = group.group_by(['POLPOS'])

    # Loop over each of the polAng values, as these are independent from
    # eachother and should be treated entirely separately from eachother.
    for polAngGroup in indexByPolAng.groups:
        # Grab the current polAng information
        thisPolAng = np.unique(polAngGroup['POLPOS'].data)[0]

        # Update the user on processing status
        print('\tPol Ang : {0}'.format(thisPolAng))

        # Test if this target should be processed by subgroups
        if thisTarget.upper() in processSubGroupList:
            # If this target has been marked to process in subgroups, then loop
            # through each subgroup and process on the fly.
            indexByGroupID = polAngGroup.group_by(['GROUP_ID'])
            for subGroup in indexByGroupID.groups:
                # Grab the numae of this subGroup and update the user on the
                # current execution status
                thisSubGroup = str(np.unique(subGroup['OBJECT'])[0])
                print('\t\tProcessing images for subgroup {0}'.format(thisSubGroup))

                # Construct the output file name and test if it alread exsits.
                outFile = '_'.join([thisTarget, thisSubGroup, str(thisPolAng)])
                outFile = os.path.join(polAngDir, outFile) + '.fits'
                if os.path.isfile(outFile):
                    print('\t\tFile {0} already exists...'.format(os.path.basename(outFile)))
                    continue

                # Read in all the images
                imgList = []
                progressString = '\t\tNumber of Images : {0}'
                for iFile, filename in enumerate(subGroup['FILENAME']):
                    # Update the user on processing status
                    print(progressString.format(iFile+1), end='\r')

                    # Read in a temporary compy of this image
                    tmpImg = ai.reduced.ReducedScience.read(filename)

                    # Compute the image statistics and store them
                    goodPix= np.isfinite(tmpImg.data)
                    if np.sum(goodPix) > 0:
                        # Find the good pixels
                        goodInds = np.where(goodPix)

                        # Compute the image median value
                        medianValue = np.median(tmpImg.data[goodInds])*tmpImg.unit
                    else:
                        print('There are no good pixels in this image...')
                        pdb.set_trace()

                    # Subtract the background value from each image
                    tmpImg = tmpImg - medianValue

                    # Apply an airmass correction to each of the images
                    tmpImg = tmpImg.correct_airmass(thisKappa)

                    # Apped the image to the imgList variable
                    imgList.append(tmpImg)

                # Create a new line for shell output
                print('')

                # Construct an ImageStack object
                imgStack = ai.utilitywrappers.ImageStack(imgList)

                # Align the images
                imgStack.align_images_with_cross_correlation(
                    subPixel=False,
                    padding=np.NaN
                )

                # Combine the images
                outImg = imgStack.combine_images()

                # Save the image
                outImg.write(outFile, dtype=np.float64)

        elif thisTarget.upper() not in processSubGroupList:
            # If this target has been marked to process in subgroups, then loop
            # through each subgroup and process on the fly.
            outFile = '_'.join([thisTarget, thisFilter, str(thisPolAng)])
            outFile = os.path.join(polAngDir, outFile) + '.fits'

            if os.path.isfile(outFile):
                print('\tFile {0} already exists...'.format(os.path.basename(outFile)))
                continue

            # Read in all the images
            imgList = []
            progressString = '\t\tNumber of Images : {0}'
            for iFile, filename in enumerate(polAngGroup['FILENAME']):
                # Update the user on processing status
                print(progressString.format(iFile+1), end='\r')

                # Read in a temporary compy of this image
                tmpImg = ai.reduced.ReducedScience.read(filename)

                # Skip any images with bad astrometry.
                if not tmpImg.has_wcs:
                    continue

                # Crop the edges of this image
                ny, nx = tmpImg.shape
                binningArray = np.array(tmpImg.binning)
                # TODO: COMPUTE the proper cropping to get a (1000, 1000) image
                cx, cy = (np.array([16, 32])/binningArray).astype(int)

                tmpImg = tmpImg[5:ny-5, 5:nx-5]

                # Compute the image statistics and store them
                goodPix= np.isfinite(tmpImg.data)
                if np.sum(goodPix) > 0:
                    # Find the good pixels
                    goodInds = np.where(goodPix)

                    # Compute the image median value
                    medianValue = np.median(tmpImg.data[goodInds])*tmpImg.unit
                else:
                    print('There are no good pixels in this image...')
                    pdb.set_trace()

                # Subtract the background value from each image
                tmpImg  = tmpImg - medianValue

                # Apply an airmass correction to each of the images
                tmpImg = tmpImg.correct_airmass(thisKappa)

                # Apped the image to the imgList variable
                imgList.append(tmpImg)

            # Create a new line for shell output
            print('')

            # Construct an ImageStack object
            imgStack = ai.utilitywrappers.ImageStack(imgList)

            # Align the images
            imgStack.align_images_with_wcs(
                subPixel=False,
                padding=np.NaN
                )

            # Combine the images
            outImg = imgStack.combine_images()

            # Save the image
            outImg.write(outFile, dtype=np.float64)

print('\nDone computing average images!')
