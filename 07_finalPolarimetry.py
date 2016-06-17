# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 09:34:43 2015

@author: jordan
"""

import os
import sys
import numpy as np
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
import pdb

# Add the AstroImage class
sys.path.append("C:\\Users\\Jordan\\Libraries\\python\\AstroImage")
import image_tools
from AstroImage import AstroImage

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================

# This is a dictionary of target/waveband combinations which have a hard time
# with the "cross_correlate" alignment method, so use "wcs" method instead
wcsAlignmentDict = {'NGC7023': ('V', 'R'),
                    'NGC2023': ('V', 'R')}

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data'

# Setup new directory for polarimetry data
polarimetryDir = os.path.join(pyPol_data, 'Polarimetry')
if (not os.path.isdir(polarimetryDir)):
    os.mkdir(polarimetryDir, 0o755)

polAngDir = os.path.join(polarimetryDir, 'polAngImgs')
if (not os.path.isdir(polAngDir)):
    os.mkdir(polAngDir, 0o755)

stokesDir = os.path.join(polarimetryDir, 'stokesImgs')
if (not os.path.isdir(stokesDir)):
    os.mkdir(stokesDir, 0o755)

# Read in the indexFile data and select the filenames
print('\nReading file index from disk')
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='ascii.csv')

print('\nReading calibration constants from disk')
calTableFile = os.path.join(pyPol_data, 'calData.csv')
calTable = Table.read(calTableFile, format='ascii.csv')

# Determine which parts of the fileIndex pertain to science images
useFiles = np.logical_and((fileIndex['Use'] == 1), (fileIndex['Dither'] == 'ABBA'))

# Cull the file index to only include files selected for use
fileIndex = fileIndex[np.where(useFiles)]

# Group the fileIndex by...
# 1. Target
# 2. Waveband
# 3. Dither (pattern)
fileIndexByTarget = fileIndex.group_by(['Target', 'Waveband', 'Dither'])

# Loop through each group
groupKeys = fileIndexByTarget.groups.keys
for group in fileIndexByTarget.groups:
    # Grab the current target information
    thisTarget   = str(np.unique(group['Target'].data)[0])
    thisWaveband = str(np.unique(group['Waveband'].data)[0])
    thisRow      = np.where(calTable['Waveband'] == thisWaveband)
    numImgs      = len(group)

    if thisTarget != 'NGC2023': continue

    print('\nProcessing images for'.format(numImgs))
    print('\tTarget   : {0}'.format(thisTarget))
    print('\tWaveband : {0}'.format(thisWaveband))

    # Read in the polAng image
    fileCheck  = True
    polAngImgs = []
    maskImgs   = []
    polAngs    = []
    for polAng in range(0,601,200):
        fileName = os.path.join(polAngDir,
            '_'.join([thisTarget, thisWaveband, str(polAng)]) + '.fits')

        # If the file exists, then read it into a dictionary
        if os.path.isfile(fileName):
            # Read in the polAng image
            polAngImg = AstroImage(fileName)

            # Find the masked pixels for this image and store them
            maskArr = (np.logical_not(np.isfinite(polAngImg.arr))).astype(int)
            maskImg = polAngImg.copy()

            # If there are masked pixels, then handle them
            if np.sum(maskArr) > 0:
                # # Locate the masked pixels
                # # goodInds = np.where(np.logical_not(maskArr))
                # maskInds = np.where(maskArr)
                #
                # # Inpaint the screwed up pixels
                # polAngImg.arr[maskInds] = np.NaN
                # # polAngImg.arr = image_tools.inpaint_nans(polAngImg.arr)
                #
                # maskImg.arr   = polAngImg.arr
                maskImg.sigma = maskArr
            else:
                maskImg.sigma = maskArr

            # Append the values to their lists
            polAngs.append(polAng)
            polAngImgs.append(polAngImg)
            maskImgs.append(maskImg)
        else:
            fileCheck = False

    # Check that all the necessarry files are present
    if not fileCheck:
        print("\tSome required Polaroid Angle files are missing.")
        continue

    # Use the "align_stack" method to align the newly created image list
    print('\nAligning images\n')
    # import matplotlib.pyplot as plt
    # plt.ion()
    # plt.imshow(maskImgs[0].sigma)
    # pdb.set_trace()
    if ((thisTarget in wcsAlignmentDict) and
        (thisWaveband in wcsAlignmentDict[thisTarget])):
        # Use WCS method to align images
        polAngImgs = image_tools.align_images(polAngImgs,
            mode='wcs', subPixel=True, padding=np.NaN)

        # Repeat the alignment on the images which have the mask stored in the
        # sigma attribute. This sholud allow us to RE-MASK after alignment
        alignedMaskImgs = image_tools.align_images(maskImgs,
            mode='wcs', subPixel=True, padding=1)
    else:
        # Otherwise use cross_correlate method (more accurate if it works)
        polAngImgs = image_tools.align_images(polAngImgs,
            mode='cross_correlate', subPixel=True, padding=np.NaN)

        # Repeat the alignment on the images which have the mask stored in the
        # sigma attribute. This sholud allow us to RE-MASK after alignment
        alignedMaskImgs = image_tools.align_images(maskImgs,
            mode='cross_correlate', subPixel=True, padding=1)

    # Replace the masked pixels in the "arr" attribute with NaNs
    for imgNum in range(len(alignedMaskImgs)):
        # Copy the image and look for masked pixels
        tmpImg   = polAngImgs[imgNum].copy()
        tmpMask  = alignedMaskImgs[imgNum].copy()
        maskInds = np.where(tmpMask.sigma > 0)

        # If some pixels are masked, replace them with NaN
        if len(maskInds[0]) > 0:
            tmpImg.arr[maskInds] = np.NaN

            # Store the masked image in the polAngImgs list
            polAngImgs[imgNum] = tmpImg

    # Convert the list into a dictionary
    polAngImgs = dict(zip(polAngs, polAngImgs)) #aligned

    #**********************************************************************
    # Stokes I
    #**********************************************************************
    # Average the images to get stokes I
    stokesI = image_tools.combine_images([polAngImgs[0],
                                          polAngImgs[200],
                                          polAngImgs[400],
                                          polAngImgs[600]])
    stokesI = 2 * stokesI

    # Perform astrometry to apply to the headers of all the other images...
    stokesI, success = image_tools.astrometry(stokesI)

    # Check if astrometry solution was successful
    if not success: pdb.set_trace()

    #**********************************************************************
    # Stokes Q
    #**********************************************************************
    # Subtract the images to get stokes Q
    A = polAngImgs[0] - polAngImgs[400]
    B = polAngImgs[0] + polAngImgs[400]

    # Divide the difference images
    stokesQ = A/B

    # Update the header to include the new astrometry
    stokesQ.header = stokesI.header

    # Mask out masked values
    Qmask = np.logical_or(np.isnan(polAngImgs[0].arr),
                          np.isnan(polAngImgs[400].arr))

    #**********************************************************************
    # Stokes U
    #**********************************************************************
    # Subtact the images to get stokes Q
    A = polAngImgs[200] - polAngImgs[600]
    B = polAngImgs[200] + polAngImgs[600]

    # Divide difference images
    stokesU = A/B

    # Update the header to include the new astrometry
    stokesU.header = stokesI.header

    # Mask out zero values
    Umask = np.logical_or(np.isnan(polAngImgs[200].arr),
                          np.isnan(polAngImgs[600].arr))

    #**********************************************************************
    # Calibration
    #**********************************************************************
    # Construct images to hold PE, sig_PE, deltaPA, and sig_deltePA
    PE       = stokesI.copy()
    PE.arr   = calTable[thisRow]['PE'].data[0]*np.ones_like(stokesI.arr)
    PE.sigma = calTable[thisRow]['s_PE'].data[0]*np.ones_like(stokesI.arr)

    # Grab the calibration data
    PAsign  = calTable[thisRow]['PAsign'].data[0]
    deltaPA = calTable[thisRow]['dPA'].data[0]
    deltaPArad = np.deg2rad(deltaPA)
    s_dPA   = calTable[thisRow]['s_dPA'].data[0]

    # Normalize the U and Q values by the polarization efficiency
    # and correct for the instrumental rotation direction
    stokesQ = 1.0    * stokesQ / PE
    stokesU = PAsign * stokesU / PE

    # Store the deltaPA values in the stokesQ and stokesQ headers
    stokesQ.header['DELTAPA'] = deltaPA
    stokesU.header['DELTAPA'] = deltaPA
    stokesQ.header['S_DPA'] = s_dPA
    stokesU.header['S_DPA'] = s_dPA

    #**********************************************************************
    # Build the polarization maps
    #**********************************************************************
    Pmap, PAmap = image_tools.build_pol_maps(stokesQ, stokesU)

    #**********************************************************************
    # Final masking and writing to disk
    #**********************************************************************
    # Generate a complete mask of all bad elements
    fullMask = np.logical_or(Qmask, Umask)

    stokesU.arr[np.where(Umask)]  = np.NaN
    stokesQ.arr[np.where(Qmask)]  = np.NaN
    Pmap.arr[np.where(fullMask)]  = np.NaN
    PAmap.arr[np.where(fullMask)] = np.NaN

    # Generate the output filenames
    Ifile = os.path.join(stokesDir,
        '_'.join([thisTarget, thisWaveband, 'I']) + '.fits')
    Ufile = os.path.join(stokesDir,
        '_'.join([thisTarget, thisWaveband, 'U']) + '.fits')
    Qfile = os.path.join(stokesDir,
        '_'.join([thisTarget, thisWaveband, 'Q']) + '.fits')
    Pfile = os.path.join(polarimetryDir,
        '_'.join([thisTarget, thisWaveband, 'P']) + '.fits')
    PAfile = os.path.join(polarimetryDir,
        '_'.join([thisTarget, thisWaveband, 'PA']) + '.fits')

    # Write to disk
    stokesI.write(Ifile)
    stokesU.write(Ufile)
    stokesQ.write(Qfile)
    Pmap.write(Pfile)
    PAmap.write(PAfile)
