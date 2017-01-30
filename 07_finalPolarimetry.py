# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 09:34:43 2015

@author: jordan
"""

import os
import sys
import subprocess
import numpy as np
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
import pdb

# Add the AstroImage class
from astroimage.astroimage import utils
from astroimage.astroimage import AstroImage

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================

# This is a list of targets which have a hard time with the "cross_correlate"
# alignment method, so use "wcs" method instead
wcsAlignmentList = ['NGC7023', 'NGC2023']

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data\\201501\\'

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
calTableFile = os.path.join(pyPol_data, 'polCalConstants.csv')
calTable = Table.read(calTableFile, format='ascii.csv')

# Determine which parts of the fileIndex pertain to science images
useFiles = np.logical_and((fileIndex['Use'] == 1), (fileIndex['Dither'] == 'ABBA'))

# Cull the file index to only include files selected for use
fileIndex = fileIndex[np.where(useFiles)]

# Group the fileIndex by...
# 1. Target
# 2. Waveband
# 3. Dither (pattern)
fileIndexByTarget = fileIndex.group_by(['Target', 'Dither'])

# Loop through each group
groupKeys = fileIndexByTarget.groups.keys
for group in fileIndexByTarget.groups:
    # Grab the current target information
    thisTarget = str(np.unique(group['Target'].data)[0])
    thisDither = str(np.unique(group['Dither'].data)[0])

    print('\nProcessing images for')
    print('\tTarget : {0}'.format(thisTarget))
    print('\tDither : {0}'.format(thisDither))

    # Initalize some dictionaries to store images and masks
    wavebandList     = []
    polAngList       = []
    polAngImgList    = []
    maskImgList      = []
    footprintImgList = []

    # Loop through available wavebands and read in all the images
    print('\n\tReading images for\n')
    wavebands = np.unique(group['Waveband'].data)
    for thisWaveband in wavebands:
        # Update the user on the processing status
        print('\t\tWaveband : {0}'.format(thisWaveband))

        # Loop through available polAngs and read in all the images
        polAngs = np.unique(group['Pol Ang'].data)
        for polAng in polAngs:
            # Update the user on the processing status
            thisPolAng = str(polAng)
            print('\t\t\tPol Ang : {0}'.format(thisPolAng))

            # Build the filename to read in
            fileName = os.path.join(polAngDir,
                '_'.join([thisTarget, thisWaveband, thisPolAng]) + '.fits')

            # If the file exists, then read it into a dictionary
            if os.path.isfile(fileName):
                # Read in the polAng image
                polAngImg = AstroImage(fileName)

                # Find the masked pixels for this image and store them
                maskImg = polAngImg.copy()
                maskArr = (np.logical_not(np.isfinite(polAngImg.arr))).astype(int)
                maskImg.arr = maskArr

                # Remove unnecessary 'sigma' arrays
                if hasattr(maskImg, 'sigma'):
                    del maskImg.sigma

                # Store a simple map containing the image footprint
                # Before the images are aligned, this is simply an array of ones
                footprintImg     = polAngImg.copy()
                footprintImg.arr = np.ones(polAngImg.arr.shape, dtype=int)

                # Remove unnecessary 'sigma' arrays
                if hasattr(footprintImg, 'sigma'):
                    del footprintImg.sigma

                # Append images and "keys" to lists
                wavebandList.append(thisWaveband)
                polAngList.append(polAng)
                polAngImgList.append(polAngImg)
                maskImgList.append(maskImg)
                footprintImgList.append(footprintImg)
            else:
                raise StandardError('Some required Polaroid Angle files are missing.')

    # Convert the wavebandList and polAngList to numpy arrays
    wavebandList = np.array(wavebandList)
    polAngList   = np.array(polAngList)

    # Use the "align_stack" method to align the newly created image list
    print('\n\t\tAligning *ALL* polAng images\n')

    # Test which kind of alignment the user has recommended for this target
    if thisTarget in wcsAlignmentList:
        # Use WCS method to compute image offsets
        imgOffsets = utils.get_img_offsets(polAngImgList,
            mode='wcs', subPixel=True)

        # Use the image offsets to align the polAngImgs, maskImgs, and footprint
        alignedPolAngImgs = utils.align_images(polAngImgList,
            offsets=imgOffsets, subPixel=True, padding=np.NaN)

        alignedMaskImgs = utils.align_images(maskImgList,
            offsets=imgOffsets, subPixel=True, padding=1)

        alignedPixMaps = utils.align_images(footprintImgList,
            offsets=imgOffsets, subPixel=True, padding=0)

        pixelCountImg  = np.sum(alignedPixMaps)
    else:
        # Otherwise use cross_correlate method (more accurate if it works)
        imgOffsets = utils.get_img_offsets(polAngImgList,
            mode='cross_correlate', subPixel=True)

        # Use the image offsets to align the polAngImgs, maskImgs, and footprint
        alignedPolAngImgs = utils.align_images(polAngImgList,
            offsets=imgOffsets, subPixel=True, padding=np.NaN)

        alignedMaskImgs = utils.align_images(maskImgList,
            offsets=imgOffsets, subPixel=True, padding=1)

        alignedPixMaps = utils.align_images(footprintImgList,
            offsets=imgOffsets, subPixel=True, padding=0)

        pixelCountImg  = np.sum(alignedPixMaps)

    # Now that the images have all been aligned, we can crop them to only
    # include areas which have sufficient coverage in all constituent images
    # Grab the image center pixel coordinates
    ny, nx = pixelCountImg.arr.shape
    yc, xc = ny//2, nx//2

    # Find the actual cut points for the cropping of the "common image"
    maxCount = len(alignedPolAngImgs)
    goodCol  = np.where(np.abs(pixelCountImg.arr[:,xc] - maxCount) < 1e-2)
    goodRow  = np.where(np.abs(pixelCountImg.arr[yc,:] - maxCount) < 1e-2)
    bt, tp   = np.min(goodCol) + 5, np.max(goodCol) - 5
    lf, rt   = np.min(goodRow) + 5, np.max(goodRow) - 5

    # Replace the masked pixels in the "arr" attribute with NaNs and crop the
    # images
    for imgNum, mask in enumerate(alignedMaskImgs):
        # Make a copy of this alignedPolAngImg to manipulate and re-save
        tmpImg = alignedPolAngImgs[imgNum].copy()

        # If some pixels are masked, replace them with NaN
        tmpMask = mask.arr > 0
        if np.sum(tmpMask) > 0:
            # Locate the for masked pixels
            maskInds = np.where(tmpMask)
            tmpImg.arr[maskInds] = np.NaN

        # Now that any necessary masking has been applied, let's proceed to
        # crop each image to only include well represented pixels
        tmpImg.crop(lf, rt, bt, tp)

        # Store the masked image in the polAngImgs list
        alignedPolAngImgs[imgNum] = tmpImg

    # Build a dictionary to store
    alignedPolAngImgs = np.array(alignedPolAngImgs)
    polAngImgDict = {}
    for waveband in wavebands:
        imgInds = np.where(wavebandList == waveband)
        subDict = dict(zip(polAngList[imgInds], alignedPolAngImgs[imgInds]))
        polAngImgDict[waveband] = subDict

    # Now that everything has been aligned and stored properly in a dictionary,
    # loop back through the dictionary and compute Stokes images
    print('\tComputing Stokes images for\n')
    for thisWaveband, polAngImgs in polAngImgDict.items():
        # Update the user on processing status
        print('\t\tWaveband : {0}'.format(thisWaveband))

        # Grab the proper row for polarimetric calibration
        thisRow = np.where(calTable['Waveband'] == thisWaveband)

        #**********************************************************************
        # Stokes I
        #**********************************************************************
        # Average the images to get stokes I
        stokesI = utils.combine_images([polAngImgs[0],
                                        polAngImgs[200],
                                        polAngImgs[400],
                                        polAngImgs[600]])
        stokesI = 2 * stokesI

        # Perform astrometry to apply to the headers of all the other images...
        stokesI.clear_astrometry()
        stokesI.filename = 'tmp.fits'
        stokesI.write()
        stokesI, success = utils.solve_astrometry(stokesI)

        # Check if astrometry solution was successful
        if not success:
            # If it was not a success, then pause and examine the results
            pdb.set_trace()
        else:
            # If the astrometry was successful, then delete the temporary file.
            # Start by setting the appropriate commands
            if 'win' in sys.platform:
                delCmd = 'del '
                shellCmd = True
            else:
                delCmd = 'rm '
                shellCmd = False

            # Use a spawned subprocess to delete the file before proceeding
            rmProc = subprocess.Popen(delCmd + 'tmp.fits', shell=shellCmd)
            rmProc.wait()
            rmProc.terminate()

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
        Pmap, PAmap = utils.build_pol_maps(stokesQ, stokesU)

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

print('Done processing images!')
