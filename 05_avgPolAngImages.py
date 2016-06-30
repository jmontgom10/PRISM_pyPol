# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 17:03:39 2015
@author: jordan
"""

import os
import sys
from datetime import datetime
import warnings
import numpy as np
from astropy.io import ascii
from astropy.table import Table as Table
from astropy.table import Column as Column
from astropy.convolution import Gaussian2DKernel
from astropy.modeling import models, fitting
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from photutils import detect_sources, Background
import subprocess
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
fileIndexByTarget = fileIndex.group_by(['Target', 'Waveband', 'Dither', 'Pol Ang'])

# Loop through each group
groupKeys = fileIndexByTarget.groups.keys
for group in fileIndexByTarget.groups:
    # Grab the current target information
    thisTarget   = str(np.unique(group['Target'].data)[0])
    thisWaveband = str(np.unique(group['Waveband'].data)[0])
    thisPolAng   = str(np.unique(group['Pol Ang'].data)[0])

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

    # Convert the image list into an indexable array
    imgList = np.array(imgList)

    # Process images in this group according to dither type
    ditherType = np.unique(group['Dither'].data)
    if ditherType == "ABBA":
        # Break the images up into "on-target" / "off-target" image lists
        Ainds = np.where(group['ABBA'].data == 'A')[0]
        Binds = np.where(group['ABBA'].data == 'B')[0]
        Aimgs = imgList[Ainds]
        Bimgs = imgList[Binds]

        # Setup some initial time to compute relative observation times
        dt0 = datetime(2000,1,1,0,0,0)

        # Loop through all "on-target" images and grab the observation time and
        # background sky count levels
        Atimes = []
        Abkgs = []
        maskImgList = []
        for Aimg in Aimgs:
            # Catch all the NaN values from masking the optical ghosts
            ghostMask    = np.logical_not(np.isfinite(Aimg.arr))
            ghostMaskImg = Aimg.copy()
            ghostMaskImg.arr = ghostMask
            maskImgList.append(ghostMaskImg)

            # Compute the image statistics and store them
            goodVals = np.isfinite(Aimg.arr)
            if np.sum(goodVals) > 0:
                goodInds = np.where(goodVals)
                mean, median, std = sigma_clipped_stats(Aimg.arr[goodInds])
                Abkgs.append(median)
            else:
                print('There are no good pixels in this on-target image...')
                pdb.set_trace()

            # Compute the relative observation time and store it
            dt = datetime.strptime(Aimg.header['date-obs'][0:19], '%Y-%m-%dT%H:%M:%S')
            Atimes.append((dt - dt0).total_seconds())

        # Convert maskImgList into an array
        maskImgs = np.array(maskImgList)

        # Loop through the background images and compute the background images,
        # observation times, and background sky count levels
        print('\tComputing average normalized background image')
        Btimes = []
        Bbkgs = []
        bkgImgs = []
        for Bimg in Bimgs:
            # Estimate the background for this off-target image
            B1bkg     = Background(Bimg.arr, (100, 100), filter_shape=(3, 3),
                                   method='median')
            threshold = B1bkg.background + 3.0*B1bkg.background_rms

            # Build a mask for any sources above the 3-sigma threshold
            sigma  = 2.0 * gaussian_fwhm_to_sigma    # FWHM = 2.
            kernel = Gaussian2DKernel(sigma, x_size=6, y_size=6)
            kernel.normalize()
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

            # Store the *NORMALIZED* background image and statistics statistics.
            bkgImgs.append(B1bkg.background/B1bkg.background_median)
            Bbkgs.append(B1bkg.background_median)

            # Compute the relative observation time and store it
            dt = datetime.strptime(Bimg.header['date-obs'][0:19], '%Y-%m-%dT%H:%M:%S')
            Btimes.append((dt - dt0).total_seconds())

        # Compute average *NORMALIZED* background image
        bkgImgs = np.array(bkgImgs)
        bkgImg  = np.mean(bkgImgs, axis=0)

        # Adjust the Btimes to be *JUST* barely positive
        t0      = np.min(Atimes) - 10
        Atimes -= t0
        Btimes -= t0

        # Perform the least squares fit to the background levels
        # Initalize a set of models to fit and a fitting object
        powerLaw_init  = models.PowerLaw1D(amplitude=1.0, x_0=1.0, alpha=-1.0)
        line_init      = models.Polynomial1D(1)
        compModel_init = line_init + powerLaw_init
        fitter         = fitting.LevMarLSQFitter()

        with warnings.catch_warnings():
            # Ignore warning from the fitter
            warnings.simplefilter("ignore")
            bkgModel = fitter(compModel_init, Btimes, Bbkgs)

        # import matplotlib.pyplot as plt
        # plt.ion()
        # plt.scatter(Atimes, Abkgs, c='blue')
        # plt.scatter(Btimes, Bbkgs, c='red')
        # plt.title('PolAng = {0}'.format(thisPolAng))
        # plt.autoscale(False)
        #
        # # Generate an array of interpolate times...
        # trange = np.max(plt.xlim()) - np.min(plt.xlim())
        # t_tmp  = trange*np.linspace(0,1,100) + np.min(plt.xlim())
        # plt.plot(t_tmp, bkgModel(t_tmp))
        # plt.savefig(thisPolAng+'.png')
        # pdb.set_trace()
        # continue

        # Cut out any Aimgs with approximate residual levels more than 3-sigma
        # from the median.
        AbkgResid = bkgModel(Atimes) - Abkgs
        goodVals  = np.abs((AbkgResid - np.median(AbkgResid))/np.std(AbkgResid)) < 2.0
        if np.sum(goodVals) > 0:
            keepInds    = np.where(goodVals)
            Aimgs1      = Aimgs[keepInds]
            Atimes1     = Atimes[keepInds]
            maskImgList = list(maskImgs[keepInds])
            mean_bkg    = np.mean((bkgModel(Atimes))[keepInds])
        else:
            print('No interpolated background levels are acceptable...')
            pdb.set_trace()

        # Loop through each preserved "on-target" image and compute the
        # background subtracted "scienc-image"
        print('Subtracting background levels from on-target images')
        sciImgList = []
        for Aimg, Atime, maskImg in zip(Aimgs1, Atimes1, maskImgList):
            # Make a copy of the "on-target" imge to manipulate
            Atmp = Aimg.copy()

            # Catch any non-finite values and "mask" them with -1e6 value
            # These will be caught in the next step and reset to NaNs
            nanInds = np.where(maskImg.arr)
            Atmp.arr[nanInds] = -1e6

            # Catch saturated negative values
            badInds = np.where(Atmp.arr < -3*read_noise/effective_gain)
            Atmp.arr[badInds] = np.NaN

            # Perform the actual background subtraction by scaling the average
            # *NORMALIZED* background image to the interpolated background level
            Atmp.arr = Atmp.arr - bkgModel(Atime)*bkgImg

            # Append the background subtracted image to the list
            sciImgList.append(Atmp.copy())

        # Now that all the backgrounds have been subtracted,
        # let's align the images and compute an average image
        # TODO Should I use "cross-correlation" alignment?
        sciImgList  = image_tools.align_images(
            sciImgList,
            padding=np.NaN,
            mode='cross_correlate',
            subPixel=True)
        avgImg = image_tools.combine_images(
            sciImgList,
            output = 'MEAN',
            mean_bkg = mean_bkg,
            effective_gain = effective_gain,
            read_noise = read_noise)

    else:
        # Handle HEX-DITHERS images
        # (mostly for calibration images as of now)
        continue
        ##### The Indexing Program should parse hex pointing positions, too.
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

    print('\tRe-solve the average image astrometry.')
    # It is only possible to "redo the astrometry" if the file on the disk
    # First make a copy of the average image
    tmpImg = avgImg.copy()

    # Clear the astrometry values from the header
    tmpImg.clear_astrometry()

    # ReplaceNaNs with something finite and name the file "tmp.fits"
    tmpImg.arr = np.nan_to_num(tmpImg.arr)
    tmpImg.filename = 'tmp.fits'

    # Delete the sigma attribute
    if hasattr(tmpImg, 'sigma'):
        del tmpImg.sigma

    # Record the temporary file to disk for performing astrometry
    tmpImg.write()

    # Solve the stacked image astrometry
    avgImg1, success = image_tools.astrometry(tmpImg)

    # With successful astrometry, save result to disk
    if success:
        print('astrometry succeded')

        # Clean up temporary variable
        del tmpImg

        # Delete the temporary file
        # Test what kind of system is running
        if 'win' in sys.platform:
            # If running in Windows,
            delCmd = 'del '
            shellCmd = True
        else:
            # If running a *nix system,
            delCmd = 'rm '
            shellCmd = False

        # Finally delete the temporary fits file
        tmpFile = os.path.join(os.getcwd(), 'tmp.fits')
        rmProc  = subprocess.Popen(delCmd + tmpFile, shell=shellCmd)
        rmProc.wait()
        rmProc.terminate()

        # Now that astrometry has been solved, let's make sure to go through and
        # mask out all pixels with zero samples.
        maskImgList = image_tools.align_images(maskImgList, padding=np.NaN)
        maskCount   = np.zeros(maskImgList[0].arr.shape, dtype=int)
        for mask in maskImgList:
            maskCount += mask.arr.astype(int)

        # Make copies of the image array
        tmpArr = avgImg.arr.copy()
        tmpSig = avgImg.sigma.copy()

        # Blank out pixels which were masked in all images
        maskPix  = (maskCount == len(maskImgList))
        if np.sum(maskPix) > 0:
            maskInds = np.where(maskPix)
            tmpArr[maskInds] = np.NaN

            # Replace the sigma array
            if hasattr(avgImg, 'sigma'):
                tmpSig[maskInds] = np.NaN

        # Replace the image array
        avgImg1.arr   = tmpArr
        avgImg1.sigma = tmpSig

        # Save the file to disk
        avgImg1.write(outFile)
    else:
        print('astrometry failed?!')
        pdb.set_trace()

print('\nDone computing average images!')
