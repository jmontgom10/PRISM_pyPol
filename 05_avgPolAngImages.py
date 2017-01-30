# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 17:03:39 2015
@author: jordan
"""

import os
import sys
import copy
from datetime import datetime
import warnings
import numpy as np
from astropy.io import ascii
from astropy.table import Table as Table
from astropy.table import Column as Column
from astropy.convolution import Gaussian2DKernel
from astropy.modeling import models, fitting
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from photutils import detect_sources, SigmaClip, Background2D, MedianBackground
from matplotlib import pyplot as plt
import pdb

# Add the AstroImage class
from astroimage.astroimage import utils
from astroimage.astroimage import AstroImage

# This script will run the image averaging step of the pyPol reduction

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
processSubGroupList = ['Taurus_Cal', 'Orion_Cal']

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data\\201501'

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
fileIndexByTarget = fileIndex.group_by(['Target', 'Waveband', 'Dither'])

# Loop through each group
groupKeys = fileIndexByTarget.groups.keys
for group in fileIndexByTarget.groups:
    # Grab the current group information
    thisTarget   = str(np.unique(group['Target'].data)[0])
    thisWaveband = str(np.unique(group['Waveband'].data)[0])
    thisDither   = str(np.unique(group['Dither'].data)[0])

    # Grab all the unique polaroid angle values reperesented within this group.
    polAngs = np.unique(group['Pol Ang'].data)

    # Update the user on processing status
    print('\nProcessing images for')
    print('Target   : {0}'.format(thisTarget))
    print('Waveband : {0}'.format(thisWaveband))
    print('Dither   : {0}'.format(thisDither))

    # Loop through the polAngs and check if these images have already been done
    fileCount = 0
    for polAng in polAngs:
        # Test if this group was previously processed.
        testFile = '_'.join([thisTarget, thisWaveband, str(polAng)])
        testPath = os.path.join(polAngDir, testFile) + '.fits'

        # Now that the filename has been constructed, test for file existence
        if os.path.isfile(testPath):
            print('File ' + testFile + ' already exists...')
            fileCount += 1

    # If all four polaroid angle images have been processed, then simply skip
    # this target/waveband combination for now
    if fileCount > 3: continue

    # Initalize a dictionary in which to store the lists of polAng images to be
    # combined and their background levels, too.
    polAngDict = {'images':[], 'background_levels':[]}
    polAngDict = {
        0:   copy.deepcopy(polAngDict),
        200: copy.deepcopy(polAngDict),
        400: copy.deepcopy(polAngDict),
        600: copy.deepcopy(polAngDict)
    }

    # Initalize a groupDict to store all the background subtracted images. If
    # this target will store subGroup information, then initalize a blank
    # dictionary so that copies of the polAngDict can be placed into individual
    # keys for each subGroup. Otherwise, just make the groupDict a straight copy
    # of the polAngDict to store ALL the polAng images from all the subgroups.
    if thisTarget in processSubGroupList:
        groupDict = {}
    else:
        groupDict = copy.deepcopy(polAngDict)


    # If the code proceeds to this point, then it must process at least one
    # subGroup. Thus, let's initalize a counter to keep track of that.
    subGroupCount = 0

    # Test which kind of dither we're dealing with and handle accordingly
    if thisDither == 'ABBA':
        # Now break the TARGET group up by its individual observational groupings
        indexBySubGroup = group.group_by(['Name'])

        # Set the background estimation properties using the following classes
        # provided by the photutils package
        sigma_clip    = SigmaClip(sigma=3., iters=10)
        bkg_estimator = MedianBackground()

        # In the following loop through each polaroid rotation anglue, we
        # will test for sunrise and sunset powerlaws in the background
        # levels of each polAng set. If the sun is setting, then a
        # decreasing power law (alpha = -1) should fit the data well. If the
        # sun is rising, then an increasing power law (alpha = +1) should
        # fit the data well.

        # Initalize sunset power law for model fitting
        sunsetPowerLaw_init = models.PowerLaw1D(
            amplitude=1.0,
            x_0=1.0,
            alpha=-1.0)

        # Initalize sunrise power law for model fitting
        sunrisePowerLaw_init = models.PowerLaw1D(
            amplitude=1.0,
            x_0=1.0,
            alpha=+1.0)

        # OLD METHOD MAY HAVE USED LINEAR FIT TO DO INTERPOLATION
        # # Initalize linear polynomial variation in background values
        # line_init = models.Polynomial1D(1)

        # Initalize model fitter for use
        fitter = fitting.LevMarLSQFitter()

        # Setup some initial time to compute relative observation times
        dt0 = datetime(2000,1,1,0,0,0)

        # Loop through each observational grouping. This allows the time-varying
        # background to be estimated using only data closely spaced in time.
        # Trying to parse all the data for each target/waveband combination
        # simultaneously would not result in accurate background estimations.
        subGroupKeys = indexBySubGroup.groups.keys
        for subGroup in indexBySubGroup.groups:
            # Update the user on the current execution status
            thisSubGroup = str(np.unique(subGroup['Name'])[0])
            print('\tProcessing background levels for subgroup {0}'.format(thisSubGroup))

            # If this group needs to be processed with independent keys for each
            # "subgroup", then test whether those files have already been done.
            if thisTarget in processSubGroupList:
                # Loop through the polAngs and check if these images have already been done
                fileCount     = 0
                for polAng in polAngs:
                    # Test if this group was previously processed.
                    testFile = '_'.join([thisTarget, thisSubGroup, str(polAng)])
                    testPath = os.path.join(polAngDir, testFile) + '.fits'

                    # Now that the filename has been constructed, test for file existence
                    if os.path.isfile(testPath):
                        print('\tFile ' + testFile + ' already exists...')
                        fileCount += 1

                if fileCount > 3:
                    # If all four polaroid angle images have been processed,
                    # then simply skip this subgroup for now.
                    continue
                else:
                    # otherwise proceed AND increment the subGroupCount by 1
                    subGroupCount += 1

                # Now that we have determined that this subGroup SHOULD be
                # processed, let's add an entry to the groupDict to store
                # the data for this subGroup.
                groupDict[thisSubGroup] = copy.deepcopy(polAngDict)

            # For an ABBA dither, we need to treat each polAng value separately.
            # Start by breaking the subGroup up into its constituent polAngs
            indexByPolAng  = subGroup.group_by(['Pol Ang'])

            # Loop through each polAng subset of the subGroup
            polAngGroupKeys = indexByPolAng.groups.keys
            for polAngGroup in indexByPolAng.groups:
                # Update the user on processing status
                thisPolAng = str(np.unique(polAngGroup['Pol Ang'])[0])
                print('\t\tPolaroid Angle : {0}'.format(thisPolAng))

                # Initalize temporary lists to hold the images, times, etc....
                Aimgs       = []
                Atimes      = []
                AmedBkgs    = []
                Btimes      = []
                BmedBkgs    = []
                bkgImgs     = []
                AinterpVals = []

                # Loop through each row of the polAng group, read in the images,
                # grab the observation times, estimate the background levels,
                # estimate the median, normalized, background images, and
                # perform the model fitting and interpolated on-target
                # background levels
                progressString = '\t\t\tImages : '
                for row in polAngGroup:
                    # Read in a temporary compy of this image
                    file1  = row['Filename']
                    tmpImg = AstroImage(file1)

                    # Compute the relative observation time and store it
                    dt = datetime.strptime(tmpImg.header['date-obs'][0:19],
                        '%Y-%m-%dT%H:%M:%S')

                    # Treat the "on-target" files
                    if row['ABBA'] == 'A':
                        progressString += 'A'
                        print(progressString, end='\r')

                        # Add the unaltered image to the Aimg list
                        Aimgs.append(tmpImg)

                        # Test if this file has an associated mask
                        maskFile = os.path.join(maskDir, os.path.basename(file1))
                        if os.path.isfile(maskFile):
                            # Read in any associated mask
                            tmpMask = AstroImage(maskFile)

                            # If there are some bad pixels to mask....
                            if np.sum(tmpMask.arr) > 0:
                                # Replace those pixels with nans
                                tmpImg.arr[np.where(tmpMask.arr)] = np.nan

                        # Store the relative observing time in the Atimes list
                        Atimes.append((dt - dt0).total_seconds())

                        # Compute the image statistics and store them
                        goodPix= np.isfinite(tmpImg.arr)
                        if np.sum(goodPix) > 0:
                            goodInds = np.where(goodPix)
                            mean, median, std = sigma_clipped_stats(
                                tmpImg.arr[goodInds])
                            AmedBkgs.append(median)
                        else:
                            print('There are no good pixels in this on-target image...')
                            pdb.set_trace()

                    if row['ABBA'] == 'B':
                        progressString += 'B'
                        print(progressString, end='\r')

                        # Store the relative observing time in the Btimes list
                        Btimes.append((dt - dt0).total_seconds())

                        # Estimate the background for this off-target image
                        B1bkg     = Background2D(tmpImg.arr, (100, 100),
                            filter_size=(3, 3),
                            sigma_clip=sigma_clip,
                            bkg_estimator=bkg_estimator)
                        threshold = B1bkg.background + 3.0*B1bkg.background_rms

                        # Build a mask for any sources above the 3-sigma
                        # threshold. Start by simply locating all ~2.0 pixel
                        # gaussian sources.
                        sigma  = 2.0 * gaussian_fwhm_to_sigma    # FWHM = 2.
                        kernel = Gaussian2DKernel(sigma, x_size=6, y_size=6)
                        kernel.normalize()
                        segm   = detect_sources(tmpImg.arr, threshold,
                                              npixels=5, filter_kernel=kernel)

                        # Build the actual mask and include a step to capture
                        # negative saturation values.
                        mask   = np.logical_or((segm.data > 0),
                                 (np.abs((tmpImg.arr -
                                  B1bkg.background)/B1bkg.background_rms) > 7.0))

                        # Estimate a 2D background image masking possible
                        # sources.
                        B1bkg = Background2D(tmpImg.arr, (100, 100), mask=mask,
                            filter_size=(3, 3),
                            sigma_clip=sigma_clip,
                            bkg_estimator=bkg_estimator)

                        # Store the median background level for this image.
                        BmedBkgs.append(B1bkg.background_median)

                        # Store the *NORMALIZED* background image.
                        bkgImgs.append(B1bkg.background/B1bkg.background_median)

                # Print a single newline character to keep the progressString
                print('')

                # Now that we've looped through all the rows for this polAng
                # part of the subGroup, compute an average of the normalized
                # background images
                bkgImgs = np.array(bkgImgs)
                bkgImgs  = np.mean(bkgImgs, axis=0)
                bkgImgs /= np.median(bkgImgs)

                # Convert into an AstroImage object
                bkgImg = Aimgs[0].copy()
                bkgImg.arr = bkgImgs
                del bkgImgs

                # Next, let's test for various background level models.
                # Grab the minimum observation time for all these images
                minTime = np.min([np.min(Atimes), np.min(Btimes)])

                # Use this as a fiducial time (plus 10 seconds to avoid power
                # law singularities)
                Atimes = np.array(Atimes) - minTime + 10
                Btimes = np.array(Btimes) - minTime + 10

                # Convert median background levels to arrays
                AmedBkgs = np.array(AmedBkgs)
                BmedBkgs = np.array(BmedBkgs)

                # Perform the test fits ignoring some common warnings
                with warnings.catch_warnings():
                    # Ignore warning from the fitter
                    warnings.simplefilter('ignore')
                    sunsetFit  = fitter(sunsetPowerLaw_init, Btimes,
                        BmedBkgs)
                    sunriseFit = fitter(sunrisePowerLaw_init, Btimes,
                        BmedBkgs)

                ######
                # TODO -- rerun M104_V5 and see if I can catch the powerlaws
                # rise in background level...
                #######
                # Examine the fits to see if significant sunrise or senset found
                sunsetTest = (sunsetFit.amplitude.value/np.median(BmedBkgs) > 1.0
                    and sunsetFit.alpha.value < -0.9)
                sunriseTest = (sunriseFit.amplitude.value/np.median(BmedBkgs) > 1.0
                    and sunriseFit.alpha.value > +0.9)

                # Test if only ONE of the two possibilities was detected. If
                # both are false, then nothing is done
                if sunsetTest != sunriseTest:
                    # Now determine which one was present
                    if sunsetTest:
                        print('Detected sunset... subtracting')
                        # Subtract sunset values from AmedBkgs and BmedBkgs
                        AmedBkgs -= sunsetFit(Atimes)
                        BmedBkgs -= sunsetFit(Btimes)
                    elif sunriseTest:
                        print('Detected sunrise... subtracting')
                        # Subtract sunset values from AmedBkgs and BmedBkgs
                        AmedBkgs -= sunriseFit(Atimes)
                        BmedBkgs -= sunriseFit(Btimes)

                # If both tests return True value, then issue an error and stop
                if sunsetTest and sunriseTest:
                    print('Detected both sunrise and setset.... not possible!')
                    pdb.set_trace()

                # Now that there is no sunrise or sunset in the background
                # levels, simply perform direct linear interpolation or
                # extrapolation to get the background level at the Atimes.
                for iTime, Atime in enumerate(Atimes):
                    # Test if this is the first or last image
                    if ((Atime < np.min(Btimes)) or (Atime > np.max(Btimes))):
                        AinterpVals.append(np.nan)

                    # Otherwise it should have bounding neighbors, and
                    # interpolation can be performed
                    else:
                        # Grab the times surrounding the current time
                        ind0 = np.max(np.where(Btimes < Atime))
                        ind1 = np.min(np.where(Btimes > Atime))

                        # Grab the times and background levels for THIS
                        # interpolation procedure
                        Btime0 = Btimes[ind0]
                        Btime1 = Btimes[ind1]
                        Bbkg0  = BmedBkgs[ind0]
                        Bbkg1  = BmedBkgs[ind1]

                        # Solve for the slope and intercept
                        thisSlope     = (Bbkg1 - Bbkg0)/(Btime1 - Btime0)
                        thisIntercept = Bbkg0 - thisSlope*Btime0

                        # Solve for the interpolated background level and store
                        # it in the growing list
                        Abkg = thisSlope*Atime + thisIntercept
                        AinterpVals.append(Abkg)

                # Now that all possible interpolations have been performed,
                # compute a median difference between the measured and
                # interpolated values.
                AinterpVals = np.array(AinterpVals)
                interpInds  = np.where(np.isfinite(AinterpVals))
                diffVals    = AmedBkgs[interpInds] - AinterpVals[interpInds]
                medDiff     = np.median(diffVals)

                # Force medDiff to be either zero or positive. A medDiff value
                # less than zero implies that the on-target images have
                # NEGATIVE background contribution from the Earth's atmosphere.
                # We assume that this cannot be and hence do not allow it.
                if medDiff < 0:
                    medDiff = 0

                # Now that we have an estimate of the median difference between
                # the off-target background level and the on-target background
                # level, simply estimate the true on-target background level as
                # the measured on-target background level minus this difference
                AbkgVals = AmedBkgs - medDiff

                # Plot the background levels and save to disk for examination...
                fig = plt.figure()
                ax  = fig.add_subplot(111)
                ax.scatter(Btimes, BmedBkgs, color='blue')
                ax.scatter(Atimes, AmedBkgs, color='red')
                ax.plot(Atimes, AbkgVals)
                ax.scatter(Atimes, AbkgVals, color='green')
                ax.set_xlabel('Time [sec]')
                ax.set_ylabel('Backhground [ADU]')
                fname = '_'.join([thisTarget, thisSubGroup, thisPolAng]) + '.png'
                fname = os.path.join(bkgPlotDir, fname)
                plt.savefig(fname, dpi=300)
                plt.close('all')

                # Now that the background level has been estimated for the
                # on-target observation times, loop through the on-target images
                # and subtract a normalized then re-scaled background image.
                # This simultaneously applies a "sky-flat" and subtracts the
                # atmospheric sky contribution from the image.
                Aimgs = [Aimg - Abkg*bkgImg for Aimg, Abkg in zip(Aimgs, AbkgVals)]

                # Build a dictionary containing the keys necessary to map into
                # the groupDict dictionary. This dictionary needs to have the
                # correct structure to keep everything in order for computing
                # either "metagroup" images for the target or "subgroup" images.
                if thisTarget in processSubGroupList:
                    tmpDict = {
                        'images': list(Aimgs),
                        'background_levels': list(AbkgVals)
                    }

                    # Since we don't need to worry about adding MORE images to
                    # this polAng entry for each subgroup, we can just replace
                    # the currently empty dictionary with the tmpDict variable.
                    groupDict[thisSubGroup][polAngKey] = tmpDict

                else:
                    tmpDict = {
                        'images': list(Aimgs),
                        'background_levels': list(AbkgVals)
                    }

                    # The storage method must APPPEND the current images and
                    # backgrounds to already existing lists stored in groupDict.
                    polAngKey = int(thisPolAng)
                    for key, value in tmpDict.items():
                        # Copy the original list of values
                        tmpList = groupDict[polAngKey][key].copy()

                        # Extend the list to include the new polAngDict values
                        tmpList.extend(value)

                        # Replace the groupDict value with the extended list
                        groupDict[polAngKey][key] = tmpList

    elif thisDither == 'HEX':
        # Treat the hex-dither images differently.
        # If the observer decided that it was appropriate to use a hex-dither
        # rather than an ABBA or BAAB dither, then presume that background
        # levels can be directly estimated from the on-target images.
        # Now break the TARGET group up by its individual observations.
        indexBySubGroup = group.group_by(['Name'])

        # Loop through each observational grouping.
        subGroupKeys = indexBySubGroup.groups.keys
        for subGroup in indexBySubGroup.groups:
            # Update the user on the current execution status
            thisSubGroup = str(np.unique(subGroup['Name'])[0])
            print('\tProcessing images for subgroup {0}'.format(thisSubGroup))

            # If this group needs to be processed with independent keys for each
            # "subgroup", then test whether those files have already been done.
            if thisTarget in processSubGroupList:
                # Loop through the polAngs and check if these images have already been done
                fileCount     = 0
                for polAng in polAngs:
                    # Test if this group was previously processed.
                    testFile = '_'.join([thisTarget, thisSubGroup, str(polAng)])
                    testPath = os.path.join(polAngDir, testFile) + '.fits'

                    # Now that the filename has been constructed, test for file existence
                    if os.path.isfile(testPath):
                        print('\tFile ' + testFile + ' already exists...')
                        fileCount += 1

                if fileCount > 3:
                    # If all four polaroid angle images have been processed,
                    # then simply skip this subgroup for now.
                    continue
                else:
                    # otherwise proceed AND increment the subGroupCount by 1
                    subGroupCount += 1

                # Now that we have determined that this subGroup SHOULD be
                # processed, let's add an entry to the groupDict to store
                # the data for this subGroup.
                groupDict[thisSubGroup] = copy.deepcopy(polAngDict)

            # Start by breaking the subGroup up into its constituent polAngs
            indexByPolAng  = subGroup.group_by(['Pol Ang'])

            # Loop through each polAng subset of the subGroup
            polAngGroupKeys = indexByPolAng.groups.keys
            for polAngGroup in indexByPolAng.groups:
                # Update the user on processing status
                thisPolAng = str(np.unique(polAngGroup['Pol Ang'])[0])
                print('\t\tPolaroid Angle : {0}'.format(thisPolAng))

                # Initalize temporary lists to hold the images, times, etc....
                imgList = []
                bkgVals = []

                # Loop through each row of the polAng group, read in the images,
                # grab the observation times, estimate the background levels,
                # estimate the median, normalized, background images, and
                # perform the model fitting and interpolated on-target
                # background levels
                progressString = '\t\t\tImages : '
                for irow, row in enumerate(polAngGroup):
                    # Update the user on processing status
                    imgNumStr = ', '.join([str(i+1) for i in range(irow+1)])
                    print(progressString + imgNumStr, end='\r')

                    # Read in a temporary compy of this image
                    file1  = row['Filename']
                    tmpImg = AstroImage(file1)

                    # Apped the image to the imgList variable
                    imgList.append(tmpImg)

                    # Compute the image statistics and store them
                    goodPix= np.isfinite(tmpImg.arr)
                    if np.sum(goodPix) > 0:
                        # Find the good pixels
                        goodInds = np.where(goodPix)
                        # mean, median, std = sigma_clipped_stats(
                        #     tmpImg.arr[goodInds])

                        # Store the background median value
                        bkgVals.append(np.median(tmpImg.arr[goodInds]))
                    else:
                        print('There are no good pixels in this image...')
                        pdb.set_trace()

                # Print a single newline character to keep the progressString
                print('')

                # Now that the images have been read in and processed, proceed
                # to compute a subtract sky-background levels.
                imgList = [img - bkg for img, bkg in zip(imgList, bkgVals)]

                # Build a dictionary containing the keys necessary to map into
                # the groupDict dictionary. This dictionary needs to have the
                # correct structure to keep everything in order for computing
                # either "metagroup" images for the target or "subgroup" images.
                if thisTarget in processSubGroupList:
                    tmpDict = {
                        'images': list(imgList),
                        'background_levels': list(bkgVals)
                    }

                    # Since we don't need to worry about adding MORE images to
                    # this polAng entry for each subgroup, we can just replace
                    # the currently empty dictionary with the tmpDict variable.
                    polAngKey = int(thisPolAng)
                    groupDict[thisSubGroup][polAngKey] = tmpDict

                else:
                    tmpDict = {
                        'images': list(imgList),
                        'background_levels': list(bkgVals)
                    }

                    # The storage method must APPPEND the current images and
                    # backgrounds to already existing lists stored in groupDict.
                    polAngKey = int(thisPolAng)
                    for key, value in tmpDict.items():
                        # Copy the original list of values
                        tmpList = groupDict[polAngKey][key].copy()

                        # Extend the list to include the new polAngDict values
                        tmpList.extend(value)

                        # Replace the groupDict value with the extended list
                        groupDict[polAngKey][key] = tmpList

        # Now that we've  attempted to read in and process any subGroup images,
        # let's test whether or not subGroups were actually processed here...
        if subGroupCount == 0: continue

    # Now that the groupDict variable contains ALL the information necessary for
    # computing average polAng images, DO JUST THAT!

    # First test whether this needs to be processed as a "metagroup" or on a
    # "subgroup" basis.
    if thisTarget in processSubGroupList:
        # Process the subgroups.
        # Loop through each sub-group and compute its output images
        for subGroupKey, subGroupDict in groupDict.items():
            thisSubGroup = str(subGroupKey)
            print('\tGenerating average images for {0}'.format(thisSubGroup))

            # Loop through each polAng value in the subGroupDict
            for polAng in subGroupDict.keys():
                thisPolAng = str(polAng)
                print('\t\tPolaroid Angle : {0}'.format(thisPolAng))

                # Align the images for addition (don't worry about sub-pixel
                # alignment precision since we are not subtracting images but
                # adding them!)
                alignedImgList = utils.align_images(
                    subGroupDict[polAng]['images'],
                    padding=np.NaN,
                    mode='WCS',
                    subPixel=False)

                # Compute the average image using the AstroImage "utilities"
                combinedImg = utils.combine_images(
                    alignedImgList,
                    subGroupDict[polAng]['background_levels'],
                    output = 'MEAN',
                    effective_gain = effective_gain,
                    read_noise = read_noise)

                # Construct an image name and save to disk!
                outputFile = '_'.join([thisTarget, thisSubGroup, thisPolAng]) + '.fits'
                outputPath = os.path.join(polAngDir, outputFile)
                combinedImg.clear_astrometry()
                combinedImg.filename = outputPath
                combinedImg.write()

                # Resolve the astrometry of this combined image
                print('\tResolving astrometry...')
                combinedImg, success = utils.solve_astrometry(combinedImg)

                # Rewrite to disk...
                combinedImg.write()

    else:
        # Process as a metagroup.
        # Loop through groupDict keys, which are the polAng values
        for polAng in groupDict.keys():
            thisPolAng = str(polAng)
            print('\tGenerating average image for')
            print('\tPolaroid Angle : {0}'.format(thisPolAng))

            # Align the images for addition (don't worry about sub-pixel
            # alignment precision since we are not subtracting images but adding
            # them!)
            alignedImgList = utils.align_images(
                groupDict[polAng]['images'],
                padding=np.NaN,
                mode='WCS',
                subPixel=False)

            # Compute the average image using the AstroImage "utilities"
            combinedImg = utils.combine_images(
                alignedImgList,
                groupDict[polAng]['background_levels'],
                output = 'MEAN',
                effective_gain = effective_gain,
                read_noise = read_noise)

            # Construct an image name and save to disk!
            outputFile = '_'.join([thisTarget, thisWaveband, thisPolAng]) + '.fits'
            outputPath = os.path.join(polAngDir, outputFile)
            combinedImg.clear_astrometry()
            combinedImg.filename = outputPath
            combinedImg.write()

            # Resolve the astrometry of this combined image
            print('\tResolving astrometry...')
            combinedImg, success = utils.solve_astrometry(combinedImg)

            # Rewrite to disk...
            combinedImg.write()

print('\nDone computing average images!')
