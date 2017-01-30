# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:59:25 2015

@author: jordan
"""

# Should I find a way to use Python "StatsModels" to do linear fitting with
# uncertainties in X and Y?

import os
import sys
import numpy as np
from astropy.wcs import WCS
from astropy.table import Table, Column, hstack, join
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from photutils import centroid_com, aperture_photometry, CircularAperture, CircularAnnulus
from scipy.odr import *
from matplotlib import pyplot as plt
import pdb

# Add the AstroImage class
from astroimage.astroimage import AstroImage

# This script will compute the photometry of polarization standard stars
# and output a file containing the polarization position angle
# additive correction and the polarization efficiency of the PRISM instrument.

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================
# Define how the font will appear in the plots
font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 14
        }

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data\\201501'

# This is the name of the file in which the calibration constants will be stored
calDataFile = os.path.join(pyPol_data, 'calData.csv')

# The user needs to specify the "Target" values associated with
# calibration data in the fileIndex.
targets = ['Taurus_Cal', 'Orion_Cal']

# Define the saturation limit to use for whether or not to trust photometry
satLimit = 1.6e4

# Define some useful conversion factors
rad2deg = (180.0/np.pi)
deg2rad = (np.pi/180.0)

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
fileIndex = Table.read(indexFile, format='ascii.csv')

# Read in the polarization standards file
print('Reading polarization data from disk')
polStandardFile = os.path.join('polStandards.csv')
polStandards = Table.read(polStandardFile, format='ascii.csv')

# Construct SkyCoord object containing the coordinates of the standards
ra1           = polStandards['RA'].data
dec1          = polStandards['Dec'].data
polStanCoords = SkyCoord(ra = ra1, dec = dec1,
    unit = (u.hour, u.deg), frame = 'fk5')

# Determine which parts of the fileIndex are usable
useFiles = fileIndex['Use'] == 1

# Further restrict the selection to only include the pre-selected targets
targetFiles = np.array([False]*len(fileIndex), dtype=bool)
for target in targets:
    targetFiles = np.logical_or(targetFiles,
                                fileIndex['Target'] == target)

# Cull the fileIndex to ONLY include the specified calibration targets
calFiles  = np.logical_and(useFiles, targetFiles)
if np.sum(calFiles) > 0:
    fileInds   = np.where(calFiles)
    fileIndex = fileIndex[fileInds]

# Group the fileIndex by waveband
fileIndexByWaveband = fileIndex.group_by(['Waveband'])

# Loop through each waveband and compute the photometry and polarization
# values for each of the calibration stars within the targets for that waveband.
for group in fileIndexByWaveband.groups:
    # Grab the current group information
    thisWaveband = str(np.unique(group['Waveband'].data)[0])

    # Define the polarization standard files
    thisFilename = 'polStandardTable_{0}.csv'.format(thisWaveband)
    outTableFile = os.path.join(pyPol_data, thisFilename)

    # Update the user on processing status
    print('\nProcessing targets for')
    print('Waveband : {0}'.format(thisWaveband))

    # Initalize the polCalTable variable to be empty. This was necessary
    # because it is impossible to combine one astropyTable with another
    # empty table. Instead, simply test if this variable has been populated.
    polCalTable = None

    # Loop through each subgroup and compute the photometry of the stars
    # in those images.
    indexBySubGroup = group.group_by(['Name'])
    subGroupKeys    = indexBySubGroup.groups.keys
    for iSubGroup, subGroup in enumerate(indexBySubGroup.groups):
        # Update the user on processing status
        thisTarget   = str(np.unique(subGroup['Target'].data)[0])
        thisSubGroup = str(np.unique(subGroup['Name'].data)[0])
        print('\tSubgroup : {0}'.format(thisSubGroup))

        # Start by breaking the subGroup up into its constituent polAngs
        indexByPolAng  = subGroup.group_by(['Pol Ang'])

        # Initalize a dictionary for storing the respective images
        subGroupImgDict = {}

        # We are assuming that each subgroup is uniqouely named. In this
        # case, it represents an independent measure of the polarization
        # of a standard star. Thus, let's loop through the expected polAng
        # files, read them in, and store them in a dictionary for later use.

        # Initalize a boolean list to track which standards appear within
        # the images of this subgroup
        polStandardBool = np.ones((len(polStandards),), dtype=bool)

        # Loop through each polAng subset of the subGroup and read in images
        polAngGroupKeys = indexByPolAng.groups.keys
        for polAngGroup in indexByPolAng.groups:
            # Generate the expected file name and attempt to read it in
            thisPolAng = str(np.unique(polAngGroup['Pol Ang'])[0])
            inputFile  = '_'.join([thisTarget, thisSubGroup, thisPolAng]) + '.fits'
            inputPath  = os.path.join(polAngDir, inputFile)

            if os.path.isfile(inputPath):
                polAngImg = AstroImage(inputPath)

                # Determine which standards appear in this image
                polStandardBool = np.logical_and(polStandardBool,
                    polAngImg.in_image(polStanCoords, edge=50))

            else:
                print('Could not find expected polAng image')
                pdb.set_trace()

            # Store this image in the dictionary
            subGroupImgDict[int(thisPolAng)] = polAngImg


        # Now that all the polAng images are stored in a dictionary, let's
        # double check that AT LEAST ONE of the standards appear in all four
        # polAng images.
        if np.sum(polStandardBool) < 1:
            errStr = '''It would seem that none of the entries in the standard
                     catalog appear in these images. Either...
                     1) You need to add entries into your polarization standard catalog
                     OR
                     2) These images don't actually contain any polarization standard star

                     If it is option (2), then make sure to eliminate this target from the
                     "targets" variable near the top of this script.'''
            print(errStr)

            pdb.set_trace()

        else:
            # At least one polarization standard was found in all four
            # images, so we can proceed to do polarimetry on that source.

            # Start by grabbing the standard(s) which appear in these imgs
            subGroupTable = polStandards[np.where(polStandardBool)]

            # Quickly build a list of columns to keep in this table
            keepKeys = ['_'.join([prefix, thisWaveband])
                for prefix in ['P', 'sP', 'PA', 'sPA']]
            keepKeys.extend(['Name', 'RA', 'Dec'])

            # Remove all the unnecessary columns from this table and Loop
            # through each row, performing polarimetry on the stars....
            subGroupTable.keep_columns(keepKeys)

            for istandard, standard in enumerate(subGroupTable):
                # Grab the name of this standard
                thisStandard = standard['Name']
                print('\t\tStandard : {0}'.format(thisStandard))

                # Loop through each polAng image, test for saturation,
                # measure star width, and perform aperture photometry.
                photDict      = {}
                for polAng, polAngImg in subGroupImgDict.items():
                    # Update the user on processing status
                    print('\t\t\tPolaroid Angle : {0}'.format(str(polAng)))

                    # Find the expectedstar coordinates in this image using
                    # the WCS in the header
                    thisWCS   = WCS(polAngImg.header)
                    skyCoord1 = SkyCoord(ra=standard['RA'], dec=standard['Dec'],
                        unit=(u.hour, u.deg), frame='icrs')
                    x1, y1    = thisWCS.all_world2pix(skyCoord1.ra, skyCoord1.dec, 0)

                    # Cut out a small subarray around the predicted position
                    lf, rt = np.int(np.round(x1 - 20)), np.int(np.round(x1 + 20))
                    bt, tp = np.int(np.round(y1 - 20)), np.int(np.round(y1 + 20))
                    tmpArr = polAngImg.arr[bt:tp, lf:rt]

                    # Test if this star appears to be saturated
                    if tmpArr.max() > satLimit:
                        # If it is saturated, then make the "saturatedStar"
                        # variable "True" so that we will know NOT to use
                        # this standard star later and break out of the loop.
                        print('\t\t\tStar is saturated!')
                        saturatedStar = True
                        break

                    # Use a centroid function to get a more precise position
                    x1, y1 = (centroid_com(tmpArr) + np.array([lf, bt]))

                    from photutils import data_properties, properties_table
                    # Measure star width properties
                    columns = ['id', 'xcentroid', 'ycentroid', 'semimajor_axis_sigma', 'semiminor_axis_sigma', 'orientation']
                    props   = data_properties(tmpArr)
                    tbl     = properties_table(props, columns=columns)

                    # Compute the axis ratio and test if it's any good
                    semimajor = tbl['semimajor_axis_sigma'].data[0]
                    semiminor = tbl['semiminor_axis_sigma'].data[0]
                    axisRatio = semimajor/semiminor
                    if axisRatio > 1.3:
                        print('\t\t\tStar is too oblate!')
                        print('\t\t\ta/b = {0}'.format(axisRatio))
                        oblateStar = True
                        break

                    # If it is not too oblate, then compute an approximate
                    # width using a geometric mean
                    starWidth = np.sqrt(semimajor*semiminor)

                    # Build a circular aperture  and a sky annulus to
                    # measure the star photometry
                    starAperture = CircularAperture((x1, y1), r=2*starWidth)
                    skyAperture  = CircularAnnulus((x1, y1),
                        r_in=3.0*starWidth, r_out=3.0*starWidth + 4)

                    # Measure the photometry (flux not magnitudes) using the
                    # polAngImg.sigma array as the source of uncertainties
                    phot_table = aperture_photometry(polAngImg.arr,
                        [starAperture, skyAperture],
                        error=polAngImg.sigma)

                    # Compute a mean background count rate within annulus
                    skyArea      = skyAperture.area()
                    bkg_mean     = phot_table['aperture_sum_1'].data[0] / skyArea
                    sig_bkg_mean = phot_table['aperture_sum_err_1'].data[0] / skyArea

                    # Compute the background contribution to the stellar flux
                    starArea    = starAperture.area()
                    bkg_sum     = bkg_mean * starArea
                    sig_bkg_sum = sig_bkg_mean * starArea

                    # Compute a final stellar flux
                    final_flux     = phot_table['aperture_sum_0'].data[0] - bkg_sum
                    sig_final_flux = np.sqrt(phot_table['aperture_sum_err_0'].data[0]**2 +
                        sig_bkg_sum)

                    # Store the star photometry (and uncertainty) in the
                    # photDict under its polAng
                    photDict[polAng] = {
                        'flux': final_flux,
                        's_flux': sig_final_flux}

                else:
                    # If the whole loop executed without any problems, then
                    # it is safe to assume that photometry can be trusted.
                    # Indicate this with the "starSaturated" boolean flag.
                    saturatedStar = False
                    oblateStar    = False

                # Now that the photometry for this star has been
                # successfully measured, let's double check that the star
                # was not saturated or oblate.
                if saturatedStar:
                    # print('\t\tAt least one photometry measurement was saturated!')
                    # print('\t\tDo not compute the observed polarization.')
                    continue
                if oblateStar:
                    # print('\t\tAt least one star was too oblate.')
                    # print('\t\tDo not compute the observed polarization.')
                    continue

                # If the all of the photometry measurements can be trusted,
                # then continue to estimate the polarization of this source.
                # ********** STOKES Q **********
                A = (photDict[0]['flux'] - photDict[400]['flux'])
                B = (photDict[0]['flux'] + photDict[400]['flux'])
                Q = A/B

                # Compute the uncertainty in that Stokes U quantity
                s_AB = np.sqrt(photDict[0]['s_flux']**2 +
                    photDict[400]['s_flux']**2)
                s_Q  = np.abs(s_AB/B)*np.sqrt(1.0 + Q**2)

                # ********** STOKES U **********
                A = (photDict[200]['flux'] - photDict[600]['flux'])
                B = (photDict[200]['flux'] + photDict[600]['flux'])
                U = A/B

                # Compute the uncertainty in that Stokes U quantity
                s_AB = np.sqrt(photDict[200]['s_flux']**2 +
                    photDict[600]['s_flux']**2)
                s_U  = np.abs(s_AB/B)*np.sqrt(1.0 + U**2)

                # ********** POLARIZATION PERCENTAGE **********
                P   = np.sqrt(U**2 + Q**2)
                s_P = np.sqrt((U*s_U)**2 + (Q*s_Q)**2)/P

                # ...and de-bias the polarization measurements
                if P/s_P <= 1:
                    P = 0
                else:
                    P = np.sqrt(P**2 - s_P**2)


                # ********** POLARIZATION POSITION ANGLE **********
                PA = np.rad2deg(0.5*np.arctan2(U, Q))

                # lazy way (assumes sigQ ~= sigU)
                # sigPA = 0.5*rad2deg*(sigP/P)

                # Real way (uses actual sigQ and sigU)
                s_PA = 0.5*rad2deg*np.sqrt((U*s_Q)**2 + (Q*s_U)**2)/P**2
                # TODO Double check that this matches the formula in PEGS_pol
                # I think that PEGS pol is actually MISSING a factor of P
                # in the denominator.

                # Scale up polarization values to percentages
                P   *= 100.0
                s_P *= 100.0

                # Check that the polarization is reasonable
                # (e.g. R-band, 20150119, HD38563A is problematic)
                if P > 10:
                    print('\tThe polarization of star {0} seems to high'.format(star))
                    print('\tskipping to next star')
                    continue

                # Now that the polarization has been measured for this
                # standard star, proceed to find the matching entry in the
                # polarization standard catalog and add that data to the
                # temporary table for this subGroup.

                # # First check that at least one star name matches.
                # matchedName = polStandards['Name'].data == thisStandard
                # if np.sum(matchedName) > 0:
                #     referenceRow = np.where(matchedName)
                # else:
                #     print('Could not match star name to anything in the table...')
                #     print('How on earth did you get here?!')
                #     pdb.set_trace()

                # Pull the proper row out of the polarization standard table
                calibrationRow = subGroupTable[np.array([istandard])]

                # Loop through the computed polarization values and add each
                # of them to the calibrationRow table, giving them an
                # appropriate column name based on the subGroup number.
                keys = ['P', 'sP', 'PA', 'sPA']
                vals = [P,   s_P,   PA,   s_PA]
                for polKey, polVal in zip(keys, vals):
                    # Construct the appropriate key for this value
                    colName = polKey + '_' + thisWaveband + str(iSubGroup+1)

                    # Add a column to the subGroupTable
                    calibrationRow[colName] = np.array([polVal])

                # Now that this row contains all the information needed to
                # form a single calibration datapoint, add it to the
                # subGroupTable variable.
                if polCalTable is None:
                    polCalTable = calibrationRow.copy()
                else:
                    polCalTable = join(polCalTable, calibrationRow,
                        join_type='outer')

        # # Join this subGroup calibration data to the overall table thus far
        # if polCalTable is None:
        #     polCalTable = subGroupTable.copy()
        # else:
        #     polCalTable = join(polCalTable, subGroupTable, join_type='outer')

    # Now that all of the calibration data has been generated, save to disk
    polCalTable.write(outTableFile)

print('Photometry of polarization calibration standards completed!')
