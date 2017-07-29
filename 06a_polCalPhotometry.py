# -*- coding: utf-8 -*-
"""
Identifies images with polarimetric calibration stars and measures the
photometry of those stars for each of the four unique polaroid filter rotation
angles. The photometric measurements are converted into polarimetric values,
which are stored in a .csv file for analysis in the next step.
"""
# Core imports
import os
import sys
import warnings

# Import scipy/numpy packages
import numpy as np

# Import astropy packages
from astropy.wcs import WCS
from astropy.table import Table, Column, hstack, join
import astropy.units as u
from astropy.coordinates import SkyCoord, FK4, FK5
from astropy.stats import sigma_clipped_stats
from photutils import (centroid_com, aperture_photometry, CircularAperture,
    CircularAnnulus)

# Import plotting utilities
from matplotlib import pyplot as plt

# Add the AstroImage class
import astroimage as ai

# This script will compute the photometry of polarization standard stars
# and output a file containing the polarization position angle
# additive correction and the polarization efficiency of the PRISM instrument.

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================
# Define how the font will appear in the plots
font = {
    'family': 'sans-serif',
    'color':  'black',
    'weight': 'normal',
    'size': 14
}

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data\\201612'

# This is the name of the file in which the calibration constants will be stored
calDataFile = os.path.join(pyPol_data, 'calData.csv')

# The user needs to specify the "Target" values associated with
# calibration data in the fileIndex.
calibrationTargets = ['Taurus_Cal', 'Orion_Cal', 'Cyg_OB2']
calibrationTargets = [t.upper() for t in calibrationTargets]

# Define the saturation limit to use for whether or not to trust photometry
satLimit = 18e3

# Setup new directory for polarimetry data
polarimetryDir = os.path.join(pyPol_data, 'Polarimetry')
if (not os.path.isdir(polarimetryDir)):
    raise ValueError('{} does not exist'.format(polAngDir))

polAngDir = os.path.join(polarimetryDir, 'polAngImgs')
if (not os.path.isdir(polAngDir)):
    raise ValueError('{} does not exist'.format(polAngDir))

# Read in the indexFile data and select the filenames
print('\nReading file index from disk')
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='ascii.csv')

# Read in the polarization standards file
print('Reading polarization data from disk')
polStandardFile = os.path.join('polStandards.csv')
polStandards = Table.read(polStandardFile, format='ascii.csv')

# Construct SkyCoord object containing the coordinates of the standards
ra1               = polStandards['RA_1950'].data
dec1              = polStandards['Dec_1950'].data
polStandardCoords = SkyCoord(ra = ra1, dec = dec1,
    unit = (u.hour, u.degree), frame = FK4(equinox='B1950'))

# Transform the coordinates to the FK5 - J2000 system.
polStandardCoords = polStandardCoords.transform_to(FK5(equinox='J2000'))

# Determine which parts of the fileIndex are usable
useFiles = (fileIndex['USE'] == 1)

# Further restrict the selection to only include the pre-selected calibration targets
targetFiles = np.array([False]*len(fileIndex), dtype=bool)
for target in calibrationTargets:
    targetFiles = np.logical_or(targetFiles,
                                fileIndex['TARGET'] == target)

# Cull the fileIndex to ONLY include the specified calibration targets
calFiles  = np.logical_and(useFiles, targetFiles)
if np.sum(calFiles) > 0:
    fileInds   = np.where(calFiles)
    fileIndex = fileIndex[fileInds]

# Group the fileIndex by waveband
fileIndexByWaveband = fileIndex.group_by(['FILTER'])

# Loop through each waveband and compute the photometry and polarization
# values for each of the calibration stars within the calibrationTargets for that waveband.
for group in fileIndexByWaveband.groups:
    # Grab the current group information
    thisFilter = str(np.unique(group['FILTER'].data)[0])

    # Define the polarization standard files
    thisFilename = 'polStandardTable_{0}.csv'.format(thisFilter)
    outTableFile = os.path.join(pyPol_data, thisFilename)

    # Update the user on processing status
    print('\nProcessing calibrationTargets for')
    print('Filter : {0}'.format(thisFilter))

    # Initalize the polCalTable variable to be empty. This was necessary
    # because it is impossible to combine one astropyTable with another
    # empty table. Instead, simply test if this variable has been populated.
    polCalTable = None

    # Loop through each subgroup and compute the photometry of the stars
    # in those images.
    indexBySubGroup = group.group_by(['GROUP_ID'])
    subGroupKeys    = indexBySubGroup.groups.keys
    for iSubGroup, subGroup in enumerate(indexBySubGroup.groups):
        # Update the user on processing status
        thisTarget   = str(np.unique(subGroup['TARGET'].data)[0])
        thisSubGroup = str(np.unique(subGroup['OBJECT'].data)[0])
        print('\tSubgroup : {0}'.format(thisSubGroup))

        # Start by breaking the subGroup up into its constituent polAngs
        indexByPolAng  = subGroup.group_by(['POLPOS'])

        # Initalize a dictionary for storing the respective images
        subGroupImgDict = {}

        # We are assuming that each subgroup is uniqouely named. In this
        # case, it represents an independent measure of the polarization
        # of a standard star. Thus, let's loop through the expected polAng
        # files, read them in, and store them in a dictionary for later use.

        # Initalize a boolean list to track which standards appear within
        # the images of this subgroup
        polStandardBool = np.ones(polStandardCoords.shape, dtype=bool)

        #
        # TODO: this can be replaced by a simple "get_sources_at_coords" call.
        #
        # Loop through each polAng subset of the subGroup and read in images
        polAngGroupKeys = indexByPolAng.groups.keys
        for polAngGroup in indexByPolAng.groups:
            # Generate the expected file name and attempt to read it in
            thisPolAng = np.unique(polAngGroup['POLPOS'])[0]
            inputFile  = '_'.join([thisTarget, thisSubGroup, str(thisPolAng)]) + '.fits'
            inputPath  = os.path.join(polAngDir, inputFile)

            # Read in the image
            polAngImg = ai.reduced.ReducedScience.read(inputPath)

            # Determine which standards appear in this image
            polStandardBool = np.logical_and(polStandardBool,
                polAngImg.in_image(polStandardCoords, edge=100))

            # Store this image in the dictionary
            subGroupImgDict[thisPolAng] = polAngImg

        # Now that all the polAng images are stored in a dictionary, let's
        # double check that AT LEAST ONE of the standards appear in all four
        # polAng images.
        if np.sum(polStandardBool) < 1:
            errStr = '''
            It would seem that none of the entries in the standard
            catalog appear in these images. Either...
            1) You need to add entries into your polarization standard catalog
            OR
            2) These images don't actually contain any polarization standard stars

            If it is option (2), then make sure to eliminate this target from the
            "calibrationTargets" variable near the top of this script.'''
            raise ValueError(errStr)


        # At least one polarization standard was found in all four
        # images, so we can proceed to do polarimetry on that source.

        # Start by grabbing the standard(s) which appear in these imgs
        goodStandardInds = np.where(polStandardBool)
        subGroupTable    = polStandards[goodStandardInds]

        # Quickly build a list of columns to keep in this table
        keepKeys = ['_'.join([prefix, thisFilter])
            for prefix in ['P', 'sP', 'PA', 'sPA']]
        keepKeys.extend(['Name', 'RA_1950', 'Dec_1950'])

        # Remove all the unnecessary columns from this table and Loop
        # through each row, performing polarimetry on the stars....
        subGroupTable.keep_columns(keepKeys)

        # Grab the original RAs and Decs
        skyCoord1 = SkyCoord(
            ra=subGroupTable['RA_1950'],
            dec=subGroupTable['Dec_1950'],
            unit=(u.hour, u.deg),
            frame=FK4(equinox='B1950')
        )
        # Convert to J2000 coordinates
        skyCoord1 = skyCoord1.transform_to(FK5(equinox='J2000'))

        # Initalize an empty dictionary for storing the photometry for each
        # polaroid rotation angle image.
        polAngPhotDict = {}

        # Initalize a boolean to track which stars pass photometry
        goodStars = True

        # Loop through each polaroid rotation angle image and perform the
        # photometry on the calibrator stars in that field.
        for polAng, polAngImg in subGroupImgDict.items():
            # Find the stears at the calibrator coordinates
            xStars, yStars = polAngImg.get_sources_at_coords(
                skyCoord1,
                satLimit=satLimit
            )

            # Check which stars were successfully located
            goodStars = np.logical_and(goodStars, np.isfinite(xStars))

            # Cull the list of star positinos to include only properly.
            goodInds  = np.where(goodStars)
            xStars    = xStars[goodInds]
            yStars    = yStars[goodInds]

            # Extend the badStars index list to include any bad stars found
            badStars = np.logical_not(goodStars)
            badInds  = np.where(badStars)

            # Do the photometry for this set of standards
            # Create a PhotometryAnalyzer object for this image
            photAnalyzer = ai.utilitywrappers.PhotometryAnalyzer(polAngImg)

            # Perform the actual stellar photometry (no curves of growth)
            # Use a static aperture for now!
            flux, uncertainty = photAnalyzer.aperture_photometry(
                xStars, yStars, 18, 24, 28
            )

            # Re-insert null photometry measurements for the culled stars
            flux = np.insert(flux, badInds[0], np.NaN)
            uncertainty = np.insert(uncertainty, badInds[0], np.NaN)

            # Store the measured photometry in the dictionary for this subGroup
            polAngPhotDict[polAng] =  {
                'flux': flux,
                's_flux': uncertainty
            }

            # # Do some debugging
            # print(flux)
            # plt.ion()
            # polAngImg.clear_astrometry()
            # polAngImg.show()
            # plt.autoscale(False)
            # plt.scatter(xStars, yStars, s=50, facecolor='none', edgecolor='red')
            #
            # import pdb; pdb.set_trace()

        # If the all of the photometry measurements can be trusted,
        # then continue to estimate the polarization of this source.
        # ********** STOKES Q **********
        A = (polAngPhotDict[0]['flux'] - polAngPhotDict[400]['flux'])
        B = (polAngPhotDict[0]['flux'] + polAngPhotDict[400]['flux'])
        Q = A/B

        # Compute the uncertainty in that Stokes U quantity
        s_AB = np.sqrt(polAngPhotDict[0]['s_flux']**2 +
            polAngPhotDict[400]['s_flux']**2)
        s_Q  = np.abs(s_AB/B)*np.sqrt(1.0 + Q**2)

        # ********** STOKES U **********
        A = (polAngPhotDict[200]['flux'] - polAngPhotDict[600]['flux'])
        B = (polAngPhotDict[200]['flux'] + polAngPhotDict[600]['flux'])
        U = A/B

        # Compute the uncertainty in that Stokes U quantity
        s_AB = np.sqrt(polAngPhotDict[200]['s_flux']**2 +
            polAngPhotDict[600]['s_flux']**2)
        s_U  = np.abs(s_AB/B)*np.sqrt(1.0 + U**2)

        # ********** POLARIZATION PERCENTAGE **********
        P   = np.sqrt(U**2 + Q**2)
        s_P = np.sqrt((U*s_U)**2 + (Q*s_Q)**2)/P

        # ...and de-bias the polarization measurements
        # TODO: ask Dan if I should be debiasing the standard star
        # calibration measurements.
        nullStarInds    = np.where(P/s_P <= 1)
        P[nullStarInds] = s_P[nullStarInds]
        P               = np.sqrt(P**2 - s_P**2)

        # ********** POLARIZATION POSITION ANGLE **********
        PA = np.rad2deg(0.5*np.arctan2(U, Q))

        # lazy way (assumes sigQ ~= sigU)
        # sigPA = 0.5*rad2deg*(sigP/P)

        # Real way (uses actual sigQ and sigU)
        # Canonical treatment is to use 0.5*(sigQ + sigU) as estimate of
        # sigP....
        # TODO: I should really just update this to include the formula I
        # use in the M82 paper.
        s_PA = 0.5*np.rad2deg(np.sqrt((U*s_Q)**2 + (Q*s_U)**2)/P**2)
        # TODO Double check that this matches the formula in PEGS_pol
        # I think that PEGS pol is actually MISSING a factor of P
        # in the denominator.

        # Scale up polarization values to percentages
        P   *= 100.0
        s_P *= 100.0

        # Check that the polarization is reasonable
        # (e.g. R-band, 20150119, HD38563A is problematic)
        badPols = P > 10
        if np.sum(badPols) > 0:
            warnings.warn("Culling anomalously high polarization observation")

            # Find the indices of the good polarization measurements
            goodPols = np.logical_not(badPols)
            goodInds = np.where(goodPols)

            # Cull the important data to include include the good measurements
            subGroupTable = subGroupTable[goodInds]
            P             = P[goodInds]
            s_P           = s_P[goodInds]
            PA            = PA[goodInds]
            s_PA          = s_PA[goodInds]

        # Construct a temporary table to hold the results of this subgroup
        columnSuffix = thisFilter + str(iSubGroup + 1)
        columnNames = [
            'Name',
            'P_' + columnSuffix,
            'sP_' + columnSuffix,
            'PA_' + columnSuffix,
            'sPA_' + columnSuffix
        ]

        # Create the table ojbect
        subGroupPolTable = Table(
            [subGroupTable['Name'], P, s_P, PA, s_PA],
            names=columnNames
        )

        # Join the temporary polarization table to the subGroupTable
        subGroupPolTable = join(
            subGroupTable,
            subGroupPolTable,
            join_type='left'
        )

        if polCalTable is None:
            polCalTable = subGroupPolTable.copy()
        else:
            # Now join this table to the master polCalTable instance using an
            # 'outer' join_type
            polCalTable = join(
                polCalTable,
                subGroupPolTable,
                join_type='outer'
            )

    # Now that all of the calibration data has been generated, save to disk
    polCalTable.write(outTableFile, overwrite=True)











            # Debugging plots code
            # plt.ion()
            # for phot, uncert in zip(photometry.T, uncertainty.T):
            #     plt.errorbar(aprs, phot, yerr=uncert, fmt='--o')
            #
            #     import pdb; pdb.set_trace()
            # continue

            # for iStandard, standard in enumerate(subGroupTable):
            #     # Grab the name of this standard
            #     thisStandard = standard['Name']
            #     print('\t\tStandard : {0}'.format(thisStandard))
            #
            #     # Loop through each polAng image, test for saturation,
            #     # measure star width, and perform aperture photometry.
            #     polAngPhotDict      = {}
            #     for polAng, polAngImg in subGroupImgDict.items():
            #         # Update the user on processing status
            #         print('\t\t\tPolaroid Angle : {0}'.format(str(polAng)))
            #
            #         # Find the expectedstar coordinates in this image using
            #         # the WCS in the header
            #         skyCoord1 = SkyCoord(ra=standard['RA_1950'], dec=standard['Dec_1950'],
            #             unit=(u.hour, u.deg), frame=FK4(equinox='B1950'))
            #         skyCoord1 = skyCoord1.transform_to(FK5(equinox='J2000'))
            #         x1, y1    = polAngImg.wcs.all_world2pix(
            #             skyCoord1.ra,
            #             skyCoord1.dec,
            #             0
            #         )
            #
            #         # Cut out a small subarray around the predicted position
            #         lf, rt = np.int(np.round(x1 - 20)), np.int(np.round(x1 + 20))
            #         bt, tp = np.int(np.round(y1 - 20)), np.int(np.round(y1 + 20))
            #         tmpArr = polAngImg.data[bt:tp, lf:rt]
            #
            #         # Test if this star appears to be saturated
            #         if tmpArr.max() > satLimit:
            #             # If it is saturated, then make the "saturatedStar"
            #             # variable "True" so that we will know NOT to use
            #             # this standard star later and break out of the loop.
            #             print('\t\t\tStar is saturated!')
            #             saturatedStar = True
            #             break
            #
            #         # Use a centroid function to get a more precise position
            #         x1, y1 = (centroid_com(tmpArr) + np.array([lf, bt]))
            #
            #         from photutils import data_properties, properties_table
            #         # Measure star width properties
            #         columns = ['id', 'xcentroid', 'ycentroid', 'semimajor_axis_sigma', 'semiminor_axis_sigma', 'orientation']
            #         props   = data_properties(tmpArr)
            #         tbl     = properties_table(props, columns=columns)
            #
            #         # Compute the axis ratio and test if it's any good
            #         semimajor = tbl['semimajor_axis_sigma'].data[0]
            #         semiminor = tbl['semiminor_axis_sigma'].data[0]
            #         axisRatio = semimajor/semiminor
            #         if axisRatio > 1.3:
            #             print('\t\t\tStar is too oblate!')
            #             print('\t\t\ta/b = {0}'.format(axisRatio))
            #             oblateStar = True
            #             break
            #
            #         # If it is not too oblate, then compute an approximate
            #         # width using a geometric mean
            #         starWidth = np.sqrt(semimajor*semiminor)
            #
            #         # Build a circular aperture  and a sky annulus to
            #         # measure the star photometry
            #         # TODO: this is a MAJOR problem! I really should do a curve
            #         # of growth analysis and then compute the optimal SNR
            #         # aperture for each star individually.
            #
            #         import pdb; pdb.set_trace()
            #
            #         # Measure the photometry (flux not magnitudes) using the
            #         # polAngImg.sigma array as the source of uncertainties
            #         phot_table = aperture_photometry(polAngImg.arr,
            #             [starAperture, skyAperture],
            #             error=polAngImg.sigma)
            #
            #         # Compute a mean background count rate within annulus
            #         skyArea      = skyAperture.area()
            #         bkg_mean     = phot_table['aperture_sum_1'].data[0] / skyArea
            #         sig_bkg_mean = phot_table['aperture_sum_err_1'].data[0] / skyArea
            #
            #         # Compute the background contribution to the stellar flux
            #         starArea    = starAperture.area()
            #         bkg_sum     = bkg_mean * starArea
            #         sig_bkg_sum = sig_bkg_mean * starArea
            #
            #         # Compute a final stellar flux
            #         final_flux     = phot_table['aperture_sum_0'].data[0] - bkg_sum
            #         sig_final_flux = np.sqrt(phot_table['aperture_sum_err_0'].data[0]**2 +
            #             sig_bkg_sum)
            #
            #         # Store the star photometry (and uncertainty) in the
            #         # polAngPhotDict under its polAng
            #         polAngPhotDict[polAng] = {
            #             'flux': final_flux,
            #             's_flux': sig_final_flux}
            #
            #     else:
            #         # If the whole loop executed without any problems, then
            #         # it is safe to assume that photometry can be trusted.
            #         # Indicate this with the "starSaturated" boolean flag.
            #         saturatedStar = False
            #         oblateStar    = False
            #
            #     # Now that the photometry for this star has been
            # # successfully measured, let's double check that the star
            # # was not saturated or oblate.
            # if saturatedStar:
            #     # print('\t\tAt least one photometry measurement was saturated!')
            #     # print('\t\tDo not compute the observed polarization.')
            #     continue
            # if oblateStar:
            #     # print('\t\tAt least one star was too oblate.')
            #     # print('\t\tDo not compute the observed polarization.')
            #     continue


        # # Join this subGroup calibration data to the overall table thus far
        # if polCalTable is None:
        #     polCalTable = subGroupTable.copy()
        # else:
        #     polCalTable = join(polCalTable, subGroupTable, join_type='outer')


print('Photometry of polarization calibration standards completed!')
