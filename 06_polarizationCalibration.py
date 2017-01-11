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
from astropy.table import Table, Column, hstack
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from photutils import daofind, aperture_photometry, CircularAperture, CircularAnnulus
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
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data'

# This is the name of the file in which the calibration constants will be stored
calDataFile = os.path.join(pyPol_data, 'calData.csv')

# The user needs to specify the "Target" values associated with
# calibration data in the fileIndex.
targets = ['Orion_Cal_20150117',
           'Orion_Cal_20150119',
           'Taurus_Cal_20150118']

# Define some useful conversion factors
rad2deg = (180.0/np.pi)
deg2rad = (np.pi/180.0)

# This is the location of the previously generated masks (step 4)
maskDir = os.path.join(pyPol_data, 'Masks')

# Setup new directory for polarimetry data
polarimetryDir = os.path.join(pyPol_data, 'Polarimetry')
if (not os.path.isdir(polarimetryDir)):
    os.mkdir(polarimetryDir, 0o755)

polAngDir = os.path.join(polarimetryDir, 'polAngImgs')
if (not os.path.isdir(polAngDir)):
    os.mkdir(polAngDir, 0o755)

# Define the polarization standard files
RtableFile = os.path.join(pyPol_data, 'polStandardTable_R.csv')
VtableFile = os.path.join(pyPol_data, 'polStandardTable_V.csv')

if (not os.path.isfile(RtableFile)) or (not os.path.isfile(VtableFile)):
    # Read in the indexFile data and select the filenames
    print('\nReading file index from disk')
    indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
    fileIndex = Table.read(indexFile, format='ascii.csv')

    # Read in the polarization standards file
    print('Reading polarization standards from disk')
    polStandardFile = os.path.join('polStandards.csv')
    polStandards = Table.read(polStandardFile, format='ascii.csv')

    ra1      = polStandards['RA'].data
    dec1     = polStandards['Dec'].data
    polStanCoords = SkyCoord(ra = ra1, dec = dec1,
        unit = (u.hour, u.deg), frame = 'fk5')

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
    indexTarget = fileIndex.group_by(['Target'])

    # Setup a table for storing the final polStandard information
    # Begin by copying the original polStandards table to get the proper formating
    polStandardTable_V = polStandards.copy()
    polStandardTable_V.remove_rows(slice(0, len(polStandardTable_V), 1))
    polStandardTable_R = polStandardTable_V.copy()

    # Remove the R-band columns from the V table and vice versa
    polStandardTable_V.remove_columns(['P_R', 'sP_R', 'PA_R', 'sPA_R'])
    polStandardTable_R.remove_columns(['P_V', 'sP_V', 'PA_V', 'sPA_V'])

    # Loop through each target
    for target in indexTarget.groups:
        # Grab the current target information,
        # and display to the user
        thisTarget   = str(np.unique(target['Target'].data)[0])
        print('\nProcessing images for')
        print('\tTarget   : {0}'.format(thisTarget))

        # Further group by waveband and night
        indexTargetWave = target.group_by(['Waveband', 'Night'])

        for waveband in indexTargetWave.groups:
            # Grab the current waveband
            thisWaveband = str(np.unique(waveband['Waveband'].data)[0])
            thisNight    = str(np.unique(waveband['Night'].data)[0])
            print('\tWaveband : {0}'.format(thisWaveband))
            print('\tNight    : {0}'.format(thisNight))

            # Generate the poper keys for storing this data in table columns
            P_key   = '_'.join(['P', thisWaveband, thisNight])
            sP_key  = '_'.join(['sP', thisWaveband, thisNight])
            PA_key  = '_'.join(['PA', thisWaveband, thisNight])
            sPA_key = '_'.join(['sPA', thisWaveband, thisNight])

            # Add the columns required for the polStandardTables
            if thisWaveband == 'V':
                polStandardTable_V.add_columns([
                    Column(name = P_key, data = np.zeros(len(polStandardTable_V))),
                    Column(name = sP_key, data = np.zeros(len(polStandardTable_V))),
                    Column(name = PA_key, data = np.zeros(len(polStandardTable_V))),
                    Column(name = sPA_key, data = np.zeros(len(polStandardTable_V)))
                ])
            elif thisWaveband == 'R':
                polStandardTable_R.add_columns([
                    Column(name = P_key, data = np.zeros(len(polStandardTable_R))),
                    Column(name = sP_key, data = np.zeros(len(polStandardTable_R))),
                    Column(name = PA_key, data = np.zeros(len(polStandardTable_R))),
                    Column(name = sPA_key, data = np.zeros(len(polStandardTable_R)))
                ])

            # Initalize an boolean list
            # for testing which stars lie within the image footprint
            polStandardBool  = [True]*len(polStandards)

            # Initalize an empty dictionary for storing polAng AstroImage objects
            polAngImgs = {}
            indexTargetWavePolAng = indexTargetWave.group_by(['Pol Ang'])
            for polAng in indexTargetWavePolAng.groups:
                # Loop through each of the polAng images,
                # and check which polarization standards are common to them all
                thisPolAng = str(np.unique(polAng['Pol Ang'].data)[0])

                # Read in the current polAng image
                inFile = os.path.join(polAngDir,
                    '_'.join([thisTarget, thisWaveband, thisPolAng]) + '.fits')

                # Read the file and store it in the dictionary
                thisImg = AstroImage(inFile)
                polAngImgs[int(thisPolAng)] = thisImg

                # Determine which standards appear in this image
                polStandardBool = np.logical_and(polStandardBool,
                    thisImg.in_image(polStanCoords, edge=50))

            # A bit of cleanup to prevent confusion down the road
            del thisPolAng
            del thisImg

            # Select the standards for this targetWave group, and grob their
            # coordinates
            thisStandard = polStandards[np.where(polStandardBool)]
            thisCoords   = polStanCoords[np.where(polStandardBool)]

            # Now that the standard stars in this target have been identified,
            # quickly loop through the standards, and add one row to the
            # polStandardTable(s) for each standard.
            for standard in thisStandard:
                if thisWaveband == 'V':
                    # Check if a row needs to be added to the table
                    if standard['Name'] in polStandardTable_V['Name']:
                        continue
                    # Otherwise add the row
                    else:
                        thisRow = dict(zip(thisStandard.keys(), standard))
                        thisRow.pop('P_R', None)
                        thisRow.pop('sP_R', None)
                        thisRow.pop('PA_R', None)
                        thisRow.pop('sPA_R', None)
                        polStandardTable_V.add_row(thisRow)
                if thisWaveband == 'R':
                    # Check if a row needs to be added to the table
                    if standard['Name'] in polStandardTable_R['Name']:
                        continue
                    # Otherwise add the row
                    else:
                        thisRow = dict(zip(thisStandard.keys(), standard))
                        thisRow.pop('P_V', None)
                        thisRow.pop('sP_V', None)
                        thisRow.pop('PA_V', None)
                        thisRow.pop('sPA_V', None)
                        polStandardTable_R.add_row(thisRow)

            # Loop back through the polAngImgs, and compute the photometry of the
            # standards for each image
            photoDict = {}
            print('\tComputing photometry for')
            for polAng, img in polAngImgs.items():
                print('\t\tPolarioid Rotation : {0}'.format(polAng))
                # Grab the WCS for this image
                thisWCS = WCS(img.header)

                # Grab the pixel coordinates for the standard(s)
                xStar, yStar = thisCoords.to_pixel(thisWCS)

                # Find all sources in the image,
                # and match to the calibration positions
                goodInds = np.where(np.isfinite(img.arr))
                mean, median, std = sigma_clipped_stats(img.arr[goodInds],
                    sigma=3.0, iters=5)

                # Make a non-NaN version of the image array
                thisArr = np.nan_to_num(img.arr.copy())

                # Detect the sources above the threshold and match to the
                # calibration positions
                threshold = median + 3.0*std
                fwhm    = 6.0
                sources = daofind(thisArr, threshold, fwhm, ratio=1.0, theta=0.0,
                                  sigma_radius=1.5, sharplo=0.2, sharphi=1.0,
                                  roundlo=-1.0, roundhi=1.0, sky=0.0,
                                  exclude_border=True)

                # Convert source positions to (RA, Dec)
                ADstars  = thisWCS.all_pix2world(sources['xcentroid'], sources['ycentroid'], 0)

                # Build a catalog of all the star positions
                sourceCat = SkyCoord(ra = ADstars[0]*u.deg, dec = ADstars[1]*u.deg, frame = 'fk5')

                # Look for the standard stars in the source catalog
                idxc1, idxc2, d2d, d3d = sourceCat.search_around_sky(thisCoords, 10.0*u.arcsec)

                # Detect which matches from catalog2 to catalog1 are unique
                # TODO understand and possibly eliminate the need for
                # using "bins" at all
                # and possibly eliminate "return_inerse = True"
                bins, freq     = np.unique(idxc1, return_inverse=True)
                cat2Keep       = np.bincount(freq) == 1
                bins, freq     = np.unique(idxc2, return_inverse=True)
                matchSourceInd = bins[np.where(cat2Keep)]

                # Build a secondary catalog of SkyCoordinates.
                # By matching this catalog to the original "thisCoords" catalog,
                # we will be able to determine if the sources were re-ordered
                # during the matching process.
                sourceRA      = ADstars[0][matchSourceInd]
                sourceDec     = ADstars[1][matchSourceInd]
                matchSourceCoord = SkyCoord(ra = sourceRA, dec = sourceDec,
                                            unit = (u.deg, u.deg), frame = 'fk5')

                # Match original catalog to new catalog
                idxc1, idxc2, d2d, d3d = matchSourceCoord.search_around_sky(thisCoords, 5.0*u.arcsec)

                # Re-order the matching indices
                # to preserve star position across catalogs
                matchSourceInd = matchSourceInd[idxc2]
                sourceNames    = thisStandard['Name'][idxc1]


                # Cull the decteced sources catalog
                # to only include the matched standard stars
                xStar1 = (sources['xcentroid'].data)[matchSourceInd]
                yStar1 = (sources['ycentroid'].data)[matchSourceInd]

                # Test with a plot
                # plt.ion()
                # fig = plt.figure(figsize = (8,8))
                # ax  = fig.add_subplot(1,1,1)
                # axIm = ax.imshow(thisArr, cmap='cubehelix',vmin=1,vmax=500, origin='lower')
                # xlim, ylim = ax.get_xlim(), ax.get_ylim()
                # line1, = ax.plot(xStar1, yStar1,
                #                 linestyle='None',
                #                 marker='o', markersize=10.0, markeredgewidth=2.0,
                #                 markeredgecolor='red', fillstyle='none')
                # ax.set_xlim(*xlim), ax.set_ylim(*ylim)
                # plt.ioff()
                # pdb.set_trace()

                # Do aperture photometry on the selected sources
                # Setup the apertures
                sourcePos = [(xs, ys) for xs, ys in zip(xStar1, yStar1)]
                apertures = CircularAperture(sourcePos, r = 20.0)
                annulus_apertures = CircularAnnulus(sourcePos, r_in=22.0, r_out=24.0)

                # Perform the basic photometry
                rawflux_table = aperture_photometry(thisArr, apertures,
                    error=img.sigma)
                bkgflux_table = aperture_photometry(thisArr, annulus_apertures,
                    error=img.sigma)
                phot_table = hstack([rawflux_table, bkgflux_table],
                    table_names=['raw', 'bkg'])

                # Compute background contribution and its uncertainty
                bkg_mean = phot_table['aperture_sum_bkg'] / annulus_apertures.area()
                bkg_sig  = phot_table['aperture_sum_err_bkg'] / annulus_apertures.area()
                bkg_sum = bkg_mean * apertures.area()
                bkg_sig = bkg_sig * apertures.area()

                # Compute the variance in the background pixels for each star
                ny, nx  = thisArr.shape
                yy, xx  = np.mgrid[0:ny, 0:nx]
                bkg_var = []
                for xs, ys in sourcePos:
                    distFromStar = np.sqrt((xx - xs)**2 + (yy - ys)**2)
                    skyPixInds   = np.where(np.logical_and(
                        (distFromStar > 22.0), (distFromStar < 24.0)))
                    bkg_var.append(np.var(thisArr[skyPixInds]))

                # Convert the background variance into an array
                bkg_var = np.array(bkg_var)

                # Compute the final photometry and its uncertainty
                final_sum = phot_table['aperture_sum_raw'] - bkg_sum
                final_sig = np.sqrt(phot_table['aperture_sum_err_raw']**2
                    + bkg_sig**2
                    + bkg_var)
                phot_table['residual_aperture_sum'] = final_sum
                phot_table['residual_aperture_sum_err'] = final_sig

                # Finally, add a name to the phot-table,
                # and save photometry table in a dictionary.
                phot_table.add_column(
                    Column(name = 'Name', data = sourceNames), index=0)
                photoDict[polAng] = phot_table

            # Isolate the flux at each rotation angle
            I000 = dict(zip(photoDict[0]['Name'].data,
                            photoDict[0]['residual_aperture_sum'].data))
            I200 = dict(zip(photoDict[200]['Name'].data,
                            photoDict[200]['residual_aperture_sum'].data))
            I400 = dict(zip(photoDict[400]['Name'].data,
                            photoDict[400]['residual_aperture_sum'].data))
            I600 = dict(zip(photoDict[600]['Name'].data,
                            photoDict[600]['residual_aperture_sum'].data))

            # Isolate the uncertainty at each rotation angle
            S000 = dict(zip(photoDict[0]['Name'].data,
                            photoDict[0]['residual_aperture_sum_err'].data))
            S200 = dict(zip(photoDict[200]['Name'].data,
                            photoDict[200]['residual_aperture_sum_err'].data))
            S400 = dict(zip(photoDict[400]['Name'].data,
                            photoDict[400]['residual_aperture_sum_err'].data))
            S600 = dict(zip(photoDict[600]['Name'].data,
                            photoDict[600]['residual_aperture_sum_err'].data))

            # Collect all the uniquely named stars in the photo dictionaries
            allStars = list(I000.keys())
            allStars.extend(I200.keys())
            allStars.extend(I400.keys())
            allStars.extend(I600.keys())
            allStars = np.unique(allStars)

            # Loop through each of the stars
            for star in allStars:
                # Check that this star is present in *ALL* of the dictionaries
                skipStar = False
                if star not in I000.keys():
                    print('\tStar {0} was not uniquely detected in the I000 image'.format(star))
                    skipStar = True
                if star not in I200.keys():
                    print('\tStar {0} was not uniquely detected in the I200 image'.format(star))
                    skipStar = True
                if star not in I400.keys():
                    print('\tStar {0} was not uniquely detected in the I400 image'.format(star))
                    skipStar = True
                if star not in I600.keys():
                    print('\tStar {0} was not uniquely detected in the I600 image'.format(star))
                    skipStar = True

                # Skip this star if it was not detected in one of the images
                if skipStar: continue

                # If the star is found in ALL the images, then continue...
                # Add this star's polarization data to the table
                # First compute U and its uncertainty
                A = I200[star] - I600[star]
                B = I200[star] + I600[star]
                sigAandB = np.sqrt(S200[star]**2 + S600[star]**2)
                U = A/B
                sigU = np.abs(U*sigAandB*np.sqrt(A**(-2) + B**(-2)))

                # then compute Q and its uncertainty
                A = I000[star] - I400[star]
                B = I000[star] + I400[star]
                sigAandB = np.sqrt(S000[star]**2 + S400[star]**2)
                Q = A/B
                sigQ = np.abs(Q*sigAandB*np.sqrt(A**(-2) + B**(-2)))

                # Now compute P and its uncertainty...
                P    = np.sqrt(U**2 + Q**2)
                sigP = np.sqrt((U*sigU)**2 + (Q*sigQ)**2)/P

                # ...and de-bias the polarization measurements
                if P/sigP <= 1:
                    P = 0
                else:
                    P = np.sqrt(P**2 - sigP**2)

                # Compute PA and its uncertainty
                PA    = 0.5*np.arctan2(U, Q)*rad2deg

                # lazy way (assumes sigQ ~= sigU)
                # sigPA = 0.5*rad2deg*(sigP/P)

                # Real way (uses actual sigQ and sigU)
                sigPA = 0.5*rad2deg*np.sqrt((U*sigQ)**2 + (Q*sigU)**2)/P**2
                # TODO Double check that this matches the formula in PEGS_pol
                # I think that PEGS pol is actually MISSING a factor of P
                # in the denominator.

                # Scale up polarization values to percentages
                P    *= 100.0
                sigP *= 100.0

                # Check that the polarization is reasonable
                # (e.g. R-band, 20150119, HD38563A is problematic)
                if P > 10:
                    print('\tThe polarization of star {0} seems to high'.format(star))
                    print('\tskipping to next star')
                    continue

                # Update the polStandardTables accordingly
                if thisWaveband == 'V':
                    # Figure out which row of the polStandardTable to modify
                    modRow = np.where(polStandardTable_V['Name'] == star)

                    # Fill in the P and PA values
                    polStandardTable_V[P_key][modRow]   = P
                    polStandardTable_V[sP_key][modRow]  = sigP
                    polStandardTable_V[PA_key][modRow]  = PA
                    polStandardTable_V[sPA_key][modRow] = sigPA
                if thisWaveband == 'R':
                    # Figure out which row of the polStandardTable to modify
                    modRow = np.where(polStandardTable_R['Name'] == star)

                    # Fill in the P and PA values
                    polStandardTable_R[P_key][modRow]   = P
                    polStandardTable_R[sP_key][modRow]  = sigP
                    polStandardTable_R[PA_key][modRow]  = PA
                    polStandardTable_R[sPA_key][modRow] = sigPA

    # Now that ALL the data have been processed....
    # Write the results to disk for future reference
    print('Saving polarization table to disk')
    polStandardTable_R.write(RtableFile, format='ascii.csv')
    polStandardTable_V.write(VtableFile, format='ascii.csv')

else:
    # If the tables are already on disk, then just read them in and execute the
    # calibration script portion of the script
    polStandardTable_V = Table.read(VtableFile, format='ascii.csv')
    polStandardTable_R = Table.read(RtableFile, format='ascii.csv')

# Build a quick Table to store the calibration results
calTable = Table(names=('Waveband', 'PE', 's_PE', 'PAsign', 'dPA', 's_dPA'),
                 dtype=('S1', 'f8', 'f8', 'f8', 'f8', 'f8'))

###########################################################################
# Calibrate V-band
###########################################################################
###############
# Get PE value
###############
# Grab the column names of the polarization measurements
polStart = lambda s: s.startswith('P_V_')
polBool  = list(map(polStart, polStandardTable_V.keys()))
polInds  = np.where(polBool)
polKeys  = np.array(polStandardTable_V.keys())[polInds]

# loop through each polarization column
P0 = list()
P1 = list()
for key in polKeys:
    # Grab the polarization measurements from this column
    tmpP = polStandardTable_V[key].data

    # Grab the non-zero measurements
    # (zero is the current "bad data" filler)
    goodInds = np.where(tmpP != 0)

    # Extend the P0 and P1 lists to include good data
    P0.extend(polStandardTable_V['P_V'].data[goodInds])
    P1.extend(polStandardTable_V[key].data[goodInds])

# Grab the names of the uncertainty columns
sigStart = lambda s: s.startswith('sP_V_')
sigBool = list(map(sigStart, polStandardTable_V.keys()))
sigInds = np.where(sigBool)
sigKeys = np.array(polStandardTable_V.keys())[sigInds]
# Loop through each uncertainty column
sP0 = list()
sP1 = list()
for key in sigKeys:
    # Grab the polarization measurements from this column
    tmpSig = polStandardTable_V[key].data

    # Grab the non-zero measurements
    # (zero is the current "bad data" filler)
    goodInds = np.where(tmpSig != 0)

    # Extend the P0 and P1 lists to include good data
    sP0.extend(polStandardTable_V['sP_V'].data[goodInds])
    sP1.extend(polStandardTable_V[key].data[goodInds])

# Define the model to be used in the fitting
def PE(slope, x):
     return slope*x

# Set up ODR with the model and data.
PEmodel = Model(PE)
data = RealData(P0, P1, sx=sP0, sy=sP1)

odr = ODR(data, PEmodel, beta0=[1.])

# Run the regression.
PEout = odr.run()

# Use the in-built pprint method to give us results.
print('V-band PE fitting results')
PEout.pprint()


print('\n\nGenerating P plot')
plt.ion()
fig = plt.figure()
ax  = fig.add_subplot(1,1,1)
ax.errorbar(P0, P1, xerr=sP0, yerr=sP1,
    ecolor='b', linestyle='None', marker=None)
ax.plot([0,max(P0)], PE(PEout.beta[0], np.array([0,max(P0)])), 'g')
plt.xlabel('Cataloged P [%]')
plt.ylabel('Measured P [%]')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xlim = 0, xlim[1]
ylim = 0, ylim[1]
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.title('V-band Polarization Efficiency')

#Compute where the annotation should be placed
ySpan = np.max(ylim) - np.min(ylim)
xSpan = np.max(xlim) - np.min(xlim)
xtxt = 0.1*xSpan + np.min(xlim)
ytxt = 0.9*ySpan + np.min(ylim)
plt.text(xtxt, ytxt, 'PE = {0:4.3g} +/- {1:4.3g}'.format(
    PEout.beta[0], PEout.sd_beta[0]), fontdict=font)
pdb.set_trace()

###############
# Get PA offset
###############
PAstart = lambda s: s.startswith('PA_V_')
PAbool  = list(map(PAstart, polStandardTable_V.keys()))
PAinds  = np.where(PAbool)
PAkeys  = np.array(polStandardTable_V.keys())[PAinds]

# loop through each column (column <==> night)
PA0_V = list()
PA1_V = list()
for key in PAkeys:
    # Grab all the measured PA values from that night
    tmpPA    =  polStandardTable_V[key].data

    # Determine where there are truely measured values
    # (zero is the currently used "empty value")
    goodInds = np.where(tmpPA != 0)

    # Add the true PA values to the list of true PA values
    PA0_V.extend(polStandardTable_V['PA_V'].data[goodInds])

    # Add the measured PA values to the list of measured PA values
    PA1_V.extend(tmpPA[goodInds])

# Convert these to arrays
PA0_V = np.array(PA0_V)
PA1_V = np.array(PA1_V)

# Grab the names of the uncertainty columns
sigStart = lambda s: s.startswith('sPA_V_')
sigBool = list(map(sigStart, polStandardTable_V.keys()))
sigInds = np.where(sigBool)
sigKeys = np.array(polStandardTable_V.keys())[sigInds]
# Loop through each uncertainty column
sPA0_V = list()
sPA1_V = list()
for key in sigKeys:
    # Grab the polarization measurements from this column
    tmpSig = polStandardTable_V[key].data

    # Grab the non-zero measurements
    # (zero is the current "bad data" filler)
    goodInds = np.where(tmpSig != 0)

    # Extend the P0 and P1 lists to include good data
    sPA0_V.extend(polStandardTable_V['sP_V'].data[goodInds])
    sPA1_V.extend(polStandardTable_V[key].data[goodInds])

# Fit a model to the PA1_V vs. PA0_V data
# Define the model to be used in the fitting
def deltaPA(B, x):
     return B[0]*x + B[1]

# Set up ODR with the model and data.
deltaPAmodel = Model(deltaPA)
data = RealData(PA0_V, PA1_V, sx=sPA0_V, sy=sPA1_V)

# On first pass, just figure out what the sign is
odr    = ODR(data, deltaPAmodel, beta0=[0.0, 90.0])
dPAout = odr.run()
PAsign = np.round(dPAout.beta[0])

# Build the proper fitter class with the slope fixed
odr = ODR(data, deltaPAmodel, beta0=[PAsign, 90.0], ifixb=[0,1])

# Run the regression.
dPAout = odr.run()

# Use the in-built pprint method to give us results.
print('V-band delta PA fitting results')
dPAout.pprint()

# Store the final calibration data in the calTable variable
calTable.add_row(['V', PEout.beta[0], PEout.sd_beta[0],
               PAsign, dPAout.beta[1], dPAout.sd_beta[1]])

# Apply the correction terms
dPAval = dPAout.beta[1]
PAcor  = ((PAsign*(PA1_V - dPAval)) + 720.0) % 180.0

# TODO
# Check if PAcor values are closer corresponding PA0_V values
# by adding or subtracting 180
PAminus = np.abs((PAcor - 180) - PA0_V ) < np.abs(PAcor - PA0_V)
if np.sum(PAminus) > 0:
    PAcor[np.where(PAminus)] = PAcor[np.where(PAminus)] - 180

PAplus = np.abs((PAcor + 180) - PA0_V ) < np.abs(PAcor - PA0_V)
if np.sum(PAplus) > 0:
    PAcor[np.where(PAplus)] = PAcor[np.where(PAplus)] + 180

# Save corrected values for possible future use
PAcor_V = PAcor.copy()

# Do a final regression to plot-test if things are right
data = RealData(PA0_V, PAcor, sx=sPA0_V, sy=sPA1_V)
odr  = ODR(data, deltaPAmodel, beta0=[1.0, 0.0], ifixb=[0,1])
dPAcor = odr.run()

# Plot up the results
# PA measured vs. PA true
print('\n\nGenerating PA plot')
fig.delaxes(ax)
ax = fig.add_subplot(1,1,1)
#ax.errorbar(PA0_V, PA1_V, xerr=sPA0_V, yerr=sPA1_V,
#    ecolor='b', linestyle='None', marker=None)
#ax.plot([0,max(PA0_V)], deltaPA(dPAout.beta, np.array([0,max(PA0_V)])), 'g')
ax.errorbar(PA0_V, PAcor, xerr=sPA0_V, yerr=sPA1_V,
    ecolor='b', linestyle='None', marker=None)
ax.plot([0,max(PA0_V)], deltaPA(dPAcor.beta, np.array([0, max(PA0_V)])), 'g')
plt.xlabel('Cataloged PA [deg]')
plt.ylabel('Measured PA [deg]')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xlim = 0, xlim[1]
ax.set_xlim(xlim)
plt.title('V-band PA offset')

#Compute where the annotation should be placed
ySpan = np.max(ylim) - np.min(ylim)
xSpan = np.max(xlim) - np.min(xlim)
xtxt = 0.1*xSpan + np.min(xlim)
ytxt = 0.9*ySpan + np.min(ylim)
plt.text(xtxt, ytxt, 'PA offset = {0:4.3g} +/- {1:4.3g}'.format(
    dPAout.beta[1], dPAout.sd_beta[1]), fontdict=font)
pdb.set_trace()

###########################################################################
# Calibrate R-band
###########################################################################
###############
# Get PE value
###############
# Grab the column names of the polarization measurement
polStart = lambda s: s.startswith('P_R_')
polBool = list(map(polStart, polStandardTable_R.keys()))
polInds = np.where(polBool)
polKeys = np.array(polStandardTable_R.keys())[polInds]

# loop through each column
P0 = list()
P1 = list()
for key in polKeys:
    # Grab the polarization measurements from this column
    tmpP = polStandardTable_R[key].data

    # Grab the non-zero measurements
    # (zero is the current "bad data" filler)
    goodInds = np.where(tmpP != 0)

    # Extend the P0 and P1 lists to include good data
    P0.extend(polStandardTable_R['P_R'].data[goodInds])
    P1.extend(polStandardTable_R[key].data[goodInds])

# Grab the column names of the polarization uncertainty
sigStart = lambda s: s.startswith('sP_R_')
sigBool = list(map(sigStart, polStandardTable_R.keys()))
sigInds = np.where(sigBool)
sigKeys = np.array(polStandardTable_R.keys())[sigInds]

# loop through each column
sP0 = list()
sP1 = list()
for key in sigKeys:
    # Grab the polarization measurements from this column
    tmp_sP = polStandardTable_R[key].data

    # Grab the non-zero measurements
    # (zero is the current "bad data" filler)
    goodInds = np.where(tmp_sP != 0)

    # Extend the P0 and P1 lists to include good data
    sP0.extend(polStandardTable_R['sP_R'].data[goodInds])
    sP1.extend(polStandardTable_R[key].data[goodInds])

# Set up ODR with the model and data.
data = RealData(P0, P1, sx=sP0, sy=sP1)

# Build the proper fitter class
odr = ODR(data, PEmodel, beta0=[-1.])

# Run the regression.
PEout = odr.run()

# Use the in-built pprint method to give us results.
print('R-band delta PE fitting results')
PEout.pprint()

# Plot up the results
# PA measured vs. PA true
print('\n\nGenerating PE plot')
fig.delaxes(ax)
ax = fig.add_subplot(1,1,1)
ax.errorbar(P0, P1, xerr=sP0, yerr=sP1,
    ecolor='b', linestyle='None', marker=None)
ax.plot([0,max(P0)], PE(PEout.beta[0], np.array([0,max(P0)])), 'g')
plt.xlabel('Cataloged P [%]')
plt.ylabel('Measured P [%]')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xlim = 0, xlim[1]
ylim = 0, ylim[1]
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.title('R-band Polarization Efficiency')

#Compute where the annotation should be placed
ySpan = np.max(ylim) - np.min(ylim)
xSpan = np.max(xlim) - np.min(xlim)
xtxt = 0.1*xSpan + np.min(xlim)
ytxt = 0.9*ySpan + np.min(ylim)
plt.text(xtxt, ytxt, 'PE = {0:4.3g} +/- {1:4.3g}'.format(
    PEout.beta[0], PEout.sd_beta[0]))
pdb.set_trace()

###############
# Get PA offset
###############
PAstart = lambda s: s.startswith('PA_R_')
PAbool  = list(map(PAstart, polStandardTable_R.keys()))
PAinds  = np.where(PAbool)
PAkeys  = np.array(polStandardTable_R.keys())[PAinds]

# loop through each column (column <==> night)
PA0_R = list()
PA1_R = list()
for key in PAkeys:
    # Grab all the measured PA values from that night
    tmpPA    =  polStandardTable_R[key].data

    # Determine where there are truely measured values
    # (zero is the currently used "empty value")
    goodInds = np.where(tmpPA != 0)

    # Add the true PA values to the list of true PA values
    PA0_R.extend(polStandardTable_R['PA_R'].data[goodInds])

    # Add the measured PA values to the list of measured PA values
    PA1_R.extend(tmpPA[goodInds])

# Convert to arrays
PA0_R = np.array(PA0_R)
PA1_R = np.array(PA1_R)

# Grab the names of the uncertainty columns
sigStart = lambda s: s.startswith('sPA_R_')
sigBool = list(map(sigStart, polStandardTable_R.keys()))
sigInds = np.where(sigBool)
sigKeys = np.array(polStandardTable_R.keys())[sigInds]
# Loop through each uncertainty column
sPA0_R = list()
sPA1_R = list()
for key in sigKeys:
    # Grab the polarization measurements from this column
    tmpSig = polStandardTable_R[key].data

    # Grab the non-zero measurements
    # (zero is the current "bad data" filler)
    goodInds = np.where(tmpSig != 0)

    # Extend the P0 and P1 lists to include good data
    sPA0_R.extend(polStandardTable_R['sP_R'].data[goodInds])
    sPA1_R.extend(polStandardTable_R[key].data[goodInds])

# Convrt to arrays
sPA0_R = np.array(sPA0_R)
sPA1_R = np.array(sPA1_R)

# Fit a model to the PA1 vs. PA0_R data
# Set up ODR with the model and data.
data = RealData(PA0_R, PA1_R, sx=sPA0_R, sy=sPA1_R)

# On first pass, just figure out what the sign is
odr    = ODR(data, deltaPAmodel, beta0=[0.0, 90.0])
dPAout = odr.run()
PAsign = np.round(dPAout.beta[0])

# Build the proper fitter class with the slope fixed
odr = ODR(data, deltaPAmodel, beta0=[PAsign, 90.0], ifixb=[0,1])

# Run the regression.
dPAout = odr.run()

# Use the in-built pprint method to give us results.
print('R-band delta PA fitting results')
dPAout.pprint()

# Store the final calibration data in the calTable variable
calTable.add_row(['R', PEout.beta[0], PEout.sd_beta[0],
               PAsign, dPAout.beta[1], dPAout.sd_beta[1]])

# Apply the correction terms
dPAval = dPAout.beta[1]
PAcor  = ((PAsign*(PA1_R - dPAval)) + 720.0) % 180.0

# Check if the correct PAs need 180 added or subtracted.
PAminus = np.abs((PAcor - 180) - PA0_R ) < np.abs(PAcor - PA0_R)
if np.sum(PAminus) > 0:
    PAcor[np.where(PAminus)] = PAcor[np.where(PAminus)] - 180

PAplus = np.abs((PAcor + 180) - PA0_R ) < np.abs(PAcor - PA0_R)
if np.sum(PAplus) > 0:
    PAcor[np.where(PAplus)] = PAcor[np.where(PAplus)] + 180

# Save corrected values for possible future use
PAcor_R = PAcor.copy()

# Do a final regression to plot-test if things are right
data = RealData(PA0_R, PAcor, sx=sPA0_R, sy=sPA1_R)
odr  = ODR(data, deltaPAmodel, beta0=[1.0, 0.0], ifixb=[0,1])
dPAcor = odr.run()


# Plot up the results
# PA measured vs. PA true
print('\n\nGenerating PA plot')
fig.delaxes(ax)
ax = fig.add_subplot(1,1,1)
#ax.errorbar(PA0_R, PA1, xerr=sPA0_R, yerr=sPA1,
#    ecolor='b', linestyle='None', marker=None)
#ax.plot([0,max(PA0_R)], deltaPA(dPAout.beta, np.array([0,max(PA0_R)])), 'g')
ax.errorbar(PA0_R, PAcor, xerr=sPA0_R, yerr=sPA1_R,
    ecolor='b', linestyle='None', marker=None)
ax.plot([0,max(PA0_R)], deltaPA(dPAcor.beta, np.array([0, max(PA0_R)])), 'g')
plt.xlabel('Cataloged PA [deg]')
plt.ylabel('Measured PA [deg]')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xlim = 0, xlim[1]
ax.set_xlim(xlim)
plt.title('R-band PA offset')

#Compute where the annotation should be placed
ySpan = np.max(ylim) - np.min(ylim)
xSpan = np.max(xlim) - np.min(xlim)
xtxt = 0.1*xSpan + np.min(xlim)
ytxt = 0.9*ySpan + np.min(ylim)

plt.text(xtxt, ytxt, 'PA offset = {0:4.3g} +/- {1:4.3g}'.format(
    dPAout.beta[1], dPAout.sd_beta[1]))
pdb.set_trace()

#####################################################
# Check if a single deltaPA value is appropriate
#####################################################
# Test if R-band and V-band deltaPA are compatible...
V_row = np.where(calTable['Waveband'] == np.array(['V']*len(calTable), dtype='S'))
R_row = np.where(calTable['Waveband'] == np.array(['R']*len(calTable), dtype='S'))
deltaDeltaPA = np.abs(calTable[R_row]['dPA'].data - calTable[V_row]['dPA'].data)
sig_deltaDPA = np.sqrt(calTable[R_row]['s_dPA'].data**2 + calTable[V_row]['s_dPA'].data**2)

# Check if this these two values are significantly different from each-other
if deltaDeltaPA/sig_deltaDPA > 3.0:
    print('These two calibration constants are significantly different')
else:
    # If they are not significantly different, then re-compute the single
    # calibration constant.
    # BUild the complete list of catalog stars...
    PA0 = list(PA0_V.copy())
    PA0.extend(PA0_R)
    PA0 = np.array(PA0)

    # ...and the catalog star uncertainties
    sPA0 = list(sPA0_V.copy())
    sPA0.extend(sPA0_R)
    sPA0 = np.array(sPA0)

    # Build the complete list of measured PAs...
    PA1 = list(PA1_V.copy())
    PA1.extend(PA1_R)
    PA1 = np.array(PA1)

    # ...and the measured PA uncertainties
    sPA1 = list(sPA1_V.copy())
    sPA1.extend(sPA1_R)
    sPA1 = np.array(sPA1)

    # Do a final regression to plot-test if things are right
    data = RealData(PA0, PA1, sx=sPA0, sy=sPA1)
    # On first pass, just figure out what the sign is
    odr    = ODR(data, deltaPAmodel, beta0=[0.0, 90.0])
    dPAout = odr.run()
    PAsign = np.round(dPAout.beta[0])

    # Build the proper fitter class with the slope fixed
    odr = ODR(data, deltaPAmodel, beta0=[PAsign, 90.0], ifixb=[0,1])

    # Run the regression.
    dPAout = odr.run()

    # Use the in-built pprint method to give us results.
    print('Final delta PA fitting results')
    dPAout.pprint()

    # Update the calibration table
    calTable['dPA']   = dPAout.beta[1]
    calTable['s_dPA'] = dPAout.sd_beta[1]

    # Apply the correction terms
    dPAval = dPAout.beta[1]
    PAcor  = ((PAsign*(PA1 - dPAval)) + 720.0) % 180.0

    # Check if the correct PAs need 180 added or subtracted.
    PAminus = np.abs((PAcor - 180) - PA0 ) < np.abs(PAcor - PA0)
    if np.sum(PAminus) > 0:
        PAcor[np.where(PAminus)] = PAcor[np.where(PAminus)] - 180

    PAplus = np.abs((PAcor + 180) - PA0 ) < np.abs(PAcor - PA0)
    if np.sum(PAplus) > 0:
        PAcor[np.where(PAplus)] = PAcor[np.where(PAplus)] + 180

    # # Save corrected values for possible future use
    # PAcor_R = PAcor.copy()

    # Do a final regression to plot-test if things are right
    data = RealData(PA0, PAcor, sx=sPA0, sy=sPA1)
    odr  = ODR(data, deltaPAmodel, beta0=[1.0, 0.0], ifixb=[0,1])
    dPAcor = odr.run()

    # Plot up the results
    # PA measured vs. PA true
    print('\n\nGenerating PA plot')
    fig.delaxes(ax)
    ax = fig.add_subplot(1,1,1)
    #ax.errorbar(PA0_R, PA1, xerr=sPA0_R, yerr=sPA1,
    #    ecolor='b', linestyle='None', marker=None)
    #ax.plot([0,max(PA0_R)], deltaPA(dPAout.beta, np.array([0,max(PA0_R)])), 'g')
    ax.errorbar(PA0, PAcor, xerr=sPA0, yerr=sPA1,
        ecolor='b', linestyle='None', marker=None)
    ax.plot([0,max(PA0)], deltaPA(dPAcor.beta, np.array([0, max(PA0)])), 'g')
    plt.xlabel('Cataloged PA [deg]')
    plt.ylabel('Measured PA [deg]')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xlim = 0, xlim[1]
    ax.set_xlim(xlim)
    plt.title('Final Combined PA offset')

    #Compute where the annotation should be placed
    ySpan = np.max(ylim) - np.min(ylim)
    xSpan = np.max(xlim) - np.min(xlim)
    xtxt = 0.1*xSpan + np.min(xlim)
    ytxt = 0.9*ySpan + np.min(ylim)

    plt.text(xtxt, ytxt, 'PA offset = {0:4.3g} +/- {1:4.3g}'.format(
        dPAout.beta[1], dPAout.sd_beta[1]))
    pdb.set_trace()
    plt.close()
    plt.ioff()

print('Writing calibration data to disk')
calTable.write(calDataFile, format='ascii.csv')

print('Calibration tasks completed!')
