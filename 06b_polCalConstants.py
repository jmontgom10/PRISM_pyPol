# -*- coding: utf-8 -*-
"""
Computes the polarimetric efficiency (PE), position angle direction (+1 or -1),
and position angle offset (DeltaPA) for the instrument.
"""
# TODO: Should I find a way to use Python "Statsodr.Models" to do linear fitting
# with uncertainties in X and Y?

# Core imports
import os
import sys
import copy

# Scipy/numpy imports
import numpy as np
from scipy import odr

# Import statsmodels for robust linear regression
import statsmodels.api as smapi

# Astropy imports
from astropy.table import Table, Column, hstack, join
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from photutils import (centroid_com, aperture_photometry, CircularAperture,
    CircularAnnulus)

# Import plotting utilities
from matplotlib import pyplot as plt

# Import the astroimage package
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
font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 14
        }

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data\\201612'

# This is the name of the file in which the calibration constants will be stored
polCalConstantsFile = os.path.join(pyPol_data, 'polCalConstants.csv')

# Read in the indexFile data and select the filenames
print('\nReading file index from disk')
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='ascii.csv')

# Group the fileIndex by waveband
fileIndexByWaveband = fileIndex.group_by(['FILTER'])

# Retrieve the waveband values within specified the calibration data
wavebands = np.unique(fileIndexByWaveband['FILTER'])

# Initalize a table to store all the measured polarizatino calibration constants
calTable = Table(names=('FILTER', 'PE', 's_PE', 'PAsign', 'D_PA', 's_D_PA'),
                 dtype=('S1', 'f8', 'f8', 'i8', 'f8', 'f8'))

# Also initalize a dictionary to store ALL of the polarization data
allPolCalDict = {}

# Loop through each waveband and compute the calibration constants from the data
# available for that waveband.
for thisFilter in wavebands:
    # Update the user on processing status
    print('\nProcessing calibration data for')
    print('Filter : {0}'.format(thisFilter))

    # Define the polarization standard files
    thisFilename = 'polStandardTable_{0}.csv'.format(thisFilter)
    polTableFile = os.path.join(pyPol_data, thisFilename)

    # Read in the polarization calibration data file
    polCalTable = Table.read(polTableFile, format='ascii.csv')

    ###############
    # Get PE value
    ###############
    # # Grab the column names of the polarization measurements
    # polStart = lambda s: s.startswith('P_' + thisFilter)
    # polBool  = list(map(polStart, polCalTable.keys()))
    # polInds  = np.where(polBool)
    # polKeys  = np.array(polCalTable.keys())[polInds]

    # Initalize a dictionary to store all the calibration measurements
    tmpDict1 = {
        'value':[],
        'uncert':[]}
    tmpDict2 = {
        'expected':copy.deepcopy(tmpDict1),
        'measured':copy.deepcopy(tmpDict1)}
    polCalDict = {
        'P':copy.deepcopy(tmpDict2),
        'PA':copy.deepcopy(tmpDict2)}

    # Quickly build a list of calibration keys
    calKeyList = ['_'.join([prefix, thisFilter])
        for prefix in ['P', 'sP', 'PA', 'sPA']]

    # Loop over each row in the calibration data table
    for istandard, standard in enumerate(polCalTable):
        # Grab the appropriate row for this standard (as a table object)
        standardTable = polCalTable[np.array([istandard])]

        # Trim off unnecessary rows before looping over what remains
        standardTable.remove_columns(['Name', 'RA_1950', 'Dec_1950'])

        # Now loop over the remaining keys and
        for key in standardTable.keys():
            # Test if this is a calibration value
            if key in calKeyList: continue

            # Test if this value is masked
            if standardTable[key].data.mask: continue

            # If this is an unmasked, non-calibration value, then store it!
            # Find out the proper calibration key for polCalTable
            calKeyInd  = np.where([key.startswith(k) for k in calKeyList])
            thisCalKey = calKeyList[calKeyInd[0][0]]

            # Begin by parsing which key we're dealing with
            dictKey = (key.split('_'))[0]
            if dictKey.endswith('A'):
                dictKey = 'PA'
            elif dictKey.endswith('P'):
                dictKey = 'P'
            else:
                print('funky keys!')
                pdb.set_trace()

            # Parse whether this is a value or an uncertainty
            if key.startswith('s'):
                val_sig = 'uncert'
            else:
                val_sig = 'value'

            # Store the expected value
            try:
                polCalDict[dictKey]['expected'][val_sig].append(
                    standardTable[thisCalKey].data.data[0])
            except:
                pdb.set_trace()

            # Store the measured value
            polCalDict[dictKey]['measured'][val_sig].append(
                standardTable[key].data.data[0])


    ###################
    # Identify Outliers
    ###################
    # Grab the FULL set of expected and measured polarization values
    expectedPol         = np.array(polCalDict['P']['expected']['value'])
    uncertInExpectedPol = np.array(polCalDict['P']['expected']['uncert'])
    measuredPol         = np.array(polCalDict['P']['measured']['value'])
    uncertInMeasuredPol = np.array(polCalDict['P']['measured']['uncert'])

    # Run a statsmodels linear regression and test for outliers
    OLSmodel = smapi.OLS(
        expectedPol,
        measuredPol,
        hasconst=False
    )
    OLSregression = OLSmodel.fit()

    # Find the outliers
    outlierTest = OLSregression.outlier_test()
    outlierBool = [t[2] < 0.5 for t in outlierTest]

    # Grab the FULL set of expected and measured polarization values
    expectedPA         = np.array(polCalDict['PA']['expected']['value'])
    uncertInExpectedPA = np.array(polCalDict['PA']['expected']['uncert'])
    measuredPA         = np.array(polCalDict['PA']['measured']['value'])
    uncertInMeasuredPA = np.array(polCalDict['PA']['measured']['uncert'])

    # Run a statsmodels linear regression and test for outliers
    OLSmodel = smapi.OLS(
        expectedPA,
        measuredPA,
        hasconst=True
    )
    OLSregression = OLSmodel.fit()

    # Find the outliers
    outlierTest = OLSregression.outlier_test()
    outlierBool = np.logical_or(
        outlierBool,
        [t[2] < 0.5 for t in outlierTest]
    )

    # Cull the list of Ps and PAs
    goodInds            = np.where(np.logical_not(outlierBool))
    expectedPol         = expectedPol[goodInds]
    uncertInExpectedPol = uncertInExpectedPol[goodInds]
    measuredPol         = measuredPol[goodInds]
    uncertInMeasuredPol = uncertInMeasuredPol[goodInds]
    expectedPA          = expectedPA[goodInds]
    uncertInExpectedPA  = uncertInExpectedPA[goodInds]
    measuredPA          = measuredPA[goodInds]
    uncertInMeasuredPA  = uncertInMeasuredPA[goodInds]

    # TODO: print an update to the user on the polarization values culled

    ###############
    # Get PE value
    ###############
    # Close any remaining plots before proceeding to show the user the graphical
    # summary of the calibration data.
    plt.close('all')

    # Define the model to be used in the fitting
    def PE(slope, x):
         return slope*x

    # Set up ODR with the model and data.
    PEmodel = odr.Model(PE)
    data = odr.RealData(
        expectedPol,
        measuredPol,
        sx=uncertInExpectedPol,
        sy=uncertInMeasuredPol
    )

    # Initalize the full odr model object
    odrObj = odr.ODR(data, PEmodel, beta0=[1.])

    # Run the regression.
    PEout = odrObj.run()

    # Use the in-built pprint method to give us results.
    print(thisFilter + '-band PE fitting results')
    PEout.pprint()


    print('\n\nGenerating P plot')
    plt.ion()
    fig = plt.figure()
    ax  = fig.add_subplot(1,1,1)
    ax.errorbar(
        polCalDict['P']['expected']['value'],
        polCalDict['P']['measured']['value'],
        xerr=polCalDict['P']['expected']['uncert'],
        yerr=polCalDict['P']['measured']['uncert'],
        ecolor='b', linestyle='None', marker=None)
    xlim = ax.get_xlim()
    ax.plot([0,max(xlim)], PE(PEout.beta[0], np.array([0,max(xlim)])), 'g')
    plt.xlabel('Cataloged P [%]')
    plt.ylabel('Measured P [%]')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xlim = 0, xlim[1]
    ylim = 0, ylim[1]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.title(thisFilter + '-band Polarization Efficiency')

    #Compute where the annotation should be placed
    ySpan = np.max(ylim) - np.min(ylim)
    xSpan = np.max(xlim) - np.min(xlim)
    xtxt = 0.1*xSpan + np.min(xlim)
    ytxt = 0.9*ySpan + np.min(ylim)
    plt.text(xtxt, ytxt, 'PE = {0:4.3g} +/- {1:4.3g}'.format(
        PEout.beta[0], PEout.sd_beta[0]), fontdict=font)

    import pdb; pdb.set_trace()

    # Test if a polarization efficiency greater than one was retrieved...
    if PEout.beta[0] > 1.0:
        print('Polarization Efficiency greater than one detected.')
        print('Forcing PE constant to be 1.0')
        PEout.beta[0] = 1.0

    ###############
    # Get PA offset
    ###############
    # Fit a model to the PA1 vs. PA0 data
    # Define the model to be used in the fitting
    def deltaPA(B, x):
         return B[0]*x + B[1]

    # Set up ODR with the model and data.
    deltaPAmodel = odr.Model(deltaPA)
    data = odr.RealData(
        expectedPA,
        measuredPA,
        sx=uncertInExpectedPA,
        sy=uncertInMeasuredPA
    )

    # On first pass, just figure out what the sign is
    odrObj = odr.ODR(data, deltaPAmodel, beta0=[0.0, 90.0])
    dPAout = odrObj.run()
    PAsign = np.round(dPAout.beta[0])

    # Build the proper fitter class with the slope fixed
    odrObj = odr.ODR(data, deltaPAmodel, beta0=[PAsign, 90.0], ifixb=[0,1])

    # Run the regression.
    dPAout = odrObj.run()

    # Use the in-built pprint method to give us results.
    print(thisFilter + '-band delta PA fitting results')
    dPAout.pprint()

    # For ease of reference, convert the expected and measured values to arrays
    PA0 = np.array(polCalDict['PA']['expected']['value'])
    PA1 = np.array(polCalDict['PA']['measured']['value'])

    # Apply the correction terms
    dPAval = dPAout.beta[1]
    PAcor  = ((PAsign*(PA1 - dPAval)) + 720.0) % 180.0

    # TODO
    # Check if PAcor values are closer corresponding PA0_V values
    # by adding or subtracting 180
    PA0     = np.array(polCalDict['PA']['expected']['value'])
    PAminus = np.abs((PAcor - 180) - PA0 ) < np.abs(PAcor - PA0)
    if np.sum(PAminus) > 0:
        PAcor[np.where(PAminus)] = PAcor[np.where(PAminus)] - 180

    PAplus = np.abs((PAcor + 180) - PA0 ) < np.abs(PAcor - PA0)
    if np.sum(PAplus) > 0:
        PAcor[np.where(PAplus)] = PAcor[np.where(PAplus)] + 180

    # Do a final regression to plot-test if things are right
    data = odr.RealData(
        PA0,
        PAcor,
        sx=polCalDict['PA']['expected']['uncert'],
        sy=polCalDict['PA']['measured']['uncert']
    )
    odrObj    = odr.ODR(data, deltaPAmodel, beta0=[1.0, 0.0], ifixb=[0,1])
    dPAcor = odrObj.run()

    # Plot up the results
    # PA measured vs. PA true
    print('\n\nGenerating PA plot')
    fig.delaxes(ax)
    ax = fig.add_subplot(1,1,1)
    #ax.errorbar(PA0_V, PA1_V, xerr=sPA0_V, yerr=sPA1_V,
    #    ecolor='b', linestyle='None', marker=None)
    #ax.plot([0,max(PA0_V)], deltaPA(dPAout.beta, np.array([0,max(PA0_V)])), 'g')
    ax.errorbar(PA0, PAcor,
        xerr=polCalDict['PA']['expected']['uncert'],
        yerr=polCalDict['PA']['measured']['uncert'],
        ecolor='b', linestyle='None', marker=None)
    xlim = ax.get_xlim()
    ax.plot([0,max(xlim)], deltaPA(dPAcor.beta, np.array([0, max(xlim)])), 'g')
    plt.xlabel('Cataloged PA [deg]')
    plt.ylabel('Measured PA [deg]')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xlim = 0, xlim[1]
    ax.set_xlim(xlim)
    plt.title(thisFilter + '-band PA offset')

    #Compute where the annotation should be placed
    ySpan = np.max(ylim) - np.min(ylim)
    xSpan = np.max(xlim) - np.min(xlim)
    xtxt = 0.1*xSpan + np.min(xlim)
    ytxt = 0.9*ySpan + np.min(ylim)
    plt.text(xtxt, ytxt, 'PA offset = {0:4.3g} +/- {1:4.3g}'.format(
        dPAout.beta[1], dPAout.sd_beta[1]), fontdict=font)

    pdb.set_trace()

    # Now that all the calibration constants have been estimated and the results
    # shown to the user (in theory for their sanity-test approval), store the
    # final calibration data in the calTable variable
    calTable.add_row([thisFilter, PEout.beta[0], PEout.sd_beta[0],
                      np.int(PAsign), dPAout.beta[1], dPAout.sd_beta[1]])

    # Store a copy of polCalDict in allPolCalDict
    allPolCalDict[thisFilter] = copy.deepcopy(polCalDict)

# Now double check if the PA offsets are agreeable. If not, keep them separate,
# but otherwise attempt to combine them...
#####################################################
# Check if a single deltaPA value is appropriate
#####################################################
# Extract the originally estimated dPA values from the table
dPAvalues = calTable['D_PA'].data
dPAsigmas = calTable['s_D_PA'].data

# Compute all possible differences in dPAs and their uncertainties
D_dPAmatrix   = np.zeros(2*dPAvalues.shape)
s_D_dPAmatrix = np.ones(2*dPAvalues.shape)

for i in range(len(dPAvalues)):
    for j in range(len(dPAvalues)):
        # Skip over trivial or redundant elements
        if j <= i: continue
        D_dPAmatrix[i,j]   = np.abs(dPAvalues[i] - dPAvalues[j])
        s_D_dPAmatrix[i,j] = np.sqrt(dPAsigmas[i]**2 + dPAsigmas[j]**2)

# Check if this these two values are significantly different from each-other
if (D_dPAmatrix/s_D_dPAmatrix > 3.0).any():
    print('Some of these calibration constants are significantly different.')
    print('Leave them as they are.')
else:
    PA0  = []
    PA1  = []
    sPA0 = []
    sPA1 = []
    for key, val in allPolCalDict.items():
        PA0.extend(val['PA']['expected']['value'])
        PA1.extend(val['PA']['measured']['value'])
        sPA0.extend(val['PA']['expected']['uncert'])
        sPA1.extend(val['PA']['measured']['uncert'])

    # Do a final regression to plot-test if things are right
    data = odr.RealData(PA0, PA1, sx=sPA0, sy=sPA1)
    # On first pass, just figure out what the sign is
    odrObj    = odr.ODR(data, deltaPAmodel, beta0=[0.0, 90.0])
    dPAout = odrOjb.run()
    PAsign = np.round(dPAout.beta[0])

    # Build the proper fitter class with the slope fixed
    odrObj = odr.ODR(data, deltaPAmodel, beta0=[PAsign, 90.0], ifixb=[0,1])

    # Run the regression.
    dPAout = odrObj.run()

    # Use the in-built pprint method to give us results.
    print('Final delta PA fitting results')
    dPAout.pprint()

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
    data = odr.RealData(PA0, PAcor, sx=sPA0, sy=sPA1)
    odrObj  = odr.ODR(data, deltaPAmodel, beta0=[1.0, 0.0], ifixb=[0,1])
    dPAcor = odrObj.run()

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

    # Pause for a double check from the user
    pdb.set_trace()

    # User approves, close the plot and proceed
    plt.close()
    plt.ioff()

    # Update the calibration table
    calTable['D_PA']   = dPAout.beta[1]
    calTable['s_D_PA'] = dPAout.sd_beta[1]

print('Writing calibration data to disk')
calTable.write(polCalConstantsFile, format='ascii.csv')

print('Calibration tasks completed!')
