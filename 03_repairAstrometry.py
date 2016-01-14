# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:56:02 2015

@author: jordan
"""

import os
import numpy as np
from astropy.io import ascii
from astropy.table import Column as Column
import astropy.coordinates as coord
import astropy.units as u
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib import pyplot as plt
import pdb

# Add the AstroImage class
sys.path.append("C:\\Users\\Jordan\\Libraries\\python\\AstroImage")
from AstroImage import AstroImage

# This script will run the image averaging step of the pyPol reduction

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================
# This is the location of all pyBDP data (index, calibration images, reduced...)
pyBDP_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyBDP_data'

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data'

# This is the location of the pyBDP processed Data
pyBDP_reducedDir = os.path.join(pyBDP_data, 'pyBDP_reduced_images')

# Read in the indexFile data and select the filenames
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='csv')

# Setup new directory for plate scale histograms data
delim         = os.path.sep
astroCheckDir = os.path.join(pyPol_data, 'astrometryCheck')
if (not os.path.isdir(astroCheckDir)):
    os.mkdir(astroCheckDir, 0o755)

failedAstroFile = os.path.join(astroCheckDir, 'failedAstro.dat')

# Determine which parts of the Fileindex pertain to science images
keepFiles = (fileIndex['Use'] == 1)

# Update the fileIndex with the paths to reduced files
fileIndex = fileIndex[np.where(keepFiles)]
fileIndex['Filename'] = fileList

# Group the fileIndex by...
# 1. Target
# 2. Waveband
# 3. Dither (pattern)
# 4. Binning
fileIndexByTarget = fileIndex.group_by(['Target', 'Waveband', 'Dither', 'Binning'])
for group in fileIndexByTarget.groups:
    # Grab the current target information
    thisTarget   = str(np.unique(group['Target'].data)[0])
    thisWaveband = str(np.unique(group['Waveband'].data)[0])
    thisDither   = str(np.unique(group['Dither'].data)[0])
    thisBinning  = str(np.unique(group['Binning'].data)[0])

    numImgs      = len(group)
    print('\nProcessing {0} images for'.format(numImgs))
    print('\tTarget        : {0}'.format(thisTarget))
    print('\tWaveband      : {0}'.format(thisWaveband))
    print('\tDither        : {0}'.format(thisDither))
    print('\tBinning       : {0}'.format(thisBinning))

    # Get plate scales for all the images of this target.
    # Load the plate scales into a list
    targetBinningList    = []
    targetPC_matrixList  = []
    targetPlateScaleList = []

    # Loop through the files
    for file in group['Filename']:
        # Load the image header
        tmpImage  = Image(file)
        tmpHeader = tmpImage.header

        # Grab the WCS
        tmpWCS    = WCS(tmpHeader)

        # Grab the PC macrices and pixel scales and store them in the list
        targetBinningList.append(tmpImage.binning)
        targetPC_matrixList.append(tmpWCS.wcs.get_pc())
        targetPlateScaleList.append(proj_plane_pixel_scales(tmpWCS))

    # Histogram the plate scales along each axis
    targetBinning     = (np.tile(np.array(targetBinningList), (2,1))).T
    targetPC_matrices = np.array(targetPC_matrixList)
    targetPlateScales = np.array(targetPlateScaleList) * 3600.0 / targetBinning

    # Check for significant deviates
    deviation    = np.abs(targetPlateScales - np.median(targetPlateScales))
    badAstroBool = np.logical_or((deviation[:,0] > 0.05),
                                 (deviation[:,1] > 0.05))
    badAstroInds = np.where(badAstroBool)[0]
    # If there were bad astrometries, then record them
    if len(badAstroInds) > 0:
        # Determine the good astrometry values
        goodAstroInds = np.where(np.logical_not(badAstroBool))[0]
        goodAstroVals = targetPC_matrices[goodAstroInds]
        goodAstroVals = np.mean(goodAstroVals, axis=0)

        # Read in a good image to get a good WCS
        goodImg = Image(group['Filename'][goodAstroInds[0]])
        goodWCS = WCS(goodImg.header)

        # Loop through all the bad files and "fix them"
        for badInd in badAstroInds:
            # Build a quick header from the WCS object
            thisFile = group['Filename'][badInd]
            thisImg  = Image(thisFile)

            # Grab the approximate pointing from the header
            thisRA  = coord.Angle(thisImg.header['TELRA'], unit=u.hour)
            thisRA.degree
            thisDec = coord.Angle(thisImg.header['TELDEC'], unit=u.degree)

           # Update the image header to contain the astrometry info
            thisImg.header['CTYPE1']  = 'RA---TAN-SIP'
            thisImg.header['CTYPE2']  = 'DEC--TAN-SIP'
            thisImg.header['CRPIX1']  = thisImg.arr.shape[1]//2
            thisImg.header['CRPIX2']  = thisImg.arr.shape[0]//2
            thisImg.header['CRVAL1']  = thisRA.deg
            thisImg.header['CRVAL2']  = thisDec.deg
            thisImg.header['PC1_1']   = goodAstroVals[0,0]
            thisImg.header['PC1_2']   = goodAstroVals[0,1]
            thisImg.header['PC2_1']   = goodAstroVals[1,0]
            thisImg.header['PC2_2']   = goodAstroVals[1,1]
            thisImg.header['CDELT1']  = 1.0
            thisImg.header['CDELT2']  = 1.0
            thisImg.header['CUNIT1']  = 'deg'
            thisImg.header['CUNIT2']  = 'deg'
            thisImg.header['LONPOLE'] = 180.0
            thisImg.header['LATPOLE'] = -2.2    #Does this need to be calculated somehow?
            thisImg.header['RADESYS'] = 'FK5'

            # Insert repaired pc matrix and plate scale
            tmpWCS                    = WCS(thisImg.header)
            targetPC_matrices[badInd] = tmpWCS.wcs.get_pc()
            targetPlateScales[badInd] = proj_plane_pixel_scales(tmpWCS) * 3600 / thisImg.binning

            # Write updated file properties to disk
            thisImg.write()

            thisFile = os.path.basename(group['Filename'][badInd])
            print('Logging bad file: ' + thisFile)
            os.system('echo "' + thisFile +
                      '" >> ' + failedAstroFile)

    # Prep for histogram plotting
    minPS    = np.min(targetPlateScales)
    maxPS    = np.max(targetPlateScales)
    binWidth = 0.0005
    numBins  = np.int(np.ceil((maxPS - minPS)/binWidth)) + 2
    binEdges = np.linspace(minPS-binWidth, maxPS+binWidth, numBins)

    n1, bins1, patches1 = plt.hist(targetPlateScales[:,0], binEdges, histtype='stepfilled',
                                rwidth=1, facecolor='red', alpha=0.7)
    n2, bins2, patches2 = plt.hist(targetPlateScales[:,1], binEdges, histtype='stepfilled',
                                rwidth=1, facecolor='blue', alpha=0.4)

    filename = (astroCheckDir + delim +
                '_'.join([thisTarget, thisWaveband, thisBinning]) + '.png')
    plt.savefig(filename)
    plt.clf()
