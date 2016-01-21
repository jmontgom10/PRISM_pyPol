# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:59:25 2015

@author: jordan
"""

import os
import sys
import numpy as np
from astropy.io import ascii
from astropy.table import Table as Table
from astropy.table import Column as Column
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from photutils import detect_sources, Background
import pdb

# Add the AstroImage class
sys.path.append("C:\\Users\\Jordan\\Libraries\\python\\AstroImage")
from AstroImage import AstroImage

# This script will compute the photometry of polarization standard stars
# and output a file containing the polarization position angle
# additive correction and the polarization efficiency of the PRISM instrument.

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data'


# The user needs to specify the "Target" values associated with
# calibration data in the fileIndex.
targets = ['Orion_Cal_20150117',
           'Orion_Cal_20150119',
           'Taurus_Cal_20150118']

# This is the location of the previously generated masks (step 4)
maskDir = os.path.join(pyPol_data, 'Masks')

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
fileIndex = Table.read(indexFile, format='csv')

# Read in the polarization standards file
print('Reading polarization standards from disk')
polStandardFile = os.path.join('polStandards.csv')
polStandards = Table.read(polStandardFile, format='csv')

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
fileIndexByTarget = fileIndex.group_by(['Target', 'Night', 'Waveband', 'Dither', 'Polaroid Angle'])


for group in fileIndexByTarget.groups:
    # Grab the current target information
    thisTarget   = str(np.unique(group['Target'].data)[0])
    thisNight    = str(np.unique(group['Night'].data)[0])
    thisWaveband = str(np.unique(group['Waveband'].data)[0])
    thisPolAng   = str(np.unique(group['Polaroid Angle'].data)[0])

    print('\nProcessing image for')
    print('\tTarget         : {0}'.format(thisTarget))
    print('\tNight          : {0}'.format(thisNight))
    print('\tWaveband       : {0}'.format(thisWaveband))
    print('\tPolaroid Angle : {0}'.format(thisPolAng))

    inFile = os.path.join(polAngDir,
        '_'.join([thisTarget, thisWaveband, thisPolAng]) + '.fits')

    print('\nComputing calibration photometry')
    thisImg = AstroImage(inFile)

    pdb.set_trace()

print('Done with this script!')
