# -*- coding: utf-8 -*-
"""
Invokes the Astrometry.net engine to solve the astrometry for each file.
"""

import os
import sys
import numpy as np
from astropy.table import Table, Column

# Add the AstroImage class
import astroimage as ai

# This script will run the astrometry step of the pyPol reduction

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================
# This is the location of all pyBDP data (index, calibration images, reduced...)
pyBDP_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyBDP_data\\201612'

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data\\201612'

# This is the location of the pyBDP processed Data
pyBDP_reducedDir = os.path.join(pyBDP_data, 'pyBDP_reduced_images')

# Read in the indexFile data and select the filenames
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='ascii.csv')
fileList  = fileIndex['FILENAME']

# Loop through each file and perform its astrometry method
for iFile, filename in enumerate(fileList):
    # Skip background files, they won't need astrometry.
    if fileIndex[iFile]['AB'] == 'B':
        continue

    # Read in the file
    tmpImg = ai.ReducedScience.read(filename)

    # Construct an AstrometrySolver object
    astroSolver = ai.AstrometrySolver(tmpImg)

    # Run the astrometry.net solver
    tmpImg, success = astroSolver.run()

    # If it worked, then write it back to disk, otherwise just continue
    if success:
        # If the astrometry was solved, then proceed to write the astro
        tmpImg.write(dtype=np.float32, clobber=True)
    else:
        print('Failed to get astrometry for {}'.format(filename))
