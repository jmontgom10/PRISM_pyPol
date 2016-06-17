# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 14:57:09 2015

@author: jordan
"""
import os
import sys
import numpy as np
from astropy.table import Table
from astropy.table import Column as Column
import pdb

# Add the AstroImage class
sys.path.append("C:\\Users\\Jordan\\Libraries\\python\\AstroImage")
from AstroImage import AstroImage

# This script will run the astrometry step of the pyPol reduction

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
fileIndex = Table.read(indexFile, format='ascii.csv')
fileList  = fileIndex['Filename']

# Loop through each file and perform its astrometry method
for file in fileList:
    # Read in the file
    tmpImg  = AstroImage(file)

    # Do the astrometry with Atrometry.net
    success = tmpImg.astrometry()
    if success:
        # If the astrometry was solved, then proceed to write the astro
        tmpImg.write()
    else:
        print('Failed to get astrometry for ')
        print(file)
