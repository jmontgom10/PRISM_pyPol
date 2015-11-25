# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 14:57:09 2015

@author: jordan
"""
import os
import numpy as np
from astropy.table import Table
from astropy.table import Column as Column
from astropy.io import ascii
import pdb
from pyPol import Image

# This script will run the astrometry step of the pyPol reduction

# Read in the indexFile data and select the filenames
indexFile = 'fileIndex.csv'
fileIndex = Table.read(indexFile, format='csv')
#fileIndex = ascii.read(indexFile, guess=False, delimiter=',')
fileList  = fileIndex['Filename']

# Loop through each file and perform its astrometry method
for file in fileList:
    # Read in the file
    tmpImg  = Image(file)
    
    # Do the astrometry with Atrometry.net
    success = tmpImg.astrometry()
    if success:
        # If the astrometry was solved, then proceed to write the astro
        tmpImg.write()
    else:
        print('Failed to get astrometry for ')
        print(file)

# Clean up residual Astrometry.net file
if os.path.isfile('none'):
    os.system('rm none')
