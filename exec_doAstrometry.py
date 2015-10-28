# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 14:57:09 2015

@author: jordan
"""
import os
import numpy as np
import pdb
from pyPol import Image

# This script will run the astrometry step of the pyPol reduction

#Setup the path delimeter for this operating system
delim = os.path.sep

# Grab all the *.fits files in the reduced science data directory
reducedDir = '/home/jordan/ThesisData/PRISM_Data/Reduced_data'
fileList   = []
for file in os.listdir(reducedDir):
    filePath = os.path.join(reducedDir, file)
    fileTest = os.path.isfile(filePath)
    extTest  = (os.path.splitext(filePath)[1] == '.fits')
    if fileTest and extTest:
        fileList.extend([os.path.join(reducedDir, file)])

# Sort the fileList
fileNums = [''.join((file.split(delim).pop().split('.'))[0:2]) for file in fileList]
fileNums = [num.split('_')[0] for num in fileNums]
sortInds = np.argsort(np.array(fileNums, dtype = np.int))
fileList = [fileList[ind] for ind in sortInds]

# Create an empty file for storing the names of failed astrometry files.
if not os.path.isfile('failedAstro.dat'):
    os.system('> failedAstro.dat')

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
        os.system('echo "' + os.path.basename(file) + '" >> failedAstro.dat')

# Clean up residual Astrometry.net file
if os.path.isfile('none'):
    os.system('rm none')
