# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 09:34:43 2015

@author: jordan
"""

import os
import numpy as np
from astropy.io import ascii
from astropy.table import Column as Column
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from photutils import detect_sources, Background
import pdb
from pyPol import Image

#******************************************************************************
# Write a quick function to provide list string searching...
def str_list_contains(inList, searchStr):
    """This function searches the elements of a list of strings for matches.
    
    parameters:
    inList    -- a list containing ONLY strings
    searchStr -- a string to search for...
    """
    
    # Check that the searchStr parameter is a string
    if not isinstance(searchStr, str):
        print('The searhcStr parameter must be a string')
        return None
    
    outList = []
    for el in inList:
        # Check that this element is also a string
        if not isinstance(el, str):
            print('All elements of the inList parameter must be pure strings')
            return None
        
        outList.append(el.__contains__(searchStr))
    
    return outList
#******************************************************************************


# This script will run the polarimetry step of the pyPol reduction

#******************************************************************************
# First the user must identify the names of the targets to be batched
#******************************************************************************

#Setup the path delimeter for this operating system
delim = os.path.sep

# Grab all the *.fits files in the reduced science data directory
reducedDir     = '/home/jordan/ThesisData/PRISM_Data/Reduced_data'
polarimetryDir = reducedDir + delim + 'Polarimetry'

# Specify target names and wavebands
targets    = ['M104', 'M78', 'M82', 'NGC2023', 'NGC7023', 'NGC_1977']
wavebands  = ['V', 'R']
#outputImgs = ['U','Q','P','PA','Intensity']


# Loop through each target
for target in targets:

    #**************************************************************************
    # NGC2023 IS TEMPORARILY OUT OF ORDER
    #**************************************************************************
    if target == 'NGC2023':
        print('Skipping target NGC2023')
        continue

    # Loop through each waveband for each target
    for waveband in wavebands:
        print('\nProcessing images for {0}_{1}'.format(target, waveband))
        
        # Check that all the necessarry files are present
        fileCheck  = True
        polAngImgs = dict()
        for polAng in range(0,601,200):
            fileName = (polarimetryDir + delim +
                        '_'.join([target, waveband, str(polAng)]) + '.fits')

            # If the file exists, then read it into a dictionary
            if os.path.isfile(fileName):
                polAngImgs[polAng] = Image(fileName)
            else:
                fileCheck = False
        
        if not fileCheck:
            print("\tSome required Polaroid Angle files are missing.")
            continue
        
        #**********************************************************************
        # Stokes Q
        #**********************************************************************
        # Generate the output filename
        outFile = (polarimetryDir + delim +
                   '_'.join([target, waveband, 'Q']) + '.fits')
        
        twoImgs = polAngImgs[400].align(polAngImgs[0])#, fractionalShift=True)
        
        stokesQ = twoImgs[0] - twoImgs[1]
        pdb.set_trace()
        
        #**********************************************************************
        # Stokes U
        #**********************************************************************
        # Generate the output filename
        outFile = (polarimetryDir + delim +
                   '_'.join([target, waveband, 'U']) + '.fits')
        
        stokesU = polAngImgs[200] - polAngImgs[600]

        #**********************************************************************
        # Pmap
        #**********************************************************************
        # Generate the output filename
        outFile = (polarimetryDir + delim +
                   '_'.join([target, waveband, 'P']) + '.fits')
        Pmap     = stokesQ.cop()
        tmpImg   = stokesU*stokesU + stokesQ*stokesQ
        Pmap.arr = np.sqrt(tmpImg.arr)

        #**********************************************************************
        # PAmap
        #**********************************************************************
        # Generate the output filename
        outFile = (polarimetryDir + delim +
                   '_'.join([target, waveband, 'PA']) + '.fits')
        
        PAmap     = stokesQ.copy()
        PAmap.arr = 0.5*np.arctan2(stokesU.arr + stokesQ.arr)
        pdb.set_trace()