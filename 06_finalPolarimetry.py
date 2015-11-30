# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 09:34:43 2015

@author: jordan
"""

import os
import numpy as np
from astropy.io import ascii
from astropy.table import Table as Table
from astropy.table import Column as Column
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from photutils import detect_sources, Background
import pdb
from pyPol import Image
from interactive_alignment import *

#Setup the path delimeter for this operating system
delim = os.path.sep

# Grab all the *.fits files in the reduced science data directory
reducedDir = '/home/jordan/ThesisData/PRISM_Data/Reduced_data'

# Setup new directory for polarimetry data
polarimetryDir = reducedDir + delim + 'Polarimetry'
if (not os.path.isdir(polarimetryDir)):
    os.mkdir(polarimetryDir, 0o755)

polAngDir = polarimetryDir + delim + 'polAngImgs'
if (not os.path.isdir(polAngDir)):
    os.mkdir(polAngDir, 0o755)

stokesDir = polarimetryDir + delim + 'stokesImgs'
if (not os.path.isdir(stokesDir)):
    os.mkdir(stokesDir, 0o755)

# Read in the indexFile data and select the filenames
print('\nReading file index from disk')
indexFile = 'fileIndex.csv'
fileIndex = Table.read(indexFile, format='csv')
#fileIndex = ascii.read(indexFile, guess=False, delimiter=',')
fileList  = fileIndex['Filename']

# Determine which parts of the fileIndex pertain to science images
useFiles = np.logical_and((fileIndex['Use'] == 1), (fileIndex['Dither'] == 'ABBA'))

# Cull the file index to only include files selected for use
fileIndex = fileIndex[np.where(useFiles)]

# Group the fileIndex by...
# 1. Target
# 2. Waveband
# 3. Dither (pattern)
fileIndexByTarget = fileIndex.group_by(['Target', 'Waveband', 'Dither'])

# Define any required conversion constants
rad2deg = (180.0/np.pi)
deg2rad = (np.pi/180.0)
wavebands = ('R', 'V')
deltaPA = (16.0*deg2rad, 19.0*deg2rad) #degrees converted into radians
deltaPA = dict(zip(wavebands, deltaPA))

# Loop through each group
groupKeys = fileIndexByTarget.groups.keys
for group in fileIndexByTarget.groups:
    # Grab the current target information
    thisTarget   = str(np.unique(group['Target'].data)[0])
    thisWaveband = str(np.unique(group['Waveband'].data)[0])
    numImgs      = len(group)
    print('\nProcessing images for'.format(numImgs))
    print('\tTarget   : {0}'.format(thisTarget))
    print('\tWaveband : {0}'.format(thisWaveband))
    
    # Read in the polAng image
    fileCheck  = True
    polAngImgs = []
    polAngs    = []
    for polAng in range(0,601,200):
        fileName = (polAngDir + delim +
                    '_'.join([thisTarget, thisWaveband, str(polAng)]) + '.fits')
        
        # If the file exists, then read it into a dictionary
        if os.path.isfile(fileName):
            polAngs.append(polAng)
            polAngImgs.append(Image(fileName))
        else:
            fileCheck = False
    
    # Check that all the necessarry files are present
    if not fileCheck:
        print("\tSome required Polaroid Angle files are missing.")
        continue
    
    # Use the "align_stack" method to align the newly created image list
    polAngImgs = Image.align_stack(polAngImgs, mode='cross_correlate', subPixel=True)
    
    # Convert the list into a dictionary
    polAngImgs = dict(zip(polAngs, polAngImgs)) #aligned
    
    #**********************************************************************
    # Stokes I
    #**********************************************************************
    stokesI = polAngImgs[0].copy()
    stokesIarr = Image.stacked_average([polAngImgs[0],
                                        polAngImgs[200],
                                        polAngImgs[400],
                                        polAngImgs[600]])
    stokesI.arr = 2 * stokesIarr
    
    #**********************************************************************
    # Stokes U
    #**********************************************************************
    # Subtract the images to get stokes U
    stokesU = (polAngImgs[200] - polAngImgs[600])/(polAngImgs[200] + polAngImgs[600])
    
    # Mask out masked values
    Umask   = np.logical_or(polAngImgs[200].arr == 0,
                            polAngImgs[600].arr == 0)
    
    #**********************************************************************
    # Stokes Q
    #**********************************************************************
    # Subtract the images to get stokes Q
    stokesQ = (polAngImgs[0] - polAngImgs[400])/(polAngImgs[0] + polAngImgs[400])
    
    # Mask out masked values
    Qmask   = np.logical_or(polAngImgs[0].arr   == 0,
                            polAngImgs[400].arr == 0)
    stokesQ.arr[np.where(Qmask)] = 0
    
    # Apply the PA correction factor
    delPA = deltaPA[thisWaveband]
    Qrot = (np.cos(2*delPA)*stokesQ.arr
          - np.sin(2*delPA)*stokesU.arr)
    Urot = (np.sin(2*delPA)*stokesQ.arr
          + np.cos(2*delPA)*stokesU.arr)
    stokesU.arr = Urot
    stokesQ.arr = Qrot
    
    # Apply masks to rotated Stokes images
    # Generate a complete mask of all bad elements
    stokesU.arr[np.where(Umask)] = 0
    stokesQ.arr[np.where(Qmask)] = 0
    fullMask = np.logical_or(Umask, Qmask)
    
    #**********************************************************************
    # Pmap
    #**********************************************************************
    Pmap = stokesU.copy()
    tmpImg = stokesU*stokesU + stokesQ*stokesQ
    Pmap.arr = np.sqrt(tmpImg.arr)
    
    #**********************************************************************
    # PAmap
    #**********************************************************************
    # Compute the PA map
    # (the minus sign in front of the arctan makes PA increase CCW)
    PAmap     = stokesQ.copy()
    tmpArr    = -0.5*np.arctan2(stokesU.arr, stokesQ.arr)*rad2deg
    tmpArr    = (tmpArr + 2*360) % 180
    PAmap.arr = tmpArr
    
    #**********************************************************************
    # Final masking and writing to disk
    #**********************************************************************
    stokesU.arr[np.where(Umask)]  = np.NaN
    stokesQ.arr[np.where(Qmask)]  = np.NaN
    Pmap.arr[np.where(fullMask)]  = np.NaN
    PAmap.arr[np.where(fullMask)] = np.NaN

    # Generate the output filenames
    Ifile = (stokesDir + delim +
               '_'.join([thisTarget, thisWaveband, 'I']) + '.fits')
    Ufile = (stokesDir + delim +
               '_'.join([thisTarget, thisWaveband, 'U']) + '.fits')
    Qfile = (stokesDir + delim +
               '_'.join([thisTarget, thisWaveband, 'Q']) + '.fits')
    Pfile = (polarimetryDir + delim +
               '_'.join([thisTarget, thisWaveband, 'P']) + '.fits')
    PAfile = (polarimetryDir + delim +
               '_'.join([thisTarget, thisWaveband, 'PA']) + '.fits')
    
    # Write to disk
    stokesI.write(Ifile)
    stokesU.write(Ufile)
    stokesQ.write(Qfile)
    Pmap.write(Pfile)
    PAmap.write(PAfile)
    