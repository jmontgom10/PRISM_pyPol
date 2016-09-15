# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:08:13 2016

@author: jordan
"""
import os
import sys
import subprocess
import warnings
from datetime import datetime
import numpy as np
from astropy.table import Table, hstack
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from astropy import units as u
from astropy.coordinates import SkyCoord
from photutils import CircularAperture
from astropy.modeling import models, fitting
from astroquery.vizier import Vizier
import scipy.odr as odr
# import statsmodels.api as smapi
# from statsmodels.formula.api import ols
# import statsmodels.graphics as smgraphics
from matplotlib import pyplot as plt
from photutils import daofind, aperture_photometry, CircularAperture, CircularAnnulus
import pdb

# Add the AstroImage class
sys.path.append("C:\\Users\\Jordan\\Libraries\\python\\AstroImage")
import image_tools
from AstroImage import AstroImage

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data'

# Setup new directory for polarimetry data
polarimetryDir = os.path.join(pyPol_data, 'Polarimetry')
if (not os.path.isdir(polarimetryDir)):
    os.mkdir(polarimetryDir, 0o755)

polAngDir = os.path.join(polarimetryDir, 'polAngImgs')
if (not os.path.isdir(polAngDir)):
    os.mkdir(polAngDir, 0o755)

stokesDir = os.path.join(polarimetryDir, 'stokesImgs')
if (not os.path.isdir(stokesDir)):
    os.mkdir(stokesDir, 0o755)


#******************************************************************************
# Define the event handlers for clicking and keying on the image display
#******************************************************************************

def on_click(event):
    global fig, artCollect, selectedStars, outFile, xStars, yStars

    x, y = event.xdata, event.ydata

    # Compute star distances from the click and update mask array
    dist = np.sqrt((xStars - x)**2 + (yStars - y)**2)

    # Toggle the selected value of the nearest star
    thisStar = np.where(dist == np.min(dist))
    selectedStars[thisStar] = not selectedStars[thisStar]

    # Build an array of colors to match which stars have been selected
    colors = ['green' if t else 'red' for t in selectedStars]

    # Update the plot to reflect the latest selection
    for artist, color in zip(artCollect, colors):
        artist.set_edgecolor(color)

    # Update the display
    fig.canvas.draw()

def on_key(event):
    global fig, artCollect, selectedStars, starCatalog

    # Save the generated mask
    if event.key == 'enter':
        print('Finished with current epoch')
        plt.close(fig)

        # Disconnect the event manager and close the figure
        fig.canvas.mpl_disconnect(cid1)
        fig.canvas.mpl_disconnect(cid2)

    else:
        print('Do not recognize key command')
#******************************************************************************

#******************************************************************************
# This is the main script that will load in file names and prepare for plotting
#******************************************************************************

# Declare global variables
global fig, artCollect, selectedStars, starCatalog, outFile, xStars, yStars

xList     = []
yList     = []
imgNum    = 0      # This number will be the FIRST image to be displayed center...
brushSize = 3      # (5xbrushSize pix) is the size of the region masked

#******************************************************************************
# This script will retrieeve the USNO-B1.0 data from Vizier and allow the user
# to select stars suitable for photometry
#******************************************************************************
# Reset ROW_LIMIT property to retrieve FULL catalog from Vizier
Vizier.ROW_LIMIT = -1

# Read in the indexFile data and select the filenames
print('\nReading file index from disk')
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='ascii.csv')

fileList  = fileIndex['Filename']

# Determine which parts of the fileIndex pertain to science images
useFiles = np.logical_and((fileIndex['Use'] == 1), (fileIndex['Dither'] == 'ABBA'))

# Cull the file index to only include files selected for use
fileIndex = fileIndex[np.where(useFiles)]

# Group the fileIndex by...
# 1. Target
# 2. Waveband
# 3. Dither (pattern)
fileIndexByTarget = fileIndex.group_by(['Target', 'Dither'])

# Loop through each group
groupKeys = fileIndexByTarget.groups.keys
for group in fileIndexByTarget.groups:
    # Grab the current target information
    thisTarget   = str(np.unique(group['Target'].data)[0])
    thisWaveband = str(np.unique(group['Waveband'].data)[0])

    print('\nProcessing images for')
    print('\tTarget   : {0}'.format(thisTarget))

    outFile = os.path.join(stokesDir, thisTarget + '_stars.csv')

    # Read in the Stokes intensity image
    I_V_file  = os.path.join(stokesDir,
             '_'.join([thisTarget, 'V', 'I']) + '.fits')
    stokesI_V = AstroImage(I_V_file)
    I_R_file  = os.path.join(stokesDir,
             '_'.join([thisTarget, 'R', 'I']) + '.fits')
    stokesI_R = AstroImage(I_R_file)

    # Store a "pixel counter" map in the sigma attributes of these images
    stokesI_V.sigma = np.ones(stokesI_V.arr.shape, dtype=int)
    stokesI_R.sigma = np.ones(stokesI_R.arr.shape, dtype=int)

    # Align intensity images and separate out pixel counting
    tmpAlignedImgs  = stokesI_V.align(stokesI_R)
    alignedImgs  = []
    pixCountImgs = []
    for imgNum, img in enumerate(tmpAlignedImgs):
        # Make a clean copy of the stokes image for image combination
        tmpImg = img.copy()
        delattr(tmpImg, 'sigma')
        alignedImgs.append(tmpImg)

        # Make a clean copy of the pixel counts for image cropping
        tmpImg     = img.copy()
        tmpCount   = (np.abs(tmpImg.sigma - 1) < 1e-2).astype(int)
        tmpImg.arr = tmpCount
        delattr(tmpImg, 'sigma')
        pixCountImgs.append(tmpImg)

    # Clean up temporary variables
    del tmpImg, tmpAlignedImgs

    # Combine intensity images (don't really care about accuracy, so let's just
    # do a straightforward arithmetic mean)
    stokesI1 = np.sum(alignedImgs)/float(len(alignedImgs))

    # Grab the image center pixel coordinates
    ny, nx = stokesI1.arr.shape
    yc, xc = ny//2, nx//2

    # Determine the region common to BOTH V- and R-band images
    pixCount = np.zeros(alignedImgs[0].arr.shape, dtype = int)
    for img in pixCountImgs:
        pixCount += img.arr.astype(int)

    # Find the actual cut points for the cropping of the "common image"
    maxCount = len(alignedImgs)
    goodCol  = np.where(np.abs(pixCount[:,xc] - maxCount) < 1e-2)
    goodRow  = np.where(np.abs(pixCount[yc,:] - maxCount) < 1e-2)
    bt, tp   = np.min(goodCol), np.max(goodCol)
    lf, rt   = np.min(goodRow), np.max(goodRow)

    # Perform the actual crop
    stokesI1.crop(lf, rt, bt, tp)

    # Resolve the astrometry
    stokesI1.filename='tmp.fits'
    stokesI1.write()
    stokesI, success = image_tools.astrometry(stokesI1)
    if success:
        # Grab the WCS of this file
        thisWCS = WCS(stokesI.header)

        # Delete the temporary file
        # Test what kind of system is running
        if 'win' in sys.platform:
            # If running in Windows,
            delCmd = 'del '
            shellCmd = True
        else:
            # If running a *nix system,
            delCmd = 'rm '
            shellCmd = False

        # Finally delete the temporary fits file
        tmpFile = os.path.join(os.getcwd(), 'tmp.fits')
        rmProc = subprocess.Popen(delCmd + tmpFile, shell=shellCmd)
        rmProc.wait()
        rmProc.terminate()
    else:
        print('Astrometry failed')
        pdb.set_trace()

    # Grab the image center pixel coordinates
    ny, nx = stokesI.arr.shape
    yc, xc = ny/2.0, nx/2.0

    # Compute image center WCS coordinates
    cenRA, cenDec = thisWCS.wcs_pix2world(xc, yc, 0)
    cenCoord      = SkyCoord(cenRA, cenDec, unit=('deg','deg'), frame='icrs')

    # Compute image width and height
    corners = thisWCS.wcs_pix2world([0, stokesI.arr.shape[0]],
        [0, stokesI.arr.shape[1]], 0)
    width   = (corners[0][0] - corners[0][1])*u.deg
    height  = (corners[1][1] - corners[1][0])*u.deg

    # Download the USNO-B1.0 catalog for this region
    starCatalog = Vizier.query_region(cenCoord, width = width, height = height,
        catalog='USNO-B1')

    # Test if only one catalog was returned
    if len(starCatalog) == 1:
        starCatalog = starCatalog[0]
    else:
        print('Catalog confusion. More than one result found!')
        pdb.set_trace()

    # Cull the catalog to only include entries with all R and B band entries
    keepStars1 = np.logical_not(starCatalog['B1mag'].mask)
    keepStars1 = np.logical_and(
        keepStars1,
        np.logical_not(starCatalog['R1mag'].mask))
    keepStars2 = np.logical_not(starCatalog['B2mag'].mask)
    keepStars2 = np.logical_and(
        keepStars2,
        np.logical_not(starCatalog['R2mag'].mask))

    # Keep any stars from which we can generate at least ONE photometry estimate
    keepStarsList = [keepStars1, keepStars2]

    apRad      = 8.0


    # Compute image count scaling
    mean, median, stddev = sigma_clipped_stats(stokesI.arr.flatten())
    thisMin, thisMax = mean - 2*stddev, mean + 13*stddev

    # Initalize a list to store the indices of all the selected stars
    finalKeepInds = []
    for iEpoch, keepStars in enumerate(keepStarsList):
        # If at least some stars pass the test, then cull the catalog to only
        # include those stars.
        if np.sum(keepStars) > 0:
            keepInds     = (np.where(keepStars))[0]
            starCatalog1 = starCatalog[keepInds]
        else:
            print('Error: No passable stars from this epoch')
            continue

        # Form a "catalog" of pixel position entries. These will be plotted over the
        # image, and the user will be asked to select which stars from the list are
        # suitable for photometry.
        ra1  = starCatalog1['_RAJ2000'].data.data*u.deg
        dec1 = starCatalog1['_DEJ2000'].data.data*u.deg

        # Propagate proper motions into ra1 and dec1 positions
        thisDate  = stokesI.header['DATE'].split('T')[0]
        thisDate  = datetime.strptime(thisDate, '%Y-%m-%d')
        yr2000    = datetime(2000,1,1)
        secPerYr  = 365.25*24*60*60
        deltaTime = ((thisDate - yr2000).total_seconds())/secPerYr
        pmRA = starCatalog1['pmRA'].data.data*(1e-3)*u.arcsec
        pmDE = starCatalog1['pmRA'].data.data*(1e-3)*u.arcsec
        ra   = ra1 + pmRA*deltaTime
        dec  = dec1 + pmDE*deltaTime

        # Compute the final star pixel positions
        xStars, yStars = thisWCS.wcs_world2pix(ra, dec, 0)

        # Initalize a list of booleans to track which stars have been selected
        selectedStars = np.zeros(len(starCatalog1), dtype=bool)

        #*************************************
        # Now prepare to plot the image
        #*************************************
        # Build the image displays
        # Start by preparing a 1x3 plotting area
        fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)

        # Populate each axis with its image
        fig, ax, axim = stokesI.show(vmin=thisMin, vmax=thisMax,
            axes=ax, cmap='gray_r', noShow=True)
        ax.set_title('Epoch {0}'.format(iEpoch+1))

        # Rescale the figure and setup the spacing between images
        fig.set_size_inches(8, 8, forward=True)
        ax.autoscale(False)

        # Plot a list of the kept USNO-B1.0 stars
        # artCollect = ax.scatter(xStars, yStars,
        #     marker='o', color='None',  edgecolors='red')
        artCollect = [plt.Circle([xs, ys], radius=apRad,
            facecolor='None', edgecolor='red')
            for xs, ys in zip(xStars, yStars)]

        for artist in artCollect:
            ax.add_patch(artist)

            # Maybe use this instead?
            # ax.add_artist(artist)

        # Connect the event manager...
        cid1 = fig.canvas.mpl_connect('button_press_event', on_click)
        cid2 = fig.canvas.mpl_connect('key_press_event', on_key)

        # NOW show the image (without continuing execution)
        # plt.ion()
        plt.show()
        # plt.ioff()
        # pdb.set_trace()

        # Check if ANY stars were selected
        if np.sum(selectedStars) > 0:
            # Grab the indices of the selected stars
            tmpSelectedInds = (np.where(selectedStars))[0]
            selectedInds    = list(tmpSelectedInds)

            # Loop through each of these selected stars and check if any have
            # nearby neighbors which might need to be disambiguated
            for starNum, ind in enumerate(tmpSelectedInds):
                # Grab the position of the current star
                xs, ys = xStars[ind], yStars[ind]

                # Compute the distance to the other stars
                dist     = np.sqrt((xStars - xs)**2 + (yStars - ys)**2)

                # Test for stars *CLOSER* than apRad and *FARTHER* than 0 pixels
                nearBool = np.logical_and(dist > 0, dist < apRad)

                # Check if the nearest star is within estimated aperture radius
                numNearStars = np.sum(nearBool)
                if numNearStars > 1:
                    # If there are multiple nearby stars, kill it!
                    print('Skipping star {0}: too many neigbors'.format(starNum))

                    # Remove this index from the final list.
                    # (Use -1 as placeholder for deletion)
                    selectedInds[starNum] = -1

                elif numNearStars == 1:
                    # # If there is a unique nearby star, print both and let the
                    # # user choose.
                    # # Grab the index of that star
                    nearInd = int((np.where(nearBool))[0])
                    #
                    # # Grab the magnitudes to print
                    # # Star 1
                    # star1Row = starCatalog1[ind]
                    # B1_1 = star1Row['B1mag']
                    # R1_1 = star1Row['R1mag']
                    # B2_1 = star1Row['B2mag']
                    # R2_1 = star1Row['R1mag']
                    #
                    # # Star 2
                    # star2Row = starCatalog1[nearInd]
                    # B1_2 = star2Row['B1mag']
                    # R1_2 = star2Row['R1mag']
                    # B2_2 = star2Row['B2mag']
                    # R2_2 = star2Row['R1mag']
                    #
                    # # Print the results
                    # print('\tB1\tR1\tB2\tR2\t')
                    # print('#1\t{0}\t{1}\t{2}\t{3}'.format(
                    #     repr(B1_1)[0:5],
                    #     repr(R1_1)[0:5],
                    #     repr(B2_1)[0:5],
                    #     repr(R2_1)[0:5]))
                    # print('#2\t{0}\t{1}\t{2}\t{3}'.format(
                    #     repr(B1_2)[0:5],
                    #     repr(R1_2)[0:5],
                    #     repr(B2_2)[0:5],
                    #     repr(R2_2)[0:5]))
                    #
                    # done = False
                    # while not done:
                    #     print('Enter "0" to use *neither* entry)')
                    #     selection = input('Which entry should be used? ')
                    #
                    #     if selection == str(0):
                    #         # Delete the original entry from the "selectedInds"
                    #         selectedInds[starNum] = -1
                    #         done = True
                    #     elif selection == str(1):
                    #         # Exit loop without swaping index
                    #         done = True
                    #     elif selection == str(2):
                    #         # Swap one entry for the other.
                    #         selectedInds[starNum] = nearInd
                    #         done = True
                    #     else:
                    #         print('Input not understood.')


                    # Just kidding... add both and let MCMC figure it out.
                    selectedInds.append(nearInd)
                else:
                    # No problems here... continue as normal.
                    pass

            # Now that the stars for this object-epoch pair have been selected,
            # save the results in the finalKeepStars list.
            finalKeepInds.extend(keepInds[selectedInds])

    # Now that all epochs of the current object have been treated, get the
    # indices which passed ANY of the epoch tests.
    finalKeepInds = np.unique(finalKeepInds)

    if len(finalKeepInds) > 0:
        print('Saving final catalog listing to disk')
        outTable = starCatalog[finalKeepInds]
        outTable.write(outFile, format='ascii.csv')
    else:
        print('Could not find ANY acceptable stars?!')
        continue

# When all is said and done, close the figure and alert the user
plt.close('all')
print('Done!')
