# -*- coding: utf-8 -*-
"""
Displays the final Stokes I image and asks the user to select which APASS stars
will be used to calibrate the photometry.
"""

# Core imports
import os
import sys
import glob
import subprocess
import warnings
from datetime import datetime

# Scipy/numpy imports
import numpy as np
import scipy.odr as odr

# Astropy imports
from astropy.table import Table, Column, hstack
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma
from astropy import units as u
from astropy.coordinates import SkyCoord
from photutils import (daofind, aperture_photometry, CircularAperture,
    CircularAnnulus, data_properties)
from astropy.modeling import models, fitting
from astroquery.vizier import Vizier

# Plotting imports
from matplotlib import pyplot as plt
import corner

# Add the AstroImage class
import astroimage as ai

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data\\201612\\'

# Setup new directory for polarimetry data
polarimetryDir = os.path.join(pyPol_data, 'Polarimetry')
if (not os.path.isdir(polarimetryDir)):
    raise ValueError('Could not find polarimetry directory')

stokesDir = os.path.join(polarimetryDir, 'stokesImgs')
if (not os.path.isdir(stokesDir)):
    raise ValueError('Could not find Stokes image directory')

# Import the utility functions to be used here...
from utils_08b import *

# Build a dictionary contaning references to these transformation functions
photometricTransforms = dict(zip(['V',     'R'    , 'V-R'],
                           [APASS_V, APASS_R, APASS_VR]))


# #******************************************************************************
# # Define the event handlers for clicking and keying on the image display
# #******************************************************************************
#
# def on_click(event):
#     global fig, artCollect, selectedStars, outFile, xStars, yStars
#
#     x, y = event.xdata, event.ydata
#
#     # Compute star distances from the click and update mask array
#     dist = np.sqrt((xStars - x)**2 + (yStars - y)**2)
#
#     # Toggle the selected value of the nearest star
#     thisStar = np.where(dist == np.min(dist))
#     selectedStars[thisStar] = not selectedStars[thisStar]
#
#     # Build an array of colors to match which stars have been selected
#     colors = ['green' if t else 'red' for t in selectedStars]
#
#     # Update the plot to reflect the latest selection
#     for artist, color in zip(artCollect, colors):
#         artist.set_edgecolor(color)
#
#     # Update the display
#     fig.canvas.draw()
#
# def on_key(event):
#     global fig, artCollect, selectedStars, starCatalog
#
#     # Save the generated mask
#     if event.key == 'enter':
#         print('Finished with current epoch')
#         plt.close(fig)
#
#         # Disconnect the event manager and close the figure
#         fig.canvas.mpl_disconnect(cid1)
#         fig.canvas.mpl_disconnect(cid2)
#
#     else:
#         print('Do not recognize key command')
# #******************************************************************************
#
# #******************************************************************************
# # This is the main script that will load in file names and prepare for plotting
# #******************************************************************************
#
# # Declare global variables
# global fig, artCollect, selectedStars, starCatalog, outFile, xStars, yStars
#
# xList     = []
# yList     = []
# imgNum    = 0      # This number will be the FIRST image to be displayed center...
# brushSize = 3      # (5xbrushSize pix) is the size of the region masked
#

#******************************************************************************
# This script will retrieeve the APASS data from Vizier and allow the user
# to select stars suitable for photometry
#******************************************************************************
# Reset ROW_LIMIT property to retrieve FULL catalog from Vizier
Vizier.ROW_LIMIT = -1

# Specify which (target, filter) pairs to process
targetsToProcess = ['NGC2023', 'NGC7023', 'NGC891', 'NGC4565']
# targetsToProcess = ['NGC891', 'NGC4565', 'NGC2023', 'NGC7023']
# filtersToProcess = ['V', 'R']

# Use the following data for final calibration
# Bands and zero point flux [in Jy = 10^(-26) W /(m^2 Hz)]
# Following table from Bessl, Castelli & Plez (1998)
# Passband  Effective wavelength (microns)  Zero point (Jy)
# U	        0.366                           1790
# B         0.438                           4063
# V         0.545                           3636
# R         0.641                           3064
# I         0.798                           2416
# J         1.22                            1589
# H         1.63                            1021
# K         2.19                            640

zeroFlux = dict(zip(['U',    'B',    'V',    'R'   , 'I'   ],
           np.array([1790.0, 4063.0, 3636.0, 3064.0, 2416.0])*u.Jy))

wavelength = dict(zip(['U',   'B',   'V',   'R'  , 'I'  ],
             np.array([0.366, 0.438, 0.545, 0.798, 0.798])*u.micron))

# Following table from Hu (2011)
# Data from Gaomeigu Observational Station
# Passband | K'(lambda) [mag/airmass] | K'' [mag/(color*airmass)]
# U	         0.560 +/- 0.023            0.061 +/- 0.004
# B          0.336 +/- 0.021            0.012 +/- 0.003
# V          0.198 +/- 0.024           -0.015 +/- 0.004
# R          0.142 +/- 0.021           -0.067 +/- 0.005
# I          0.093 +/- 0.020            0.023 +/- 0.006


# Following table from Schmude (1994)
# Data from Texas A & M University Observatory
# Passband | K(lambda) [mag/airmass] | dispersion on K(lambda)
# U	         0.60 +/- 0.05             0.120
# B          0.40 +/- 0.06             0.165
# V          0.26 +/- 0.03             0.084
# R          0.19 +/- 0.03             0.068
# I          0.16 +/- 0.02             0.055

kappa = dict(zip(['U',    'B',    'V',    'R'   ],
                 [0.60,   0.40,   0.26,   0.19  ]))

# Loop through each target
for thisTarget in targetsToProcess:
    # Update the user on progress
    print('Processing files for')
    print('Target : {}'.format(thisTarget))

    # Search for all Stokes Intensity images
    Ifile            = os.path.join(stokesDir, thisTarget + '*I.fits')
    Ifiles           = glob.glob(Ifile)
    stokesIimgs      = [ai.reduced.ReducedScience.read(file1) for file1 in Ifiles]
    filtersToProcess = [img.header['FILTNME3'].strip() for img in stokesIimgs]

    # Compose a dictionary of stokes I, U, and Q images
    stokesIdict      = dict(zip(filtersToProcess, stokesIimgs))

    for thisFilter in filtersToProcess:
        # Update the user on progress
        print('Filter : {}'.format(thisFilter))

        # Construct the name of the output file
        outFile = os.path.join(stokesDir, thisTarget + '_stars.csv')

        # TODO: generalize this for any bands and any number of bands
        # Read in the Stokes intensity image
        try:
            I_V_file  = os.path.join(stokesDir,
                     '_'.join([thisTarget, 'V', 'I']) + '.fits')
            stokesI_V = ai.reduced.ReducedScience.read(I_V_file)
            Vexists = True
        except:
            Vexists = False
        try:
            I_R_file  = os.path.join(stokesDir,
                     '_'.join([thisTarget, 'R', 'I']) + '.fits')
            stokesI_R = ai.reduced.ReducedScience.read(I_R_file)
            Rexists = True
        except:
            Rexists = False

        # Check that both bands are present
        if not (Vexists and Rexists):
            raise ValueError('Could not find both optical bands.')

        #
        # Don't need to compute alignment because they're already perfectly
        # aligned!
        #
        # # Compute image alignment offsets
        # imgStack = ai.utilitywrappers.ImageStack([stokesI_V, stokesI_R])
        # dx, dy   = imgStack.get_wcs_offsets()\

        # Grab the image center pixel coordinates
        ny, nx = stokesI_V.shape
        yc, xc = ny/2.0, nx/2.0

        # Compute image center WCS coordinates
        cenRA, cenDec = stokesI_V.wcs.wcs_pix2world(xc, yc, 0, ra_dec_order=True)
        cenCoord      = SkyCoord(cenRA, cenDec, unit='deg', frame='fk5')

        # Compute image width and height
        width, height = stokesI_V.pixel_scales*np.array(stokesI_V.shape[::-1])*u.pix

        # Download the APASS catalog for this region
        # For some
        starCatalog = Vizier.query_region(cenCoord,
            width = height, height = width, catalog='APASS')

        # Test if only one catalog was returned
        if len(starCatalog) == 1:
            starCatalog = starCatalog[0]
        else:
            ValueError('Catalog confusion. More than one result found!')

        # Compute image to be displayed
        combinedImage = stokesI_V + stokesI_R

        # Grab the image axes and overplot the stars
        sx, sy = combinedImage.wcs.wcs_world2pix(
            starCatalog['_RAJ2000'],
            starCatalog['_DEJ2000'],
            0)

        # Compute the nearest neighbor distances
        separations = []
        for sx1, sy1 in zip(sx, sy):
            distances = np.sqrt((sx - sx1)**2 + (sy - sy1)**2)
            distances.sort()
            separations.append(distances[1])

        # Test for saturations
        maxVals = []
        for sx1, sy1 in zip(sx, sy):
            bt, tp = np.int(np.round(sy1 - 10)), np.int(np.round(sy1 + 10))
            lf, rt = np.int(np.round(sx1 - 10)), np.int(np.round(sx1 + 10))
            maxVals.append(np.max([
                stokesI_V.data[bt:tp, lf:rt].max(),
                stokesI_R.data[bt:tp, lf:rt].max()
            ]))

        # Find stars far from the image edges
        ny, nx = combinedImage.shape
        minDistFromEdge = 50
        farFromEdge = (sx > minDistFromEdge)
        farFromEdge = np.logical_and(farFromEdge, (sx < nx - minDistFromEdge - 1))
        farFromEdge = np.logical_and(farFromEdge, (sy > minDistFromEdge))
        farFromEdge = np.logical_and(farFromEdge, (sy < ny - minDistFromEdge - 1))

        # Find isolated stars, good for photometry
        goodStars = np.array(separations) > 15
        goodStars = np.logical_and(goodStars, np.array(maxVals) < 1.6e4)
        goodStars = np.logical_and(goodStars, farFromEdge)
        goodInds  = np.where(goodStars)

        # Cull to only include the good stars
        starCatalog = starCatalog[goodInds]
        sx , sy     = sx[goodInds], sy[goodInds]

        # Refine star positions using the photutils.data_properties function
        sxRefined = []
        syRefined = []
        for sx1, sy1 in zip(sx, sy):
            cutout           = combinedImage.data[sy1-10:sy1+10,sx1-10:sx1+10]
            cutoutProperties = data_properties(cutout)

            sxRefined.append(sx1 + cutoutProperties.xcentroid.value - 11)
            syRefined.append(sy1 + cutoutProperties.ycentroid.value - 11)

        # Replaced the estimated positions with the refined positions
        sx, sy = sxRefined, syRefined

        # Store the star positions in a single array
        xyStars     = np.array([sx, sy]).T

        # Analyze the PSF of the images
        photAnalyzer_V = ai.utilitywrappers.PhotometryAnalyzer(stokesI_V)
        photAnalyzer_R = ai.utilitywrappers.PhotometryAnalyzer(stokesI_R)
        _, psf_V       = photAnalyzer_V.get_psf()
        _, psf_R       = photAnalyzer_R.get_psf()
        FWHMs          = [np.sqrt(psf_V['smajor']*psf_V['sminor']),
                          np.sqrt(psf_R['smajor']*psf_R['sminor'])]
        apRad          = 3*np.max(FWHMs)
        anInRad        = apRad + 2.0
        anOutRad       = apRad + 4.0

        # Establish circular apertures for photometry
        apertures = CircularAperture(xyStars, r = apRad)
        annulus_apertures = CircularAnnulus(xyStars,
            r_in = anInRad, r_out = anOutRad)

        # Initalize a dictionary to store the photometry tables
        photDict = {}
        for key, img1 in zip(['V', 'R'], [stokesI_V, stokesI_R]):
            # Perform the basic photometry
            rawflux_table = aperture_photometry(img1.data, apertures,
                error=img1.uncertainty)
            bkgflux_table = aperture_photometry(img1.data, annulus_apertures,
                error=img1.uncertainty)
            phot_table = hstack([rawflux_table, bkgflux_table],
                table_names=['raw', 'bkg'])

            # Compute background contribution and subtract from raw photometry
            bkg_mean = phot_table['aperture_sum_bkg'] / annulus_apertures.area()
            bkg_sig  = phot_table['aperture_sum_err_bkg'] / annulus_apertures.area()

            bkg_sum = bkg_mean * apertures.area()
            bkg_sig = bkg_sig * apertures.area()

            # Compute the variance in the background pixels for each star
            ny, nx  = img1.shape
            yy, xx  = np.mgrid[0:ny, 0:nx]
            bkg_var = []

            # Loop through each star and add the local variance to the uncertainty
            for xy in xyStars:
                xs, ys = xy
                distFromStar = np.sqrt((xx - xs)**2 + (yy - ys)**2)
                skyPixInds   = np.where(np.logical_and(
                    (distFromStar > anInRad), (distFromStar < anOutRad)))
                bkg_var.append(np.var(img1.data[skyPixInds]))

            # Convert the background variance into an array
            bkg_var = np.array(bkg_var)

            # Compute the final photometry and its uncertainty
            final_sum = phot_table['aperture_sum_raw'] - bkg_sum
            final_sig = np.sqrt(phot_table['aperture_sum_err_raw']**2
                + bkg_sig**2
                + bkg_var)
            phot_table['residual_aperture_sum'] = final_sum
            phot_table['residual_aperture_sum_err'] = final_sig

            # Compute the signal-to-noise ratio and find the stars with SNR < 3.0
            SNR = final_sum/final_sig

            # Now estimate the photometry from USNO-B1.0 and store it for later use
            catMags, sigCatMags = photometricTransforms[key](starCatalog)
            phot_table[key+'_catMag']    = catMags
            phot_table[key+'_sigCatMag'] = sigCatMags

            # Loop through all the stars and detect any duplicate entries. Mark each
            # entry with a semi-unique 'Star ID'
            # Extract the star positions from the photometry table
            # (this is redundant but a nice confirmation that these will be right)
            xStars = phot_table['xcenter_raw']
            yStars = phot_table['ycenter_raw']

            # Initalize an empty list to store the starIDs
            starIDs = -1*np.ones(len(phot_table), dtype=int)
            for ind, row in enumerate(phot_table):
                # Skip over any rows that have been previously treated
                if starIDs[ind] > 0: continue

                # Compute the distance between the current star and all other stars
                xs, ys = row['xcenter_raw'], row['ycenter_raw']
                dist   = np.sqrt((xs - xStars)**2 + (ys - yStars)**2).value

                if np.sum(dist < 2.0) > 0:
                    # Mark all stars within 2.0 pixels of the current star with an
                    # identical ID.
                    IDinds = np.where(dist < 2.0)
                    starIDs[IDinds] = ind

            # Add the StarID column to the phot_table
            phot_table.add_column(Column(name='star_id', data=starIDs), index=0)

            # plt.ion()
            # plt.imshow(stokesIdict[key].data, vmin=0,vmax=800,cmap='gray_r')
            # apertures.plot(color='r')
            # plt.scatter(phot_table['xcenter_raw'], phot_table['ycenter_raw'],
            #     marker='x', color='red')
            # import pdb; pdb.set_trace()

            # Sort the phot_table by starID
            sortInds   = phot_table['star_id'].data.argsort()
            phot_table = phot_table[sortInds]

            # Store this photometry table in the dictionary for later use
            photDict[key] = phot_table

    # I need to simultaneously solve a set of linear regressions for photometric
    # zero-point magnitudes and color correction terms
    #
    # E.g.
    # (V_corrected - V_apparent) = a_0 + a_1 * (V_apparent - R_apparent)
    # and
    # (R_corrected - R_apparent) = a_2 + a_3 * (V_apparent - R_apparent)
    # and
    # (V_corrected - R_corrected) = a_4 + a_5 * (V_apparent - R_apparent)
    #

    # Grab all the successfully measured bandpasses
    bandKeys1 = [key for key in stokesIdict.keys()]

    # Ensure that they're in wavelength order
    # Start by constructing an array with
    # Column 0: list of wavebands
    # Column 1: list of wavelengths for that bands
    bandLamArr = np.array([[key, val] for key, val in wavelength.items()])

    # Figure our how to sort this array by increasing wavelength, and create a
    # list of possible wavebands in that sorted order
    sortArr  = bandLamArr[:,1].argsort()
    bandList = (bandLamArr[:,0].flatten())[sortArr]

    # Loop through the wavebands and construct a wavelength ordered list of
    # observed waveband keys in the stokesIdict dictionary.
    bandKeys = []
    for band in bandList:
        if band in bandKeys1:
            bandKeys.append(band)

    # Loop through the bands and construct keys for a "color dictionary"
    colorKeys = []
    for ind, band1 in enumerate(bandKeys[0:len(bandKeys)-1]):
        # Only worry about colors from adjacent wavebands, one index over
        band2 = bandKeys[ind+1]
        colorKeys.append('-'.join([band1, band2]))

    # Prepare for the linear regressions to be done on each band and color
    # Define the model to be used in the fitting
    def lineFunc(B, x):
        return B[1] + B[0]*x

    # Set up ODR with the model and data.
    lineModel = odr.Model(lineFunc)

    # loop through each linear regression
    for colorKey in colorKeys:
        print('Preparing the model outliers with MCMC')

        # Setup the walker count, burn-in steps, and production steps
        n_walkers       = 100
        n_burn_in_steps = 1000
        n_steps         = 2000

        # Treat the photometric regressions for this set of bands...
        # Establish the boundaries of acceptable parameters for the prior
        labels = [
            r"$\theta$",
            r"$b_p$"]

        # Create a separate set of labels and indices for those parameters which
        # will be plotted in the posterior distribution "corner plot"
        plotLabels = [
            r"$\theta$",
            r"$b_p$"
        ]
        plotInds = np.array([0,1])

        bounds1 = [(-1.0, 1.0),     # Theta (angle of the line slope)
                   (24.0, 30.0)     # b_perp (min-dist(line-origin))
                   ]

        bounds2 = [(+0.0, +1.5),    # Theta (angle of the line slope)
                   (18.0, 28.0),    # b_perp (min-dist(line-origin))
                   (0.0, 1.0),      # Pb (Probability of sampling an outliers)
                   (-8.0, +8.0),    # Mx (<x> of outlier distribution)
                   (-2.0, 5.0),     # lnVx (log-x-variance of outlier distribution)
                   (-8.0, +8.0),    # My (<y> of outlier distribution)
                   (-2.0, 5.0)]     # lnVy (log-y-variance of outlier distribution)

        boundsC = [(-0.5, +1.0),    # Theta (angle of the line slope)
                   (-0.4, +0.75),   # b_perp (min-dist(line-origin))
                   (0.0, 1.0),      # Pb (Probability of sampling an outliers)
                   (-8.0, +8.0),    # Mx (<x> of outlier distribution)
                   (-2.0, 5.0),     # lnVx (log-x-variance of outlier distribution)
                   (-8.0, +8.0),    # My (<y> of outlier distribution)
                   (-2.0, 5.0)]     # lnVy (log-y-variance of outlier distribution)

        # Parse the bands used in this color
        band1, band2 = colorKey.split('-')

        # Grab the photometry table for these two bands
        phot_table1 = photDict[band1]
        phot_table2 = photDict[band2]

        # Double check that the star IDs are all matched up
        if len(phot_table1) != len(phot_table2):
            raise RuntimeError('Photometry tables do not match!')

        totalMatch = np.sum(phot_table1['star_id'].data == phot_table2['star_id'].data)
        if totalMatch < len(phot_table1):
            raise RuntimeError('Photometry tables do not match!')

        # Since we have confirmed that all the starIDs match up, we will store
        # the values from the first phot_table
        starIDs = phot_table['star_id'].data

        # Grab the fluxes for the calibration stars for these two bands
        flux1    = phot_table1['residual_aperture_sum'].data
        flux2    = phot_table2['residual_aperture_sum'].data
        sigFlux1 = phot_table1['residual_aperture_sum_err'].data
        sigFlux2 = phot_table2['residual_aperture_sum_err'].data

        # Compute the instrumental magnitudes for these two bands
        instMags1    = -2.5*np.log10(flux1)
        instMags2    = -2.5*np.log10(flux2)
        sigInstMags1 = 2.5*np.abs(sigFlux1/(flux1*np.log(10)))
        sigInstMags2 = 2.5*np.abs(sigFlux2/(flux2*np.log(10)))

        # Now grab the catalog magnitudes for the calibration stars
        catMags1    = phot_table1[band1+'_catMag'].data
        catMags2    = phot_table2[band2+'_catMag'].data
        sigCatMags1 = phot_table1[band1+'_sigCatMag'].data
        sigCatMags2 = phot_table2[band2+'_sigCatMag'].data

        # Begin by culling any data from extremely unexpected regions
        # Compute the catalog colors for these stars
        catColors, sig_catColors = photometricTransforms[colorKey](starCatalog)

        ########################################################################
        ################################ BAND 1 ################################
        ########################################################################
        print('Running {0}-band regression'.format(band1))
        x1  = instMags1 - instMags2
        y1  = catMags1 - instMags1
        sx1 = np.sqrt(sigInstMags1**2 + sigInstMags2**2)
        sy1 = np.sqrt(sigCatMags1**2 + sigInstMags1**2)

        # Do a double check that all the data have finite values
        keepBool = np.isfinite(x1)
        keepBool = np.logical_and(keepBool, np.isfinite(y1))
        keepBool = np.logical_and(keepBool, np.isfinite(sx1))
        keepBool = np.logical_and(keepBool, np.isfinite(sy1))
        keepInds = np.where(keepBool)

        # Cull the temporary variables to only include good data
        x1  = x1[keepInds]
        y1  = y1[keepInds]
        sx1 = sx1[keepInds]
        sy1 = sy1[keepInds]

        # Perform the MCMC sampling of the posterior
        data = (x1, y1, sx1, sy1)
        sampler1 = MCMCfunc(data, bounds1,
            n_walkers=n_walkers,
            n_burn_in_steps=n_burn_in_steps,
            n_steps=n_steps)

        # Plot the posteriors to see if a reasonable result was obtained.
        plt.ion()
        plotSamples = sampler1.flatchain[:,plotInds]
        plotBounds  = np.array(bounds1)[plotInds]
        corner.corner(plotSamples, bins=100,
            range=plotBounds,
            labels=plotLabels)

        # Setup the orthogonal-distance-regression (ODR) for band 1
        data1 = odr.RealData(x1, y1, sx=sx1, sy=sy1)
        odr1  = odr.ODR(data1, lineModel, beta0=[0.0, np.median(y1)])
        out1  = odr1.run()

        plt.ion()
        plt.figure()
        plt.errorbar(x1, y1, xerr=sx1, yerr=sy1, linestyle='none')
        plt.scatter(x1, y1)
        xlims = np.array([0, 1.2])
        plt.plot(xlims, xlims*out1.beta[0] + out1.beta[1], color='b')

        out1b = np.median(plotSamples, axis=0)
        out1b[1] = np.max(plotSamples[:,1])
        plt.plot(xlims, xlims*out1b[0] + out1b[1], color='r')
        import pdb; pdb.set_trace()
        plt.close('all')

        # Compute the posterior probability that each data-point is "good"
        norm = 0.0
        post_prob1 = np.zeros(len(data[0]))
        for i in range(sampler1.chain.shape[1]):
            for j in range(sampler1.chain.shape[0]):
                ll_fg, ll_bg = sampler1.blobs[i][j]
                post_prob1 += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
                norm += 1
        post_prob1 /= norm

        # Track the outliers from the band-1 MCMC fit
        keepBool = (post_prob1 > 0.5)

        import pdb; pdb.set_trace()

        ########################################################################
        ################################ BAND 2 ################################
        ########################################################################
        print('Running {0}-band regression'.format(band2))
        x2  = instMags1 - instMags2
        y2  = catMags2 - instMags2
        sx2 = np.sqrt(sigInstMags1**2 + sigInstMags2**2)
        sy2 = np.sqrt(sigCatMags2**2 + sigInstMags2**2)

        # Perform the MCMC sampling of the posterior
        data = (x2, y2, sx2, sy2)
        sampler2 = MCMCfunc(data, bounds2,
            n_walkers=n_walkers,
            n_burn_in_steps=n_burn_in_steps,
            n_steps=n_steps)

        # # Plot the posteriors to see if a reasonable result was obtained.
        # plotSamples = sampler2.flatchain[:,plotInds]
        # plotBounds  = np.array(bounds1)[plotInds]
        # corner.corner(plotSamples, bins=100,
        #     range=plotBounds,
        #     labels=plotLabels)
        # pdb.set_trace()
        # plt.close('all')

        # Compute the posterior probability that each data-point is "good"
        norm = 0.0
        post_prob2 = np.zeros(len(data[0]))
        for i in range(sampler2.chain.shape[1]):
            for j in range(sampler2.chain.shape[0]):
                ll_fg, ll_bg = sampler2.blobs[i][j]
                post_prob2 += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
                norm += 1
        post_prob2 /= norm

        # Track the outliers from the band-2 MCMC fit
        keepBool = np.logical_and(keepBool, (post_prob2 > 0.5))

        ########################################################################
        ############################# COLOR-COLOR ##############################
        ########################################################################
        # Begin by culling any data marked as outliers in band1 or band2 MCMC.
        keepInds = np.where(keepBool)

        # Now cull the data and perform ODR fits
        print('Culling probable outlier data')
        starIDs       = starIDs[keepInds]
        instMags1     = instMags1[keepInds]
        instMags2     = instMags2[keepInds]
        sigInstMags1  = sigInstMags1[keepInds]
        sigInstMags2  = sigInstMags2[keepInds]
        catMags1      = catMags1[keepInds]
        catMags2      = catMags2[keepInds]
        sigCatMags1   = sigCatMags1[keepInds]
        sigCatMags2   = sigCatMags2[keepInds]
        catColors     = catColors[keepInds]
        sig_catColors = sig_catColors[keepInds]

        # Make sure to identically cull the posterior probabilities too!
        post_prob1 = post_prob1[keepInds]
        post_prob2 = post_prob2[keepInds]

        print('Running final color-color regression')
        # Compute the colors for these stars
        xC  = instMags1 - instMags2
        yC  = catColors
        sxC = np.sqrt(sigInstMags1**2 + sigInstMags2**2)
        syC = sig_catColors

        # Perform the MCMC sampling of the posterior
        data = (xC, yC, sxC, syC)
        samplerC = MCMCfunc(data, boundsC,
            n_walkers=n_walkers,
            n_burn_in_steps=n_burn_in_steps,
            n_steps=n_steps)

        # Compute the posterior probability that each data-point is "good"
        norm = 0.0
        post_probC = np.zeros(len(data[0]))
        for i in range(samplerC.chain.shape[1]):
            for j in range(samplerC.chain.shape[0]):
                ll_fg, ll_bg = samplerC.blobs[i][j]
                post_probC += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
                norm += 1
        post_probC /= norm

        # # Track the outliers from the color-color MCMC fit
        # keepBool = np.logical_and(keepBool, (post_probC > 0.5))
        if np.sum(post_probC < 0.5) > 0:
            print('Color-Color fit still has some outliers?!')
            pdb.set_trace()

        # Grab the "confidence intervals" for each parameter
        truthRanges = [(v[1], v[2]-v[1], v[1]-v[0]) for v in
            zip(*np.percentile(samplerC.flatchain, [16, 50, 84], axis=0))]
        truths = [t[0] for t in truthRanges]

        # Convert these data to slope-intercept space
        tmpData = np.array([
            np.tan(samplerC.flatchain[:,0]),
            samplerC.flatchain[:,1]/np.cos(samplerC.flatchain[:,0])])

        # Compute the median slope and intercept and the covariance matrix
        truthsC = np.percentile(tmpData, 50, axis=1)
        covC    = np.cov(tmpData)

        ########################################################################
        ############################# FINAL PLOTS ##############################
        ########################################################################
        ### BAND 1 ###
        # Compute magnitudes and colors
        x1  = instMags1 - instMags2
        y1  = catMags1 - instMags1
        sx1 = np.sqrt(sigInstMags1**2 + sigInstMags2**2)
        sy1 = np.sqrt(sigCatMags1**2 + sigInstMags1**2)

        # Setup the orthogonal-distance-regression (ODR) for band 1
        data1 = odr.RealData(x1, y1, sx=sx1, sy=sy1)
        odr1  = odr.ODR(data1, lineModel, beta0=[0.0, np.median(y1)])
        out1  = odr1.run()

        # Plot the fitted data with its errorbars
        plt.errorbar(x1, y1, xerr=sx1, yerr=sy1,
            fmt='None', ecolor='k', zorder=999)
        plt.scatter(x1, y1, marker='s',
            c=post_prob1, cmap="gray_r", vmin=0, vmax=1, zorder=1000)
        # plt.scatter(x1[badInds1], y1[badInds1], marker='x', c='r', zorder=10001)

        # Overplot the fitted line
        xl = np.array(plt.xlim())
        if np.min(xl) > 0:
            xl = np.array([0, xl[1]])
            plt.xlim(xl)
        plt.autoscale(False)
        slope, intercept = out1.beta
        plt.plot(xl, slope*xl + intercept, c='k', lw=2.0)
        plt.xlabel(r'$V_{\rm inst}-R_{\rm inst}$')
        plt.ylabel(r'$V_{\rm cat}-V_{\rm inst}$')

        # Save the figure to disk
        fname = os.path.join(stokesDir, thisTarget + '_regression1.png')
        plt.savefig(fname, dpi=300)
        plt.close('all')

        ### BAND 2 ###
        # Compute magnitudes and colors
        x2  = instMags1 - instMags2
        y2  = catMags2 - instMags2
        sx2 = np.sqrt(sigInstMags1**2 + sigInstMags2**2)
        sy2 = np.sqrt(sigCatMags2**2 + sigInstMags2**2)

        # Setup the orthogonal-distance-regression (ODR) for band 2
        data2 = odr.RealData(x2, y2, sx=sx2, sy=sy2)
        odr2  = odr.ODR(data2, lineModel, beta0=[0.0, np.median(y2)])
        out2  = odr2.run()

        # Plot the fitted data with its errorbars
        plt.errorbar(x2, y2, xerr=sx2, yerr=sy2,
           fmt='None', ecolor='k', zorder=999)
        plt.scatter(x2, y2, marker='s',
            c=post_prob2, cmap="gray_r", vmin=0, vmax=1, zorder=1000)
        # plt.scatter(x2[badInds2], y2[badInds2], marker='x', c='r', zorder=10001)

        # Overplot the fitted line
        xl = np.array(plt.xlim())
        if np.min(xl) > 0:
            xl = np.array([0, xl[1]])
            plt.xlim(xl)
        plt.autoscale(False)
        slope, intercept = out2.beta
        plt.plot(xl, slope*xl + intercept, c='k', lw=2.0)
        plt.xlabel(r'$V_{\rm inst}-R_{\rm inst}$')
        plt.ylabel(r'$R_{\rm cat}-R_{\rm inst}$')

        # Save the figure to disk
        fname = os.path.join(stokesDir, thisTarget + '_regression2.png')
        plt.savefig(fname, dpi=300)
        plt.close('all')

        ### COLOR-COLOR ###
        # Plot the posteriors to see if a reasonable result was obtained.
        plotSamples = samplerC.flatchain[:,plotInds]
        plotBounds  = np.array(boundsC)[plotInds]
        corner.corner(plotSamples, bins=100,
            range=plotBounds,
            labels=plotLabels)

        # Save the figure to disk
        fname = os.path.join(stokesDir, thisTarget + '_MCMC.png')
        plt.savefig(fname, dpi=300)
        plt.close('all')

        # Setup the orthogonal-distance-regression (ODR) for band 1
        dataC = odr.RealData(xC, yC, sx=sxC, sy=syC)
        odrC  = odr.ODR(dataC, lineModel, beta0=[0.0, np.median(yC)])
        outC  = odrC.run()

        # Plot the fitted data with its errorbars
        plt.errorbar(xC, yC, xerr=sxC, yerr=syC,
            fmt='None', ecolor='k', zorder=999)
        plt.scatter(xC, yC, marker='s',
            c=post_probC, cmap="gray_r", vmin=0, vmax=1, zorder=1000)
        # plt.scatter(xC[badIndsC], yC[badIndsC], marker='x', c='r', zorder=10001)

        # Grab (and set) the x-axis boundaries
        xl = np.array(plt.xlim())
        if np.min(xl) > 0:
            xl = np.array([0, xl[1]])
            plt.xlim(xl)
        plt.autoscale(False)

        # Plot a sampling of acceptable MCMC parameters
        samples = samplerC.flatchain[:,0:2]
        for theta, b_perp in samples[np.random.randint(len(samples), size=1000)]:
            m, b = np.tan(theta), b_perp/np.cos(theta)
            plt.plot(xl, m*xl + b, color="k", alpha=0.025)
        slope, intercept = np.tan(truths[0]), truths[1]/np.cos(truths[0])
        plt.plot(xl, slope*xl + intercept,
            color='blue')

        # Overplot the ODR fit
        slope1, intercept1 = outC.beta
        plt.plot(xl, slope1*xl + intercept1,
            color='red', linestyle='dashed', linewidth=2.0)
        plt.xlabel(r'$V_{\rm inst}-R_{\rm inst}$')
        plt.ylabel(r'$V_{\rm cat}-R_{\rm cat}$')

        # Save the figure to disk
        fname = os.path.join(stokesDir, thisTarget + '_colorRegression.png')
        plt.savefig(fname, dpi=300)
        plt.close('all')


        ########################################################################
        print('...Photometric and Color transformations...\n')
        print(('\t({0}cal - {0}inst) = {3:.4g} + {2:.4g}*({0}inst - {1}inst)').format(
            band1, band2, *out1.beta))
        print(('\t({1}cal - {1}inst) = {3:.4g} + {2:.4g}*({0}inst - {1}inst)').format(
            band1, band2, *out2.beta))
        print(('\t({0}cal - {1}cal)  = {3:.4g} + {2:.4g}*({0}inst - {1}inst)').format(
            band1, band2, *truthsC))

        # Compute the photometrically calibrated images
        # Frist grab the airmass corrected images from the stokesIdict_AMC
        img1 = stokesIdict_AMC[band1]
        img2 = stokesIdict_AMC[band2]

        # Grab the U and Q images, too...
        Qimg1 = stokesQdict[band1]
        Qimg2 = stokesQdict[band2]
        Uimg1 = stokesUdict[band1]
        Uimg2 = stokesUdict[band2]

        # Compute the flux ratio, fix negative values, and convert to colors
        fluxRatio = img1/img2

        # First fix the indefinite values
        badPix = np.logical_not(np.isfinite(fluxRatio.arr))
        if np.sum(badPix) > 0:
            badInds = np.where(badPix)
            fluxRatio.arr[badInds] = 1e-6
            if hasattr(fluxRatio, 'uncertainty'):
                fluxRatio.sigma[badInds] = 1e-6

        # Then fix the negative values
        badPix = (fluxRatio.arr < 1e-6)
        if np.sum(badPix) > 0:
            badInds = np.where(badPix)
            fluxRatio.arr[badInds] = 1e-6
            if hasattr(fluxRatio, 'uncertainty'):
                fluxRatio.sigma[badInds] = 1e-6

        # Compute the instrumental color map and convert it to calibrated scale
        instColor = -2.5*np.log10(fluxRatio)

        # Use covariance matrix from MCMC process to estimate uncertainty in the
        # calibrated color map
        sig_m2    = covC[0,0]
        sig_b2    = covC[1,1]
        rhosmsb   = covC[0,1]
        sig_Color = np.sqrt(sig_m2*(instColor.arr**2) + sig_b2
            + 2*rhosmsb*instColor.arr + instColor.sigma**2)

        # Compute the calibrated color image and replace the simple uncertainty
        # with the full blown uncertainty...
        calColor       = instColor.copy()
        calColor.arr   = truthsC[1] + truthsC[0]*instColor.arr
        calColor.sigma = sig_Color

        # Now that the color-map has been computed, apply the calibration
        # transformations to the individual bands
        # First, compute the correction factors based on the regression above
        CF1_1     = 10.0**(-0.4*out1.beta[1])
        sig_CF1_1 = np.abs(CF1_1*0.4*np.log(10)*out1.sd_beta[1])
        CF1_2     = (fluxRatio)**out1.beta[0]
        CF2_1     = 10.0**(-0.4*out2.beta[1])
        sig_CF2_1 = np.abs(CF2_1*0.4*np.log(10)*out2.sd_beta[1])
        CF2_2     = (fluxRatio)**out1.beta[0]

        # Apply the color term to the actual image before writing to disk
        img1a = img1*CF1_2
        img2a = img2*CF2_2

        # Grab the pixel area to include in the linear scaling constants
        wcs1     = WCS(img1.header)
        wcs2     = WCS(img2.header)

        # Store the linear scaling constants in the "BSCALE" keyword
        BSCALE1  = zeroFlux[band1]*CF1_1
        BSCALE2  = zeroFlux[band2]*CF2_1
        SBSCALE1 = zeroFlux[band1]*sig_CF1_1
        SBSCALE2 = zeroFlux[band2]*sig_CF2_1

        # Describe the scaling offset and linear units
        BUNIT = 'Jy'
        BZERO = 0.0

        # Store the calibration data in the header
        img1a.header.set('BUNIT', value = BUNIT,
                   comment='Physical units for the image')
        img1a.header.set('BSCALE', value = BSCALE1, after='BUNIT',
                   comment='Conversion factor for physical units', )
        img1a.header.set('SBSCALE', value = SBSCALE1, after='BUNIT',
                   comment='Conversion factor for physical units', )
        img1a.header.set('BZERO', value = BZERO, after='BSCALE',
                   comment='Zero level for the physical units.')
        img2a.header.set('BUNIT', value = BUNIT,
                   comment='Physical units for the image')
        img2a.header.set('BSCALE', value = BSCALE2, after='BUNIT',
                   comment='Conversion factor for physical units', )
        img2a.header.set('SBSCALE', value = SBSCALE2, after='BUNIT',
                   comment='Conversion factor for physical units', )
        img2a.header.set('BZERO', value = BZERO, after='BSCALE',
                   comment='Zero level for the physical units.')

        # Finally write the updated image to disk
        img1aCalFile = '_'.join([thisTarget, band1, 'I', 'cal']) + '.fits'
        img1aCalFile = os.path.join(stokesDir, img1aCalFile)
        img1a.write(img1aCalFile)

        img2aCalFile = '_'.join([thisTarget, band2, 'I', 'cal']) + '.fits'
        img2aCalFile = os.path.join(stokesDir, img2aCalFile)
        img2a.write(img2aCalFile)

        Qimg1CalFile = '_'.join([thisTarget, band1, 'Q', 'cal']) + '.fits'
        Qimg1CalFile = os.path.join(stokesDir, Qimg1CalFile)
        Qimg1.write(Qimg1CalFile)

        Qimg2CalFile = '_'.join([thisTarget, band2, 'Q', 'cal']) + '.fits'
        Qimg2CalFile = os.path.join(stokesDir, Qimg2CalFile)
        Qimg2.write(Qimg2CalFile)

        Uimg1CalFile = '_'.join([thisTarget, band1, 'U', 'cal']) + '.fits'
        Uimg1CalFile = os.path.join(stokesDir, Uimg1CalFile)
        Uimg1.write(Uimg1CalFile)

        Uimg2CalFile = '_'.join([thisTarget, band2, 'U', 'cal']) + '.fits'
        Uimg2CalFile = os.path.join(stokesDir, Uimg2CalFile)
        Uimg2.write(Uimg2CalFile)

        colorFile = '_'.join([thisTarget, band1+'-'+band2]) + '.fits'
        colorFile = os.path.join(stokesDir, colorFile)
        calColor.write(colorFile)

print('Done!')
