import pdb
import os
import psutil
import time
import numpy as np
from scipy import signal
from scipy import ndimage
import scipy.stats
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from photutils import daofind

class Image(object):
    """An object which stores an image array and header and provides a
    read method and an overscan correction method.
    """
    
    def __init__(self, filename=''):
        if len(filename) > 0:
            try:
                HDUlist     = fits.open(filename)
                self.header = HDUlist[0].header.copy()
                floatFlag   = self.header['BITPIX'] < 0
                numBits     = np.abs(self.header['BITPIX'])
                
                # Determine the appropriate data type for the array.
                if floatFlag:
                    if numBits >= 64:
                        dataType = np.float64
                    else:
                        dataType = np.float32
                else:
                    if numBits >= 64:
                        dataType = np.int64
                    elif numBits >= 32:
                        dataType = np.int32
                    else:
                        dataType = np.int16
                
                # Store the data as the correct data type
                self.arr      = HDUlist[0].data.astype(dataType, copy = False)

                # Check that binning makes sense and store it if it does
                uniqBins = np.unique((self.header['CRDELT1'],
                                     self.header['CRDELT2']))
                if len(uniqBins) > 1:
                    raise ValueError('Binning is different for each axis')
                else:
                    self.binning  = int(uniqBins[0])

                self.filename = filename
                self.dtype    = dataType
                HDUlist.close()

            except:
                print('File not found')
    
#        def read_file(self, filename):
    
    def __pos__(self):
        # Implements behavior for unary positive (e.g. +some_object)
        print('not yet implemented')
    
    def __neg__(self):
        # Implements behavior for negation (e.g. -some_object)
        print('not yet implemented')
    
    def __abs__(self):
        # Implements behavior for the built in abs() function.
        print('not yet implemented')
    
    def __add__(self, other):
        # Implements addition.
        bothAreImages = isinstance(other, self.__class__)
        oneIsInt      = isinstance(other, int)
        oneIsFloat    = isinstance(other, float)
        
        if bothAreImages:
            # Check that image shapes make sense
            shape1     = self.arr.shape
            shape2     = other.arr.shape
            if shape1 == shape2:
                output     = self.copy()
                output.arr = self.arr + other.arr
            else:
                print('Cannot add images with different shapes')
                return None
        elif oneIsInt or oneIsFloat:
            output     = self.copy()
            output.arr = self.arr + other
        else:
            print('Attempting to add image and something weird...')
            return None
        
        # Return the added image
        return output
    
    def __radd__(self, other):
        # Implements reverse addition.
        if other == 0:
            return self
        else:
            return self.__add__(other)
    
    def __sub__(self, other):
        # Implements subtraction.
        bothAreImages = isinstance(other, self.__class__)
        oneIsInt      = isinstance(other, int)
        oneIsFloat    = isinstance(other, float)
        
        if bothAreImages:
            # Check that image shapes make sense
            shape1     = self.arr.shape
            shape2     = other.arr.shape
            if shape1 == shape2:
                output     = self.copy()
                output.arr = self.arr - other.arr
            else:
                print('Cannot subtract images with different shapes')
                return None
        elif oneIsInt or oneIsFloat:
            output     = self.copy()
            output.arr = self.arr - other
        else:
            print('Attempting to subtract image and something weird...')
            return None
        
        # Return the subtracted image
        return output

    def __rsub__(self, other):
        # Implements reverse subtraction.
        if other == 0:
            return self
        else:
            output = self.__sub__(other)
            output.arr = -output.arr
            return output
    
    def __mul__(self, other):
        # Implements multiplication.
        bothAreImages = isinstance(other, self.__class__)
        oneIsInt      = isinstance(other, int)
        oneIsFloat    = isinstance(other, float)
        if bothAreImages:
            # Check that image shapes make sense
            shape1     = self.arr.shape
            shape2     = other.arr.shape
            if shape1 == shape2:
                output     = self.copy()
                output.arr = self.arr * other.arr
            else:
                print('Cannot subtract images with different shapes')
                return None
        elif oneIsInt or oneIsFloat:
            output     = self.copy()
            output.arr = self.arr * other
        else:
            print('Attempting to subtract image and something weird...')
            return None
        
        # Retun the multiplied image
        return output
    
    def __rmul__(self, other):
        # Implements reverse multiplication.
        if other == 0:
            output = self.copy()
            output.arr = np.zeros(self.arr.shape)
            return output
        else:
            return self.__mul__(other)
    
    def __div__(self, other):
        # Implements division using the / operator.
        bothAreImages = isinstance(other, self.__class__)
        oneIsInt      = isinstance(other, int)
        oneIsFloat    = isinstance(other, float)
        
        if bothAreImages:
            # Check that image shapes make sense
            shape1     = self.arr.shape
            shape2     = other.arr.shape
            if shape1 == shape2:
                output     = self.copy()
                output.arr = self.arr / other.arr
            else:
                print('Cannot subtract images with different shapes')
                return None
        elif oneIsInt or oneIsFloat:
            output     = self.copy()
            output.arr = self.arr / other
        else:
            print('Attempting to subtract image and something weird...')
            return None
        
        # Return the divided image
        return output
    
    def __rdiv__(self, other):
        # Implements reverse multiplication.
        if other == 0:
            output = self.copy()
            output.arr = np.zeros(self.arr.shape)
            return output
        else:
            output = Image()
            output.arr = np.ones(self.arr.shape)
            output = output.__mul__(other)
            output = output.__div__(self)
            return output
    
    def __truediv__(self, other):
        # Implements true division (converting to float).
        # TODO catch division by zero (how to handle it?)
        return self.__div__(other)
    
    def __rtruediv__(self, other):
        # Implements reverse true division (converting to float).
        #TODO catch division by zero (how to handle it?)
        if other == 0:
            output = self.copy()
            output.arr = np.zeros(self.arr.shape)
            return output
        else:
            return self.__rdiv__(other)
    
    def copy(self):
        # Define a fresh image to copy all the properties
        output = Image()
        if hasattr(self, 'arr'):
            output.arr = self.arr.copy()
        if hasattr(self, 'binning'):
            output.binning = self.binning
        if hasattr(self, 'header'):
            output.header = self.header.copy()
        if hasattr(self, 'filename'):
            output.filename = self.filename
        
        # I should use "setattr('arr')" or something like that
        # to make sure that dtype is defined as soon as "arr" is defined
        if hasattr(self, 'dtype'):
            output.dtype = self.dtype

        return output
    
    def rebin(self, binning, copy=False):
        """Rebins the using the INTEGER number of bins specified by 'binning'.
        If the image shape is a non-integer multiple of the image array, then
        zero-padding is added to make the image shape a multiple of the
        binning. If 'copy' is True, then the method will return a new copy of
        the image with its array rebinned. Otherwise, the image will be
        rebinned in place.
        """
        # Make a copy of the array before applying padding or rebinning
        rebinArr = self.arr.copy()
        
        # Check if Y-axis needs padding
        remainderY = rebinArr.shape[0] % binning
        remainderX = rebinArr.shape[1] % binning
        if (remainderY != 0) or (remainderX != 0):
            rebinArr = np.pad(rebinArr, ((0,remainderY),(0, remainderX)),
                              mode='constant')
        
        # Get the new shape for the array and compute the rebinning shape
        newShape = self.arr.shape[0]//binning, self.arr.shape[1]//binning
        sh = (newShape[0],rebinArr.shape[0]//newShape[0],
              newShape[1],rebinArr.shape[1]//newShape[1])
        
        # Reassign the image array to be the rebinned array
        rebinArr = rebinArr.reshape(sh).mean(-1).mean(1)
        
        if copy:
            # If a copy was requesty, then return a copy of the original image
            # with a newly rebinned array
            outImg     = self.copy()
            outImg.arr = rebinArr
            return outImg
        else:
            # Otherwise place the rebinned array directly into the Image object
            self.arr = rebinArr
    
    def write(self, filename = ''):
        # Test if a filename was provided and default to current filename
        if len(filename) == 0:
            filename = self.filename
        
        # Write the file to disk
        fits.writeto(filename, self.arr, header = self.header, clobber = True)
    
    def shift(self, dx, dy):
        """A method to shift the image dx pixels to the right and dy pixels up.
        This method will conserve flux!
        
        parameters:
        dx -- number of pixels to shift right (negative is left)
        dy -- number of pixels to shift up (negative is down)
        """
        
        # Store the original shape of the image array
        ny, nx = self.arr.shape
        
        # Check if the X shift is an integer value
        if round(float(dx), 12).is_integer():
            # Force the shift to an integer value
            dx   = np.int(round(dx))
            padX = np.abs(dx)
            
            if dx > 0:
                # Shift image to the right
                shiftImg = np.pad(self.arr, ((0,0), (padX, 0)), mode='constant')
                shiftImg = shiftImg[:,:nx]
                self.arr = shiftImg
            if dx < 0:
                # Shift image to the left
                shiftImg = np.pad(self.arr, ((0,0), (0,padX)), mode='constant')
                shiftImg = shiftImg[:,padX:]
                self.arr = shiftImg
        else:
            # The x-shift is non-integer...
            shiftX = np.int(np.ceil(np.abs(dx)))
            if dx > 0:
                padLf  = (shiftX - 1, 1)
                padRt  = (shiftX, 0)
                
                # Compute the contributing fractions from the left and right images
                fracLf = (shiftX - np.abs(dx))
                fracRt = 1 - fracLf
            elif dx < 0:
                padLf  = (0, shiftX)
                padRt  = (1, shiftX - 1)
                
                # Compute the contributing fractions from the left and right images
                fracLf = 1 - (shiftX - np.abs(dx))
                fracRt = 1 - fracLf
            else:
                padLf  = (0,0)
                padRt  = (0,0)
            
            # Generate the left and right images
            imgLf = np.pad(self.arr,
                           ((0,0), padLf), mode='constant')
            imgRt = np.pad(self.arr,
                           ((0,0), padRt), mode='constant')
            
            # Combine the weighted images                           
            shiftImg = fracLf*imgLf + fracRt*imgRt
            
            # ...and slice off the padded portions
            ny, nx = shiftImg.shape
            if dx > 0:
                shiftImg = shiftImg[:,:(nx-shiftX)]
            elif dx < 0:
                shiftImg = shiftImg[:,shiftX:]
            
            # Place the shifted image in the arr attribute
            self.arr = shiftImg
        
        # Check if the Y shift is an integer value
        if round(float(dy), 6).is_integer():
            # Force the shift to an integer value
            dy   = np.int(round(dy))
            padY = np.abs(dy)
            
            if dy > 0:
                # Shift image to the right
                shiftImg = np.pad(self.arr, ((padY, 0), (0,0)), mode='constant')
                shiftImg = shiftImg[:ny,:]
                self.arr = shiftImg
            if dy < 0:
                # Shift image to the left
                shiftImg = np.pad(self.arr, ((0,padY), (0,0)), mode='constant')
                shiftImg = shiftImg[padY:,:]
                self.arr = shiftImg
        else:
            # The y-shift is non-integer...
            shiftY = np.int(np.ceil(np.abs(dy)))
            if dy > 0:
                padBot  = (shiftY - 1, 1)
                padTop  = (shiftY, 0)

                # Compute the contributing fractions from the top and bottom images
                fracBot = (shiftY - np.abs(dy))
                fracTop = 1 - fracBot

            elif dy < 0:
                padBot  = (0, shiftY)
                padTop  = (1, shiftY - 1)
                
                # Compute the contributing fractions from the top and bottom images
                fracBot = 1 - (shiftY - np.abs(dy))
                fracTop = 1 - fracBot
            else:
                padBot  = (0,0)
                padTop  = (0,0)
            
            # Generate the left and right images
            imgBot = np.pad(self.arr,
                            (padBot, (0,0)), mode='constant')
            imgTop = np.pad(self.arr,
                            (padTop, (0,0)), mode='constant')
            
            # Combine the weighted images
            shiftImg = fracBot*imgBot + fracTop*imgTop
            
            # ...and slice off the padded portions
            ny, nx = shiftImg.shape
            if dy > 0:
                shiftImg = shiftImg[:(ny-shiftY),:]
            elif dy < 0:
                shiftImg = shiftImg[shiftY:,:]
            
            # Place the shifted image in the arr attribute
            self.arr = shiftImg
        
        # Update the header astrometry
        self.header['CRPIX1'] = self.header['CRPIX1'] + dx
        self.header['CRPIX2'] = self.header['CRPIX2'] + dy
        
    
    def align(self, img, fractionalShift=False, mode='wcs',
              offsets=False):
        """A method to align the self image with an other image
        using the astrometry from each header to shift an INTEGER
        number of pixels.
        
        parameters:
        img             -- the image with which self will be aligned
        fractionalShift -- if True, then images are shifted
                           to be aligned with sub-pixel precision
        mode            -- ['wcs' | 'cross-correlate']
                           defines the method used to align the two images
        """
        #**********************************************************************
        # It is unclear if this routine can handle images of different size.
        # It definitely assumes an identical plate scale...
        # Perhaps I needto be adjusting for each of these???
        #**********************************************************************
        
        # Align self image with img image
        
        # Pad the arrays to make sure they are the same size
        ny1, nx1 = self.arr.shape
        ny2, nx2 = img.arr.shape
        newSelf  = self.copy()
        newImg   = img.copy()
        if (nx1 > nx2):
            padX    = nx1 - nx2
            newImg.arr = np.pad(newImg.arr, ((0,0), (0,padX)), mode='constant')
            # Update the header information
            newImg.header['NAXIS1'] = img.header['NAXIS1'] + padX
            del padX
        if (nx1 < nx2):
            padX     = nx2 - nx1
            newSelf.arr = np.pad(newSelf.arr, ((0,0), (0,padX)), mode='constant')
            newSelf.header['NAXIS1'] = self.header['NAXIS1'] + padX
            del padX
        
        if (ny1 > ny2):
            padY    = ny1 - ny2
            newImg.arr = np.pad(newImg.arr, ((0,padY),(0,0)), mode='constant')
            newImg.header['NAXIS2'] = img.header['NAXIS2'] + padY
            del padY
        if (ny1 < ny2):
            padY     = ny2 - ny1
            newSelf.arr = np.pad(newSelf.arr, ((0,padY),(0,0)), mode='constant')
            newSelf.header['NAXIS2'] = self.header['NAXIS2'] + padY
            del padY
        
        if mode == 'wcs':
            # Grab self image WCS and pixel center
            wcs1   = WCS(self.header)
            wcs2   = WCS(img.header)
            x1     = np.mean([wcs1.wcs.crpix[0], wcs2.wcs.crpix[0]])
            y1     = np.mean([wcs1.wcs.crpix[1], wcs2.wcs.crpix[1]])
            
            # Convert pixels to sky coordinates
            RA1, Dec1 = wcs1.all_pix2world(x1, y1, 0)
    
            # Grab the WCS of the alignment image and convert back to pixels
            x2, y2 = wcs2.all_world2pix(RA1, Dec1, 0)
            x2, y2 = float(x2), float(y2)
            
            # Compute the image difference
            dx = x2 - x1
            dy = y2 - y1
            
            # Compute the padding amounts (odd vs. even in python 3.x)
            if (np.int(np.round(dx)) % 2) == 1:
                padX = np.int(np.round(np.abs(dx))/2 + 1)
            else:
                padX = np.int(np.round(np.abs(dx))/2)
            
            if (np.int(np.round(dy)) % 2) == 1:
                padY = np.int(np.round(np.abs(dy))/2 + 1)
            else:
                padY = np.int(np.round(np.abs(dy))/2)
            
            # Construct the before-after padding combinations
            if dx > 0:
                selfDX = (np.int(np.round(np.abs(dx)-padX)), padX)
            else:
                selfDX = (padX, np.int(np.round(np.abs(dx)-padX)))
            
            if dy > 0:
                selfDY = (np.int(np.round(np.abs(dy)-padY)), padY)
            else:
                selfDY = (padY, np.int(np.round(np.abs(dy)-padY)))
            
            imgDX = selfDX[::-1]
            imgDY = selfDY[::-1]
            
            # Compute the shifting amount
            if fractionalShift:
                selfShiftX = +0.5*dx
                imgShiftX  = selfShiftX - dx
                selfShiftY = +0.5*dy
                imgShiftY  = selfShiftY - dy
            else:
                selfShiftX = +np.int(np.round(0.5*dx))
                imgShiftX  = -np.int(np.round(dx - selfShiftX))
                selfShiftY = +np.int(np.round(0.5*dy))
                imgShiftY  = -np.int(np.round(dy - selfShiftY))
    
            
            # Define the padding widths
            # (recall axis ordering is 0=z, 1=y, 2=x, etc...)
            selfPadWidth = np.array((selfDY, selfDX), dtype=np.int)
            imgPadWidth  = np.array((imgDY,  imgDX), dtype=np.int)
            
            # Compute the padding to be added and pad the images
            newSelf.arr = np.pad(self.arr, selfPadWidth, mode='constant')
            newImg.arr  = np.pad(img.arr,  imgPadWidth,  mode='constant')
            
            # Account for un-balanced padding with an initial shift left or down
            initialXshift = selfPadWidth[1,0] - imgPadWidth[1,0]
            if initialXshift > 0:
                newSelf.shift(-initialXshift, 0)
            elif initialXshift < 0:
                newImg.shift(-initialXshift, 0)
            
            initialYshift = selfPadWidth[0,0] - imgPadWidth[0,0]
            if initialYshift > 0:
                newSelf.shift(0, -initialYshift)
            elif initialYshift < 0:
                newImg.shift(0, -initialYshift)
            
            # Update header info
            newSelf.header['NAXIS1'] = newSelf.arr.shape[1]
            newSelf.header['NAXIS2'] = newSelf.arr.shape[0]
            newImg.header['NAXIS1']  = newImg.arr.shape[1]
            newImg.header['NAXIS2']  = newImg.arr.shape[0]
            
            # Shift the images
            newSelf.shift(selfShiftX, selfShiftY)
            newImg.shift(imgShiftX, imgShiftY)
            
            # Save the total offsets
            # TODO check that this is correct
            dx_tot, dy_tot = selfShiftX - imgShiftX, selfShiftY - imgShiftY
                
        elif mode == 'cross_correlate':
            """
            n. b. This method appears to produce results accurate to better
            than 0.1 pixel as determined by simply copying an image, shifting
            it an arbitrary amount, and attempting to recover that shift.
            """
            
            # Do an array flipped convolution, which is a correlation.
            corr = signal.fftconvolve(newSelf.arr, newImg.arr[::-1, ::-1], mode='same')
            
            # Check for the maximum of the cross-correlation function
            peak1  = np.unravel_index(corr.argmax(), corr.shape)
            dy, dx = np.array(peak1) - np.array(corr.shape)//2
            
            # Apply the initial (integer) shifts to the images
            img1 = newImg.copy()
            img1.shift(dx, dy)
            
            # Combine images to find the brightest 25 (or 16, or 9 stars in the image)
            comboImg = newSelf + img1
            
            # Get the image statistics
            mean, median, std = sigma_clipped_stats(comboImg.arr, sigma=3.0, iters=5)
            
            # Find the stars in the images
            sources = daofind(comboImg.arr - median, fwhm=3.0, threshold=5.*std)
            
            # Sort the sources lists by brightness
            sortInds = np.argsort(sources['mag'])
            sources  = sources[sortInds]
            
            # Remove detections within 20 pixels of the image edge
            # (This guarantees that the star-cutout process will succeed)
            ny, nx   = comboImg.arr.shape
            goodX    = np.logical_and(sources['xcentroid'] > 20,
                                     sources['xcentroid'] < (nx - 20))
            goodY    = np.logical_and(sources['ycentroid'] > 20,
                                     sources['ycentroid'] < (ny - 20))
            goodInds = np.where(np.logical_and(goodX, goodY))
            sources  = sources[goodInds]
            
            # Cull the saturated stars from the list
            numStars = len(sources)
            yy, xx   = np.mgrid[0:ny,0:nx]
            badStars = []
            for iStar in range(numStars):
                # Grab pixels less than 10 from the star
                xStar, yStar = sources[iStar]['xcentroid'], sources[iStar]['ycentroid']
                starDist  = np.sqrt((xx - xStar)**2 + (yy - yStar)**2)
                starPatch = comboImg.arr[np.where(starDist < 10)]
                
                # Test if there are bad pixels within 10 from the star
                numBadPix = np.sum(np.logical_or(starPatch > 12e3, starPatch < -100))
                
                # Append the test result to the "badStars" list
                badStars.append(numBadPix > 0)
            
            sources = sources[np.where(np.logical_not(badStars))]
            
            # Cull the list to the brightest few stars
            numStars = len(sources)
            if numStars > 25:
                numStars = 25
            elif numStars > 16:
                numStars = 16
            elif numStars > 9:
                numStars = 9
            else:
                print('There are not very many stars. Is something wrong?')
                pdb.set_trace()
            
            sources = sources[0:numStars]
            
            # Chop out the sections around each star,
            # and build a "starImage"
            starCutout  = 20
            numZoneSide = np.int(np.round(np.sqrt(numStars)))
            starImgSide = starCutout*numZoneSide
            starImg1 = np.zeros((starImgSide, starImgSide))
            starImg2 = np.zeros((starImgSide, starImgSide))
            # Loop through each star to be cut out
            iStar = 0
            for xZone in range(numZoneSide):
                for yZone in range(numZoneSide):
                    # Grab the star positions
                    xStar = np.round(sources['xcentroid'][iStar])
                    yStar = np.round(sources['ycentroid'][iStar])
                    
                    # Establish the cutout bondaries
                    btCut = yStar - np.floor(0.5*starCutout)
                    tpCut = btCut + starCutout
                    lfCut = xStar - np.floor(0.5*starCutout)
                    rtCut = lfCut + starCutout
                            
                    # Establish the pasting boundaries
                    btPaste = starCutout*yZone
                    tpPaste = starCutout*(yZone + 1)
                    lfPaste = starCutout*xZone
                    rtPaste = starCutout*(xZone + 1)
            
                    # Chop out the star and place it in the starImg
                    #    (sqrt-scale cutouts (~SNR per pixel) to emphasize alignment
                    #    of ALL stars not just bright stars).
                    # Apply accurate flooring of values at 0 (rather than simply using np.abs)
                    starImg1[btPaste:tpPaste, lfPaste:rtPaste] = np.sqrt(np.abs(newSelf.arr[btCut:tpCut, lfCut:rtCut]))
                    starImg2[btPaste:tpPaste, lfPaste:rtPaste] = np.sqrt(np.abs(img1.arr[btCut:tpCut, lfCut:rtCut]))
                    
                    # Increment the star counter
                    iStar += 1
            
            # Do an array flipped convolution, which is a correlation.
            corr  = signal.fftconvolve(starImg1, starImg2[::-1, ::-1], mode='same')
            corr  = 100*corr/np.max(corr)
            
            # Check for the maximum of the cross-correlation function
            peak2 = np.unravel_index(corr.argmax(), corr.shape)
            
            # Chop out the central peak
            peakSz = 6
            btCorr = peak2[0] - peakSz
            tpCorr = btCorr + 2*peakSz + 1
            lfCorr = peak2[1] - peakSz
            rtCorr = lfCorr + 2*peakSz + 1
            corr1  = corr[btCorr:tpCorr, lfCorr:rtCorr]
            
            # Get the gradient of the cross-correlation function
            tmp     = Image()
            tmp.arr = corr1
            Gx, Gy  = tmp.gradient()
            grad    = np.sqrt(Gx**2 + Gy**2)
            
            # Fill in edges to remove artifacts
            gradMax      = np.max(grad)
            grad[0:3, :] = gradMax
            grad[:, 0:3] = gradMax
            grad[grad.shape[0]-3:grad.shape[0], :] = gradMax
            grad[:, grad.shape[1]-3:grad.shape[1]] = gradMax
            
            # Grab the index of the minimum
            yMin, xMin = np.unravel_index(grad.argmin(), grad.shape)
            
            # Chop out the central zone and grab the minimum of the gradient
            cenSz = 3
            bot   = yMin - cenSz//2
            top   = bot + cenSz
            lf    = xMin - cenSz//2
            rt    = lf + cenSz
            
            # Grab the region near the minima
            yy, xx   = np.mgrid[bot:top, lf:rt]
            Gx_plane = Gx[bot:top, lf:rt]
            Gy_plane = Gy[bot:top, lf:rt]
            
            # Fit planes to the x and y gradients...Gx
            px_init = models.Polynomial2D(degree=1)
            py_init = models.Polynomial2D(degree=1)
            #fit_p   = fitting.LevMarLSQFitter()
            fit_p   = fitting.LinearLSQFitter()
            px      = fit_p(px_init, xx, yy, Gx_plane)
            py      = fit_p(py_init, xx, yy, Gy_plane)
            
            # Solve these equations using NUMPY
            #0 = px.c0_0 + px.c1_0*xx_plane + px.c0_1*yy_plane
            #0 = py.c0_0 + py.c1_0*xx_plane + py.c0_1*yy_plane
            #
            # This can be reduced to Ax = b, where
            # 
            A = np.matrix([[px.c1_0.value, px.c0_1.value],
                           [py.c1_0.value, py.c0_1.value]])
            b = np.matrix([[-px.c0_0.value],
                           [-py.c0_0.value]])
            
            # Now we can use the build in numpy linear algebra solver
            x_soln = np.linalg.solve(A, b)
            
            # Finally convert back into an absolute image offset
            dx1 = lfCorr + (x_soln.item(0) - (numZoneSide*starCutout)//2)
            dy1 = btCorr + (x_soln.item(1) - (numZoneSide*starCutout)//2)
            dx_tot = dx + dx1
            dy_tot = dy + dy1
            
        else:
            print('mode rot recognized')
            pdb.set_trace(0)
        
        if offsets:
            # Return the image offsets
            return [dx_tot, dy_tot]
        else:
            #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            # IN THE FUTURE THERE SHOULD BE SOME PADDING ADDED TO PREVENT DATA
            # LOSS
            #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            # Apply the final shifts
            newSelf.shift(-0.5*dx_tot, -0.5*dy_tot)
            newImg.shift(+0.5*dx_tot, +0.5*dy_tot)

            # Retun the aligned Images (not the same size as the input images)
            return [newSelf, newImg]
    
    def align_stack(imgList, padding=0, mode='wcs', subPixel=False):
        """A method to align the a whole stack of images using the astrometry
        from each header to shift an INTEGER number of pixels.
        
        parameters:
        imgList -- the list of image to be aligned.
        padding -- the value to use for padding the edges of the aligned
                   images. Common values are 0 and NaN.
        mode    -- ['wcs' | 'cross_correlate'] the method to be used for
                   aligning the images in imgList. 'wcs' uses the astrometry
                   in the header while 'cross_correlation' selects a reference
                   image and computes image offsets using cross-correlation.
        """
        # Catch the case where imgList has only one image
        if len(imgList) <= 1:
            print('Must have more than one image in the list to be aligned')
            return
        
        # Catch the case where imgList has only two images
        if len(imgList) <= 2:
            return imgList[0].align(imgList[1], mode=mode)
        
        #**********************************************************************
        # Get the offsets using whatever mode was selected
        #**********************************************************************
        if mode == 'wcs':
            # Compute the relative position of each of the images in the stack
            wcs1      = WCS(imgList[0].header)
            x1, y1    = imgList[0].arr.shape[1]//2, imgList[0].arr.shape[0]//2
            
            # Append the first image coordinates to the list
            shapeList = [imgList[0].arr.shape]
            imgXpos   = [float(x1)]
            imgYpos   = [float(y1)]
            
            # Convert pixels to sky coordinates
            skyCoord1 = pixel_to_skycoord(x1, y1, wcs1, origin=0, mode='wcs', cls=None)
            
            # Loop through all the remaining images in the list
            # Grab the WCS of the alignment image and convert back to pixels
            for img in imgList[1:]:
                wcs2   = WCS(img.header)
                x2, y2 = skycoord_to_pixel(skyCoord1, wcs2, origin=0, mode='wcs')
                shapeList.append(img.arr.shape)
                imgXpos.append(float(x2))
                imgYpos.append(float(y2))
            
        elif mode == 'cross_correlate':
            # Use the first image in the list as the "reference image"
            refImg = imgList[0]
            
            # Initalize empty lists for storing offsets and shapes
            shapeList = [refImg.arr.shape]
            imgXpos   = [0.0]
            imgYpos   = [0.0]
            
            # Loop through the rest of the images.
            # Use cross-correlation to get relative offsets,
            # and accumulate image shapes
            for img in imgList[1:]:
                dx, dy = refImg.align(img, mode='cross_correlate', offsets=True)
                shapeList.append(img.arr.shape)
                imgXpos.append(-dx)
                imgYpos.append(-dy)
        else:
            print('mode not recognized')
            pdb.set_trace()
        
        # Make sure all the images are the same size
        shapeList = np.array(shapeList)
        nyFinal   = np.max(shapeList[:,0])
        nxFinal   = np.max(shapeList[:,1])
        
        # Compute the median pointing
        x1 = np.median(imgXpos)
        y1 = np.median(imgYpos)
        
        # Compute the relative pointings from the median position
        dx = x1 - np.array(imgXpos)
        dy = y1 - np.array(imgYpos)
        
        # Compute the each distance from the median pointing
        imgDist   = np.sqrt(dx**2.0 + dy**2.0)
        centerImg = np.where(imgDist == np.min(imgDist))[0][0]
        
        # Set the "reference image" to the one closest to the median pointing
        x1, y1 = imgXpos[centerImg], imgYpos[centerImg]
        
        # Recompute the offsets from the reference image
        # (add an 'epsilon' shift to make sure ALL images get shifted
        # at least a tiny bit... this guarantees the images all get convolved
        # by the pixel shape.)
        epsilon = 1e-4
        dx = x1 - np.array(imgXpos) + epsilon
        dy = y1 - np.array(imgYpos) + epsilon
        
        # Check for integer shifts
        for dx1, dy1 in zip(dx, dy):
            if dx1.is_integer(): pdb.set_trace()
            if dy1.is_integer(): pdb.set_trace()
        
        # Compute the total image padding necessary to fit the whole stack
        padLf     = np.int(np.round(np.abs(np.min(dx))))
        padRt     = np.int(np.round(np.max(dx)))
        padBot    = np.int(np.round(np.abs(np.min(dy))))
        padTop    = np.int(np.round(np.max(dy)))
        totalPadX = padLf  + padRt
        totalPadY = padBot + padTop
        
        # Test for sanity
        if (totalPadX > 400) or (totalPadY > 400):
            print('there is a problem with the alignment')
            pdb.set_trace()

        # compute padding
        padX     = (padLf, padRt)
        padY     = (padBot, padTop)
        padWidth = np.array((padY,  padX), dtype=np.int)
        
        # Create an empty list to store the aligned images
        alignedImgList = []
        
        # Loop through each image and pad it accordingly
        for i in range(len(imgList)):
            # Make a copy of the image
            newImg     = imgList[i].copy()
            
            # Check if this image needs an initial padding to match final size
            if (nyFinal, nxFinal) != imgList[i].arr.shape:
                padX       = nxFinal - imgList[i].arr.shape[1]
                padY       = nyFinal - imgList[i].arr.shape[0]
                initialPad = ((0, padY), (0, padX))
                newImg.arr = np.pad(newImg.arr, initialPad,
                                    mode='constant', constant_values=padding)
            
            # Apply the more complete padding
            newImg.arr = np.pad(newImg.arr, padWidth,
                                mode='constant', constant_values=padding)
            
            # Update the header information
            newImg.header['NAXIS1'] = newImg.arr.shape[1]
            newImg.header['NAXIS2'] = newImg.arr.shape[0]

            # Shift the images
            if subPixel:
                shiftX = dx[i]
                shiftY = dy[i]
            else:
                shiftX = np.int(np.round(dx[i]))
                shiftY = np.int(np.round(dy[i]))
            newImg.shift(shiftX, shiftY)
            
            # Append the shifted image
            alignedImgList.append(newImg)
        
        return alignedImgList
    
    def stacked_average(imgList, clipSigma = 3.0):
        """Compute the median filtered mean of a stack of images.
        Standard deviation is computed from the variance of the stack of
        pixels.
        
        parameters:
        imgList   -- a list containing Image class objects.
        clipSigma -- the level at which to trim outliers (default = 3)
        """
        numImg = len(imgList)
        print('\nEntered averaging method')
        if numImg > 1:
            # Test for the correct number of bits in each pixel
            dataType    = imgList[0].dtype
            if dataType == np.int16:
                numBits = 16
            elif (dataType == np.int32) or (dataType == np.float32):
                numBits = 32
            elif (dataType == np.int64) or (dataType == np.float64):
                numBits = 64

            # Compute the number of pixels that fit under the memory limit.
            memLimit    = (psutil.virtual_memory().available/
                          (numBits*(1024**2)))
            memLimit    = int(10*np.floor(memLimit/10.0))
            numStackPix = memLimit*(1024**2)*8/numBits
            ny, nx      = imgList[0].arr.shape
            numRows     = int(np.floor(numStackPix/(numImg*nx)))
            if numRows > ny: numRows = ny
            numSections = int(np.ceil(ny/numRows))
            
            # Compute the number of subsections and display stats to user
            print('\nAiming to fit each stack into {0:g}MB of memory'.format(memLimit))
            print('\nBreaking stack of {0:g} images into {1:g} sections of {2:g} rows'
              .format(numImg, numSections, numRows))
            
            # Initalize an array to store the final averaged image
            finalImg = np.zeros((ny,nx))
            
            # Compute the stacked average of each section
            #
            #
            # TODO Check that this section averaging is working correctly!!!
            #
            #
            for thisSec in range(numSections):
                # Calculate the row numbers for this section
                thisRows = (thisSec*numRows,
                            min([(thisSec + 1)*numRows, ny]))
                
                # Stack the selected region of the images.
                secRows = thisRows[1] - thisRows[0]
                stack   = np.ma.zeros((numImg, secRows, nx), dtype = dataType)
                for i in range(numImg):
                    stack[i,:,:] = imgList[i].arr[thisRows[0]:thisRows[1],:]

                # Catch and mask any NaNs or Infs
                # before proceeding with the average
                NaNsOrInfs  = np.logical_or(np.isnan(stack.data),
                                            np.isinf(stack.data))
                stack.mask  = NaNsOrInfs
                
                # Now that the bad values have been saved,
                # replace them with "good" values
                stack.data[np.where(NaNsOrInfs)] = -1*(10**6)
                
                print('\nAveraging rows {0[0]:g} through {0[1]:g}'.format(thisRows))
            
                # Iteratively clip outliers until answer converges.
                # Use the stacked median for first image estimate.
                outliers = np.zeros(stack.shape, dtype = bool)
               
                # This loop will iterate until the mask converges to an
                # unchanging state, or until clipSigma is reached.
                numLoops  = round((clipSigma - 2)/0.2) + 1
                numPoints = np.zeros((secRows, nx), dtype=int) + 16
                scale     = np.zeros((secRows, nx)) + 2.0
                for iLoop in range(numLoops):
                    print('\tProcessing section for sigma = {0:g}'.format(2.0+iLoop*0.2))
                    # Loop through the stack, and find the outliers.
                    imgEstimate = np.ma.median(stack, axis = 0).data
                    stackSigma  = np.ma.std(stack, axis = 0).data
                    for j in range(numImg):
                        deviation       = np.absolute(stack.data[j,:,:] - imgEstimate)
                        outliers[j,:,:] = (deviation > scale*stackSigma)

                    # Save the outliers to the mask
                    stack.mask = np.logical_or(outliers, NaNsOrInfs)
                    # Save the number of unmasked points along AXIS
                    numPoints1 = numPoints
                    # Total up the new number of unmasked points...
                    numPoints  = np.sum(np.invert(stack.mask), axis = 0)
                    # Figure out which columns have improved results
                    nextScale  = (numPoints != numPoints1)
                    scale     += 0.2*nextScale
                    if np.sum(nextScale) == 0: break

                # Now that this section has been averaged, store it in output.
                finalImg[thisRows[0]:thisRows[1],:] = np.mean(stack, axis = 0)
            return finalImg
        else:
            return imgList[0].arr

    def astrometry(self, override = False):
        """A method to invoke astrometry.net
        and solve the astrometry of the image.
        """
        # Test if the astrometry has already been solved
        try:
            # Try to grab the 'WCSAXES' card from the header
            self.header['WCSAXES']
            
            # If the user forces an override, then set doAstrometry=True
            doAstrometry = override
        except:
            # If there was no 'WCSAXES' card, then set doAstrometry=True
            doAstrometry = True

        
        if doAstrometry:
            # Setup the basic input/output command options
            outputCmd    = ' --out tmp'
            noPlotsCmd   = ' --no-plots'
            overwriteCmd = ' --overwrite'
#            dirCmd       = ' --dir debug'
            dirCmd = ''
    
            # Provide a guess at the plate scale
            scaleLowCmd  = ' --scale-low 0.25'
            scaleHighCmd = ' --scale-high 1.8'
            scaleUnitCmd = ' --scale-units arcsecperpix'
            
            # Provide some information about the approximate location
            raCmd        = ' --ra ' + self.header['TELRA']
            decCmd       = ' --dec ' + self.header['TELDEC']
            radiusCmd    = ' --radius 0.3'
            
            # This is reduced data, so we won't need to clean up the image
#            imageOptions = '--no-fits2fits --no-background-subtraction'
            imageOptions = ''
            
            # Prevent writing any except the "tmp.wcs" file.
            # In the future it may be useful to set '--index-xyls'
            # to save star coordinates for photometry.
            noOutFiles = ' --axy none --corr none' + \
                         ' --match none --solved none' + \
                         ' --new-fits none --rdls none' + \
                         ' --solved none --index-xyls none'
           
            # Build the final command
            command      = 'solve-field' + \
                           outputCmd + \
                           noPlotsCmd + \
                           overwriteCmd + \
                           dirCmd + \
                           scaleLowCmd + \
                           scaleHighCmd + \
                           scaleUnitCmd + \
                           raCmd + \
                           decCmd + \
                           radiusCmd + \
                           imageOptions + \
                           noOutFiles + \
                           ' ' + self.filename
            
            # Run the command in the terminal
            os.system(command)
            
            # Construct the path to the newly created WCS file
            filePathList = self.filename.split(os.path.sep)
            if len(filePathList) > 1:
                wcsPath = os.path.dirname(self.filename) + os.path.sep + 'tmp.wcs'
            else:
                wcsPath = 'tmp.wcs'
            
            # Read in the tmp.wcs file and create a WCS object
            if os.path.isfile(wcsPath):
                HDUlist = fits.open(wcsPath)
                HDUlist[0].header['NAXIS'] = self.header['NAXIS']
                wcsObj = WCS(HDUlist[0].header)
                
                # Build a quick header from the WCS object
                wcsHead = wcsObj.to_header()
                
                # Update the image header to contain the astrometry info
                for key in wcsHead.keys():
                    self.header[key] = wcsHead[key]

                # Delete the WCS file, so it doesn't get used for
                # a different file.
                os.system('rm ' + wcsPath)
                
                # If everything has worked, then return a True success value
                return True
            else:
                # If there was no WCS, then return a False success value
                return False
        else:
            print('Astrometry for {0:s} already solved.'.
              format(os.path.basename(self.filename)))
    
    def gradient(self, kernel='sobel'):
        """Computes and returns the gradient (Gx, Gy) of the image using
        either the Sobel or Prewitt opperators.
        """
        if kernel.upper() == 'SOBEL':
            Gx = ndimage.sobel(self.arr, axis=1)
            Gy = ndimage.sobel(self.arr, axis=0)
        elif kernel.upper() == 'PREWITT':
            Gx = ndimage.prewitt(self.arr, axis=1)
            Gy = ndimage.prewitt(self.arr, axis=0)
        else:
            print('kernel value not recognized')
            pdb.set_trace()
        
        return (Gx, Gy)

    def show(self, axes = None, interactive=True, scale='linear', noShow=False,
             **kwargs):
        """Displays the image to the user for interaction (including clicking?)
        This method includes all the same keyword arguments as the "imshow()"
        method from matplotlip.pyplot. This allows the user to control how the
        image is displayed.
        
        Additional keyword arguments include
        scale -- ['linear' | 'log']
                 allows the user to specify if if the image stretch should be
                 linear or log space
        """
        
        # Set the scaling for the image
        if scale == 'linear':
            showArr = self.arr            
        elif scale == 'log':
            showArr = np.log10(self.arr)
            vmin    = np.log10(vmin)
            vmax    = np.log10(vmax)
        elif scale == 'asinh':
            showArr = np.arcsinh(self.arr)
            vmin    = np.arcsinh(vmin)
            vmax    = np.arcsinh(vmax)
        else:
            print('The provided "scale" keyword is not recognized')
            pdb.set_trace()
        
        # Create the figure and axes for displaying the image
        if axes is None:
            # Create a new figure and axes
            fig  = plt.figure(figsize = (8,8))
            axes = fig.add_subplot(1,1,1)
            axIm = axes.imshow(showArr, **kwargs)

        else:
            fig  = axes.figure
            axIm = axes.imshow(showArr, **kwargs)        
        # TODO detirmine how to handle axis labels and titles
        
#        axes.set_xlabel('X position [pix]')
#        axes.set_ylabel('Y position [pix]')
#        axes.set_title(os.path.basename(self.filename))
        
        # Display the image to the user, if requested
        if not noShow:
            plt.ion()
            fig.show()
            plt.ioff()
        
        # Return the graphics objects to the user
        return (fig, axes, axIm)

    def coordinates(axes):
        """This method will overplot the image coordinate markings (RA, Dec)
        on the specified axes.
        """
        
        