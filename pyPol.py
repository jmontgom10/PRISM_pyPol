import pdb
import psutil
import time
import numpy as np
from scipy import ndimage
import scipy.stats
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel

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
    
    def write(self, filename = ''):
        # Test if a filename was provided and default to current filename
        if len(filename) == 0:
            filename = self.filename
        fits.writeto(filename, self.arr, header = self.header, clobber = True)
    
    def copy(self):
        output = Image()
        output.arr = self.arr.copy()
        output.binning = self.binning
        output.header = self.header.copy()
        output.filename = self.filename.copy()
        output.dtype = self.dtype
        return output
    
    # TODO define a subtraction method for image array
    # test for if it's another image or a similarly sized array
    
    def shift(self, dx, dy):
        """A method to shift the image dx pixels to the right and dy pixels up.
        This method will conserve flux!
        
        parameters;
        dx -- number of pixels to shift right (negative is left)
        dy -- number of pixels to shift up (negative is down)
        """
        
        # Define the (before, after) padding pairs for integer shifts
        shiftX = np.int(np.ceil(np.abs(dx)))
        shiftY = np.int(np.ceil(np.abs(dy)))
        if dx > 0:
            padLf  = (shiftX - 1, 1)
            padRt  = (shiftX, 0)
        elif dx < 0:
            padLf  = (0, shiftX)
            padRt  = (1, shiftX - 1)
        else:
            padLf  = (0,0)
            padRt  = (0,0)
        
        if dy > 0:
            padBot = (shiftY - 1, 1)
            padTop = (shiftY, 0)
        elif dy < 0:
            padBot = (0, shiftY)
            padTop = (1, shiftY - 1)
        else:
            padBot = (0,0)
            padTop = (0,0)
        
        # Create the padding shifted versions of the array
        img00 = np.pad(self.arr,
                       (padBot, padLf), mode='constant')
        img01 = np.pad(self.arr,
                       (padTop, padLf), mode='constant')
        img10 = np.pad(self.arr,
                       (padBot, padRt), mode='constant')
        img11 = np.pad(self.arr,
                       (padTop, padRt), mode='constant')
        
        # Now compute the output image
        fracX = shiftX - np.abs(dx)
        fracY = shiftY - np.abs(dy)
        imgOut = (fracX*(1-fracY))*img00 + \
                 ((1-fracX)*(1-fracY))*img10 + \
                 (fracX*fracY)*img01 + \
                 ((1-fracX)*fracY)*img11
        
        # ...and slice off the padded portions
        ny, nx = imgOut.shape
        if dx > 0:
            imgOut = imgOut[:,:(nx-shiftX)]
        elif dx < 0:
            imgOut = imgOut[:,shiftX:]
        
        if dy > 0:
            imgOut = imgOut[:(ny-shiftY),:]
        elif dy < 0:
            imgOut = imgOut[shiftY:,:]
        
        # Check that things are still working correctly
        if imgOut.shape != self.arr.shape:
            pdb.set_trace()
        
        # Replace theimage array with the shifted version
        self.arr = imgOut
        

    def align(self, img):
        """A method to align the self image with an other image
        using the astrometry from each header to shift an INTEGER
        number of pixels.
        
        parameters:
        img -- the image with which self will be aligned
        """
        # Align self image with img image
        
        # Grab self image WCS and pixel center
        wcs1   = WCS(self.header)
        x1, y1 = self.arr.shape[0]//2, self.arr.shape[1]//2
        
        # Convert pixels to sky coordinates
        skyCoord1 = pixel_to_skycoord(x1, y1, wcs1, origin=0, mode='wcs', cls=None)

        # Grab the WCS of the alignment image and convert back to pixels
        wcs2   = WCS(img.header)
        x2, y2 = skycoord_to_pixel(skyCoord1, wcs2, origin=0, mode='wcs')
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
        selfShiftX = +np.int(np.round(0.5*dx))
        imgShiftX  = -np.int(np.round(dx - selfShiftX))
        selfShiftY = +np.int(np.round(0.5*dy))
        imgShiftY  = -np.int(np.round(dy - selfShiftY))
        
        # Define the padding widths
        # (recall axis ordering is 0=z, 1=y, 2=x, etc...)
        selfPadWidth = np.array((selfDY, selfDX), dtype=np.int)
        imgPadWidth  = np.array((imgDY,  imgDX), dtype=np.int)
        
        # Compute the padding to be added and pad the images
        newSelf     = self.copy()
        newImg      = img.copy()
        newSelf.arr = np.pad(self.arr, selfPadWidth, mode='constant')
        newImg.arr  = np.pad(img.arr,  imgPadWidth,  mode='constant')

        # Shift the images
        newSelf.shift(selfShiftX, selfShiftY)
        newImg.shift(imgShiftX, imgShiftY)
        
        # Update the header astrometry
        newSelf.header['CRPIX1'] = newSelf.header['CRPIX1'] + selfShiftX
        newSelf.header['CRPIX2'] = newSelf.header['CRPIX2'] + selfShiftY
        newImg.header['CRPIX1']  = newImg.header['CRPIX1'] + imgShiftX
        newImg.header['CRPIX2']  = newImg.header['CRPIX2'] + imgShiftY

        # Retun the aligned Images (not the same size as the input images)
        return [newSelf, newImg]
    
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
                    stack.mask = outliers                    
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
