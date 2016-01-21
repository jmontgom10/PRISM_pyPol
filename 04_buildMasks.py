import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import ascii
from astropy.table import Column as Column
from astropy.table import Table as Table
import pdb

global maskDir

# Add the AstroImage class
sys.path.append("C:\\Users\\Jordan\\Libraries\\python\\AstroImage")
from AstroImage import AstroImage

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
fileIndex = Table.read(indexFile, format='csv')

# The user can speed up the process by defining the "Target" values from
# the fileIndex to be considered for masking.
# Masks can onlybe produced for targets in this list.
targets = ['NGC2023', 'NGC7023', 'NGC1977', 'M78']

# Setup new directory for polarimetry data
maskDir = os.path.join(pyPol_data, 'Masks')
if (not os.path.isdir(maskDir)):
    os.mkdir(maskDir, 0o755)

# Read in the indexFile data and select the filenames
print('\nReading file index from disk')
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='csv')

# Determine which parts of the fileIndex pertain to science images
useFiles = np.logical_and(fileIndex['Use'] == 1,
                          fileIndex['Dither'] == 'ABBA')

# Further restrict the selection to only include the selected targets
targetFiles = np.array([False]*len(fileIndex), dtype=bool)
for target in targets:
    targetFiles = np.logical_or(targetFiles,
                                fileIndex['Target'] == target)

# Cull the fileIndex to ONLY include the specified targets
fileIndex = fileIndex[np.where(np.logical_and(useFiles, targetFiles))]


#******************************************************************************
# Define the event handlers for clicking and keying on the image display
#******************************************************************************

def on_click(event):
    global xList, yList, xx, yy
    global fig, brushSize, axarr, maskImg, thisAxImg

    x, y = event.xdata, event.ydata
#    xList.append(x)
#    yList.append(y)

    # Compute distances from the click and update mask array
    dist     = np.sqrt((xx - x)**2 + (yy - y)**2)
    maskInds = np.where(dist < brushSize*5)
    if event.button == 1:
        maskImg.arr[maskInds] = 1
    if (event.button == 2) or (event.button == 3):
        maskImg.arr[maskInds] = 0

    # Update contour plot (clear old lines redo contouring)
    axarr[1].collections = []
    axarr[1].contour(xx, yy, maskImg.arr, levels=[0.5], colors='white', alpha = 0.2)

    # Update the display
    fig.canvas.draw()

def on_key(event):
    global fileList, targetList, fig, imgNum, brushSize
    global maskDir, maskImg
    global prevImg,   thisImg,   nextImg
    global prevAxImg, thisAxImg, nextAxImg
    global prevTarget, thisTarget, nextTarget
    global prevMin,   thisMin,   nextMin
    global prevMax,   thisMax,   nextMax
    global prevLabel, thisLabel, nextLabel

    # Handle brush sizing
    if event.key == '1':
        brushSize = 1
    elif event.key == '2':
        brushSize = 2
    elif event.key == '3':
        brushSize = 3
    elif event.key == '4':
        brushSize = 4
    elif event.key == '5':
        brushSize = 5
    elif event.key == '6':
        brushSize = 6

    # Increment the image number
    if event.key == 'right' or event.key == 'left':
        if event.key == 'right':
            #Advance to the next image
            imgNum += 1

            # Read in the new files
            prevImg = thisImg
            thisImg = nextImg
            nextImg = AstroImage(fileList[(imgNum + 1) % len(fileList)])

            # Update target info
            prevTarget = thisTarget
            thisTarget = nextTarget
            nextTarget = targetList[(imgNum + 1) % len(fileList)]

            # Compute new image display minima
            prevMin = thisMin
            thisMin = nextMin
            nextMin = np.median(nextImg.arr) - 0.25*np.std(nextImg.arr)

            # Compute new image display maxima
            prevMax = thisMax
            thisMax = nextMax
            nextMax = np.median(nextImg.arr) + 2*np.std(nextImg.arr)

        if event.key == 'left':
            #Move back to the previous image
            imgNum -= 1

            # Read in the new files
            nextImg = thisImg
            thisImg = prevImg
            prevImg = AstroImage(fileList[(imgNum - 1) % len(fileList)])

            # Update target info
            nextTarget = thisTarget
            thisTarget = prevTarget
            prevTarget = targetList[(imgNum - 1) % len(fileList)]

            # Compute new image display minima
            nextMin = thisMin
            thisMin = prevMin
            prevMin = np.median(prevImg.arr) - 0.25*np.std(prevImg.arr)

            # Compute new image display maxima
            nextMax = thisMax
            thisMax = prevMax
            prevMax = np.median(prevImg.arr) + 2*np.std(prevImg.arr)

        #*******************************
        # Update the displayed mask
        #*******************************

        # Check which mask files might be usable...
        prevMaskFile = os.path.join(maskDir,
            os.path.basename(prevImg.filename))
        thisMaskFile = os.path.join(maskDir,
            os.path.basename(thisImg.filename))
        nextMaskFile = os.path.join(maskDir,
            os.path.basename(nextImg.filename))
        if os.path.isfile(thisMaskFile):
            # If the mask for this file exists, use it
            print('using this mask: ',os.path.basename(thisMaskFile))
            maskImg = AstroImage(thisMaskFile)
        elif os.path.isfile(prevMaskFile) and (prevTarget == thisTarget):
            # Otherwise check for the mask for the previous file
            print('using previous mask: ',os.path.basename(prevMaskFile))
            maskImg = AstroImage(prevMaskFile)
        elif os.path.isfile(nextMaskFile) and (nextTarget == thisTarget):
            # Then check for the mask of the next file
            print('using next mask: ',os.path.basename(nextMaskFile))
            maskImg = AstroImage(nextMaskFile)
        else:
            # If none of those files exist, build a blank slate
            # Build a mask template (0 = not masked, 1 = masked)
            maskImg       = thisImg.copy()
            maskImg.filename = thisMaskFile
            maskImg.arr   = maskImg.arr.astype(np.int16) * np.int16(0)
            maskImg.dtype = np.byte
            maskImg.header['BITPIX'] = 16

        # Update contour plot (clear old lines redo contouring)
        axarr[1].collections = []
        axarr[1].contour(xx, yy, maskImg.arr, levels=[0.5], colors='white', alpha = 0.2)

        # Reassign image display limits
        prevAxImg.set_clim(vmin = prevMin, vmax = prevMax)
        thisAxImg.set_clim(vmin = thisMin, vmax = thisMax)
        nextAxImg.set_clim(vmin = nextMin, vmax = nextMax)

        # Display the new images
        prevAxImg.set_data(prevImg.arr)
        thisAxImg.set_data(thisImg.arr)
        nextAxImg.set_data(nextImg.arr)

        # Update the annotation
        axList = fig.get_axes()
        axList[1].set_title(os.path.basename(thisImg.filename))

        prevStr   = (str(prevImg.header['OBJECT']) + '\n' +
                     str(prevImg.header['FILTNME3'] + '\n' +
                     str(prevImg.header['POLPOS'])))
        thisStr   = (str(thisImg.header['OBJECT']) + '\n' +
                     str(thisImg.header['FILTNME3'] + '\n' +
                     str(thisImg.header['POLPOS'])))
        nextStr   = (str(nextImg.header['OBJECT']) + '\n' +
                     str(nextImg.header['FILTNME3'] + '\n' +
                     str(nextImg.header['POLPOS'])))
        prevLabel.set_text(prevStr)
        thisLabel.set_text(thisStr)
        nextLabel.set_text(nextStr)

        # Update the display
        fig.canvas.draw()

    # Save the generated mask
    if event.key == 'enter':
        # Make sure the header has the right values
        maskImg.header = thisImg.header

        # Write the mask to disk
        print('Writing mask for file ', os.path.basename(maskImg.filename))
        maskImg.write()

    # Clear out the mask values
    if event.key == 'backspace':
        # Clear out the mask array
        maskImg.arr = maskImg.arr * np.byte(0)

        # Update contour plot (clear old lines redo contouring)
        axarr[1].collections = []
        axarr[1].contour(xx, yy, maskImg.arr, levels=[0.5], colors='white', alpha = 0.2)

        # Update the display
        fig.canvas.draw()

#******************************************************************************

#******************************************************************************
# This is the main script that will load in file names and prepare for plotting
#******************************************************************************

# Declare global variables
#global xList, yList
global xx, yy
global fileList, targetList, fig, imgNum, maskImg
global prevImg,   thisImg,   nextImg
global prevTarget, thisTarget, nextTarget
global prevAxImg, thisAxImg, nextAxImg
global prevMin,   thisMin,   nextMin
global prevMax,   thisMax,   nextMax
global prevLabel, thisLabel, nextLabel
xList     = []
yList     = []
imgNum    = 0      # This number will be the FIRST image to be displayed center...
brushSize = 3      # (5xbrushSize pix) is the size of the region masked

#******************************************************************************
# This script will run the mask building step of the pyPol reduction
#******************************************************************************

# Group the fileIndex by...
# 1. Target
# 2. Waveband
# 3. Dither (pattern)
# 4. Polaroid Angle
fileIndexByTarget = fileIndex.group_by(['Dither', 'Target', 'Waveband', 'Polaroid Angle'])

# Identify the "ON" and "OFF" images for each group,
# and generate a final list of images that need masking
fileList = []
targetList = []
for group in fileIndexByTarget.groups:
    # Count the images
    numImgs = len(group)

    # Test if numImgs matches the ABBA pattern
    if (numImgs % 4) != 0:
        print('The ABBA pattern is not there...')
        pdb.set_trace()
    else:
        # Get the on target listings for the filenames and target values
        imgOnTarget   = [True, False, False, True]*np.int(numImgs/4)
        onTargetFiles = (group['Filename'])[np.where(imgOnTarget)]
        onTargetTargets = (group['Target'])[np.where(imgOnTarget)]

        # Strip the leading and trailing spaces on the target names
        for i, target in enumerate(onTargetTargets):
            onTargetTargets[i] = target.strip()

        # Add the information to the fileList and targetList variables
        fileList.extend(onTargetFiles)
        targetList.extend(onTargetTargets)

#*************************************
# Now prepare to plot the first images
#*************************************

# Read in an image for masking
prevImg = AstroImage(fileList[imgNum - 1])
thisImg = AstroImage(fileList[imgNum])
nextImg = AstroImage(fileList[imgNum + 1])

# Log the targets of the curent panes
prevTarget = targetList[imgNum - 1]
thisTarget = targetList[imgNum]
nextTarget = targetList[imgNum + 1]

###
# For some reason the prevTarget, thisTarget, and nextTaret
# variables are not accessible from the event managers the way that
# prevImg, thisImg, and nextImg are.
# I have definitely declared them to be global variables...
# Perhaps they're getting treated as local variables
# because they are modified elsewhere???


# Test if a mask has already been generated for this images
maskFile = os.path.join(maskDir, os.path.basename(thisImg.filename))
if os.path.isfile(maskFile):
    # If the mask file exists, use it
    maskImg = AstroImage(maskFile)
else:
    # If the mask file does not exist, build a blank slate
    # Build a mask template (0 = not masked, 1 = masked)
    maskImg       = thisImg.copy()
    maskImg.filename = maskFile
    maskImg.arr   = maskImg.arr.astype(np.int16) * np.int16(0)
    maskImg.dtype = np.byte
    maskImg.header['BITPIX'] = 16

# Generate 2D X and Y position maps
maskShape = maskImg.arr.shape
grids     = np.mgrid[0:maskShape[0], 0:maskShape[1]]
xx        = grids[1]
yy        = grids[0]

# Build the image displays
# Start by preparing a 1x3 plotting area
fig, axarr = plt.subplots(1, 3, sharey=True)

# Compute image count scaling
prevMin = np.median(prevImg.arr) - 0.25*np.std(prevImg.arr)
prevMax = np.median(prevImg.arr) + 2*np.std(prevImg.arr)
thisMin = np.median(thisImg.arr) - 0.25*np.std(thisImg.arr)
thisMax = np.median(thisImg.arr) + 2*np.std(thisImg.arr)
nextMin = np.median(nextImg.arr) - 0.25*np.std(nextImg.arr)
nextMax = np.median(nextImg.arr) + 2*np.std(nextImg.arr)

# Populate each axis with its image
tmpFig, tmpAx, prevAxImg = prevImg.show(axes = axarr[0], cmap='cubehelix',
                                        vmin = prevMin, vmax = prevMax, noShow = True)
tmpFig, tmpAx, thisAxImg = thisImg.show(axes = axarr[1], cmap='cubehelix',
                                        vmin = thisMin, vmax = thisMax, noShow = True)
tmpFig, tmpAx, nextAxImg = nextImg.show(axes = axarr[2], cmap='cubehelix',
                                        vmin = nextMin, vmax = nextMax, noShow = True)

# Add a contour of the mask array
maskContour = axarr[1].contour(xx, yy, maskImg.arr,
                               levels=[0.5], origin='lower', colors='white', alpha = 0.2)

# Rescale the figure and setup the spacing between images
#fig.set_figheight(5.575, forward=True)
#fig.set_figwidth(17.0, forward=True)
fig.set_size_inches(17, 5.675, forward=True)
plt.subplots_adjust(left = 0.04, bottom = 0.04, right = 0.98, top = 0.96,
                    wspace = 0.02, hspace = 0.02)

# Add some figure annotation
thisTitle = axarr[1].set_title(os.path.basename(thisImg.filename))
prevStr   = (str(prevImg.header['OBJECT']) + '\n' +
             str(prevImg.header['FILTNME3'] + '\n' +
             str(prevImg.header['POLPOS'])))
thisStr   = (str(thisImg.header['OBJECT']) + '\n' +
             str(thisImg.header['FILTNME3'] + '\n' +
             str(thisImg.header['POLPOS'])))
nextStr   = (str(nextImg.header['OBJECT']) + '\n' +
             str(nextImg.header['FILTNME3'] + '\n' +
             str(nextImg.header['POLPOS'])))
prevLabel = axarr[0].text(20, 875, prevStr,
                          color = 'white', size = 'medium')
thisLabel = axarr[1].text(20, 875, thisStr,
                          color = 'white', size = 'medium')
nextLabel = axarr[2].text(20, 875, nextStr,
                          color = 'white', size = 'medium')
thisShape = thisImg.arr.shape
redLines  = axarr[1].plot([thisShape[0]/2, thisShape[0]/2], [0, thisShape[1]],
                          '-r',
                          [0, thisShape[0]], [thisShape[1]/2, thisShape[1]/2],
                          '-r', alpha = 0.4)
#********************************************
#log this for future use!
#********************************************
# A more standard way to handle mouse clicks?
#xyList = fig.ginput(n=-1, timeout=-30, show_clicks=True,
#                    mouse_add=1, mouse_pop=3, mouse_stop=2)
#********************************************

# Connect the event manager...
cid1 = fig.canvas.mpl_connect('button_press_event',on_click)
cid2 = fig.canvas.mpl_connect('key_press_event', on_key)

# NOW show the image (without continuing execution)
# plt.ion()
plt.show()
# plt.ioff()
#
# pdb.set_trace()
# Disconnect the event manager and close the figure
fig.canvas.mpl_disconnect(cid1)
fig.canvas.mpl_disconnect(cid2)

# Close the plot
plt.close()
