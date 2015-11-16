import os
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import ascii
from astropy.table import Column as Column
import pdb
from pyPol import Image

#******************************************************************************
# Write a quick function to provide list string searching...
#******************************************************************************
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
    global fileList, fig, imgNum, brushSize
    global maskDir, maskImg
    global prevImg,   thisImg,   nextImg
    global prevAxImg, thisAxImg, nextAxImg
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
            nextImg = Image(fileList[imgNum + 1])
            
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
            prevImg = Image(fileList[imgNum - 1])
    
            # Compute new image display minima
            nextMin = thisMin
            thisMin = prevMin
            prevMin = np.median(prevImg.arr) - 0.25*np.std(prevImg.arr)
            
            # Compute new image display maxima
            nextMax = thisMax
            thisMax = prevMax
            prevMax = np.median(prevImg.arr) + 2*np.std(prevImg.arr)
        
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
        
        # Generate the correct filename and write to disk
        filename = maskDir + os.path.sep + os.path.basename(thisImg.filename)
        maskImg.write(filename)
    
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
global fileList, fig, imgNum, maskDir, maskImg
global prevImg,   thisImg,   nextImg
global prevAxImg, thisAxImg, nextAxImg
global prevMin,   thisMin,   nextMin
global prevMax,   thisMax,   nextMax
global prevLabel, thisLabel, nextLabel
xList     = []
yList     = []
imgNum    = 0      # This number will be the FIRST image to be displayed center...
brushSize = 3      # (5xbrushSize pix) is the size of the region masked

#******************************************************************************
# First the user must identify the names of the targets to be batched
# (in this case we only need to worry about reflection nebulae)
#******************************************************************************
targets = ['NGC2023', 'NGC7023', 'NGC_1977']

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

# Setup new directory for polarimetry data
maskDir = reducedDir + delim + 'Masks'
if (not os.path.isdir(maskDir)):
    os.mkdir(maskDir, 0o755)

# Read the fileIndex back in as an astropy Table
print('\nReading file index from disk')
indexFile = 'fileIndex.dat'
fileIndex = ascii.read(indexFile)

# Determine which parts of the Fileindex pertain to science images
keepFiles = []
for file in fileIndex['Filename']:
    Filename = file.split(delim)
    Filename.reverse()
    Filename = reducedDir + delim + Filename[0]
    keepFiles.append(Filename in fileList)

# Update the fileIndex with the paths to reduced files
fileIndex = fileIndex[np.where(keepFiles)]
fileIndex['Filename'] = fileList

# Read the groupDither back in as an astropy Table
print('\nReading dither index from disk')
groupDither = ascii.read('groupDither.dat')


# Use the groupDither information to add a "Dither" column to the fileIndex
nullStr    = 'ThisDitherIsNotRecorded'
ditherList = np.array([nullStr]*len(fileIndex))
for group, dither in groupDither:
    groupInds = np.where(str_list_contains(fileIndex['Group'].data, group))
    ditherList[groupInds] = dither

# Add a "Dither" column to the fileIndex
fileIndex.add_column(Column(name='Dither', data=ditherList), index=5)

# Prepare to add a 'Target' Column to the fileIndex
nullStr   = "ThisIsNotATarget"
groupList = []
groupList.extend(fileIndex['Group'].data)
targetList = np.array([nullStr]*len(groupList))

# Loop through each of the targets and identify
# which groups are assigned to each target.
for target in targets:
    targetInds = np.where(str_list_contains(groupList, target))
    targetList[targetInds] = target

# Add a "Target" column to the fileIndex
fileIndex.add_column(Column(name='Target', data=targetList), index=2)

# Remove non-target elements of the fileIndex
keepFiles = [not i for i in str_list_contains(targetList, nullStr)]
fileIndex = fileIndex[np.where(keepFiles)]

# Group the fileIndex by...
# 1. Target
# 2. Waveband
# 3. Dither (pattern)
# 4. Polaroid Angle
fileIndexByTarget = fileIndex.group_by(['Dither', 'Target', 'Waveband', 'Polaroid Angle'])

# Identify the "ON" and "OFF" images for each group,
# and generate a final list of images that need masking
fileList = []
for group in fileIndexByTarget.groups:
    # Count the images
    numImgs = len(group)
    
    # Test if numImgs matches the ABBA pattern
    if (numImgs % 4) != 0:
        print('The ABBA pattern is not there...')
        pdb.set_trace()
    else:
        imgOnTarget   = [True, False, False, True]*np.int(numImgs/4)
        onTargetInds  = np.where(imgOnTarget)[0]
        onTargetFiles = (group['Filename'])[np.where(imgOnTarget)]
        fileList.extend(onTargetFiles)

#******************************************************************************
# Sort the final file list
# NOT SORTING MAKES THE MASKING TASK A BIT EASIER
#fileList = np.sort(fileList)
#******************************************************************************

#*************************************
# Now prepare to plot the first images
#*************************************

# Read in an image for masking
prevImg = Image(fileList[imgNum - 1])
thisImg = Image(fileList[imgNum])
nextImg = Image(fileList[imgNum + 1])

# Build a mask template (0 = not masked, 1 = masked)
maskImg       = thisImg.copy()
#maskImg.arr   = maskImg.arr * 0
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
#plt.ion()
plt.show()
#plt.ioff()

# Disconnect the event manager and close the figure
fig.canvas.mpl_disconnect(cid1)
fig.canvas.mpl_disconnect(cid2)

# Close the plot
plt.close()
