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
#def display_tri_plot():
#    global fig, ax

def on_click(event):
    global xList,yList
    
    x, y  = event.xdata, event.ydata
    xList.append(x)
    yList.append(y)
    
    # Update mask array
    
    
#    print("({0}, {1})".format(np.round(x), np.round(y)))
    
    return

def on_key(event):
    global fileList, fig, imgNum, prevLabel, thisLabel, nextLabel
    
    # increment the image number
    if event.key == 'right':
        #Advance to the next image
        imgNum = imgNum + 1
    
    if event.key == 'left':
        #Move back to the previous image
        imgNum = imgNum - 1
    
    print('plotting image number {0}'.format(imgNum))
    
    # Save the generate mask
    
    # Read in thenew files
    prevImg = Image(fileList[imgNum - 1])
    thisImg = Image(fileList[imgNum])
    nextImg = Image(fileList[imgNum + 1])

    # Display the new images
    axList = fig.get_axes()
    tmpFig, tmpAx = prevImg.show(axes = axList[0], cmap='cubehelix',
                                 vmin = 0.01, vmax = 300, noShow = True)
    tmpFig, tmpAx = thisImg.show(axes = axList[1], cmap='cubehelix',
                                 vmin = 0.01, vmax = 300, noShow = True)
    tmpFig, tmpAx = nextImg.show(axes = axList[2], cmap='cubehelix',
                                 vmin = 0.01, vmax = 300, noShow = True)

    # Update the annotation
    axList[1].set_title(os.path.basename(thisImg.filename))
    prevLabel.set_text(prevImg.header['POLPOS'])
    thisLabel.set_text(thisImg.header['POLPOS'])
    nextLabel.set_text(nextImg.header['POLPOS'])
    
    # Update the display
    fig.canvas.draw()
#    display_tri_plot()
    

#******************************************************************************

#******************************************************************************
# This is the main script that will load in file names and prepare for plotting
#******************************************************************************

# Declare global variables
global xList, yList, fileList, fig, imgNum, prevLabel, thisLabel, nextLabel
xList  = []
yList  = []
imgNum = 0      # This number will be the FIRST image to be displayed center...

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
polarimetryDir = reducedDir + delim + 'Polarimetry'
if (not os.path.isdir(polarimetryDir)):
    os.mkdir(polarimetryDir, 0o755)

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
fileIndexByTarget = fileIndex.group_by(['Target', 'Waveband', 'Dither', 'Polaroid Angle'])

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

# Sort the final file list
fileList = np.sort(fileList)

#*************************************
# Now prepare to plot the first images
#*************************************














# Read in an image for masking
#fileToMask = fileList'/home/jordan/ThesisData/PRISM_Data/Reduced_data/20150117.145.fits'
prevImg = Image(fileList[imgNum - 1])
thisImg = Image(fileList[imgNum])
nextImg = Image(fileList[imgNum + 1])

# Build a mask template (0 = not masked, 1 = masked)
maskImg     = thisImg.copy()
maskImg.arr = maskImg.arr * 0

# Build the image displays
# Start by preparing a 1x3 plotting area
fig, axarr = plt.subplots(1, 3, sharey=True)

# Loop through each axis and populate it.
tmpFig, tmpAx = prevImg.show(axes = axarr[0], cmap='cubehelix',
                                     vmin = 0.01, vmax = 300, noShow = True)
tmpFig, tmpAx = thisImg.show(axes = axarr[1], cmap='cubehelix',
                                     vmin = 0.01, vmax = 300, noShow = True)
tmpFig, tmpAx = nextImg.show(axes = axarr[2], cmap='cubehelix',
                                     vmin = 0.01, vmax = 300, noShow = True)

plt.subplots_adjust(left = 0.04, bottom = 0.04, right = 0.98, top = 0.96,
                    wspace = 0.02, hspace = 0.02)

# Reset figure aspect ratio
#fig.set_figheight(5.575, forward=True)
#fig.set_figwidth(17.0, forward=True)
fig.set_size_inches(17, 5.675, forward=True)

# Add some figure annotation
thisTitle = axarr[1].set_title(os.path.basename(thisImg.filename))
prevLabel = axarr[0].text(20, 950, prevImg.header['POLPOS'],
                          color = 'white', size = 'medium')
thisLabel = axarr[1].text(20, 950, thisImg.header['POLPOS'],
                          color = 'white', size = 'medium')
nextLabel = axarr[2].text(20, 950, nextImg.header['POLPOS'],
                          color = 'white', size = 'medium')

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
plt.ion()
plt.show()
plt.ioff()
pdb.set_trace()

# Disconnect the event manager and close the figure
fig.canvas.mpl_disconnect(cid1)
fig.canvas.mpl_disconnect(cid2)

# Close the plot
plt.close()
