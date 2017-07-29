# -*- coding: utf-8 -*-
"""
Restructures the rawFileIndex from PRISM_pyBDP to contain ONLY the science
images, and break those up into individual groups based on changes in

1) OBJECT (object name)
2) FILTER (optical filter value)
3) EXPTIME (the exposure time of the images)
4) BINNING (the 1x1, 2x2, 3x3, or 4x4 binning of the images)
5) Pointing changes (more than 1.5 degrees of chang is considered a new group)

Attempts to associate each group with a target in the 'targetList' variable on
the basis of the string in the OBJECT column of that group.

Saves the index file with a USE and GROUP_ID columns added to the table.
"""

#Import whatever modules will be used
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy import stats

# Add the AstroImage class
import astroimage as ai

# Set the directory for the pyPol reduced data
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data\\201612\\'

# Set the filename for the reduced data indexFile
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')

# Compose a list of expected targets. All groups will be assigned to ONE of
# these targets within a given tolerance. If no match is found for a group
# within this list of targets, then an error will be raised.
targetList = [
    'NGC7023',
    'NGC2023',
    'Merope',
    'NGC4565',
    'NGC891',
    'Orion_Cal',
    'Cyg_OB2'
]

# Force all the targets to be upper case to remove ambiguity
targetList = [t.upper() for t in targetList]

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================
# This is the location of all pyBDP data (index, calibration images, reduced...)
pyBDP_data='C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyBDP_data\\201612'

# This is the location where all pyPol data will be saved
pyPol_data='C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data\\201612'
if (not os.path.isdir(pyPol_data)):
    os.mkdir(pyPol_data, 0o755)

# This is the location of the pyBDP processed Data
pyBDP_reducedDir = os.path.join(pyBDP_data, 'pyBDP_reduced_images')

# Load the index of raw files
rawFileIndexFile = os.path.join(pyBDP_data, 'rawFileIndex.csv')
rawFileIndex     = Table.read(rawFileIndexFile)

# Cull the index to only include the science images
scienceBool      = rawFileIndex['OBSTYPE'] == 'OBJECT'
scienceInds      = np.where(scienceBool)
reducedFileIndex = rawFileIndex[scienceInds]

# Remap the filenames to be the reduced filenames
fileBasenames    = [os.path.basename(f) for f in reducedFileIndex['FILENAME']]
reducedFilenames = [os.path.join(pyBDP_reducedDir, f) for f in fileBasenames]
reducedFileIndex['FILENAME'] = reducedFilenames

# Find the breaks in observation procedure. These are candidates for group
# boundaries.
# 1) OBJECT changes
objectChange = (reducedFileIndex['OBJECT'][1:] != reducedFileIndex['OBJECT'][0:-1])

# 2) OBSTYPE changes
obstypeChange = (reducedFileIndex['OBSTYPE'][1:] != reducedFileIndex['OBSTYPE'][0:-1])

# 3) FILTER changes
filterChange = (reducedFileIndex['FILTER'][1:] != reducedFileIndex['FILTER'][0:-1])

# 4) EXPTIME changes
expTimeChange = (reducedFileIndex['EXPTIME'][1:] != reducedFileIndex['EXPTIME'][0:-1])

# 5) BINNING changes
binningChange = (reducedFileIndex['BINNING'][1:] != reducedFileIndex['BINNING'][0:-1])

# 6) Pointing changes
# Look for any pointing differences 1.5 degree (or more) for further separations
allPointings   = SkyCoord(
    reducedFileIndex['TELRA'],
    reducedFileIndex['TELDEC'],
    unit=(u.hour, u.degree)
)
medianDecs     = 0.5*(allPointings[1:].ra.to(u.rad) + allPointings[0:-1].ra.to(u.rad))
deltaDec       = allPointings[1:].dec - allPointings[0:-1].dec
deltaRA        = (allPointings[1:].ra - allPointings[0:-1].ra)*np.cos(medianDecs)
deltaPointing  = np.sqrt(deltaRA**2 + deltaDec**2)
pointingChange = deltaPointing > (1.5*u.deg)

# Identify all changes
allChanges = objectChange
allChanges = np.logical_or(allChanges, obstypeChange)
allChanges = np.logical_or(allChanges, filterChange)
allChanges = np.logical_or(allChanges, expTimeChange)
allChanges = np.logical_or(allChanges, binningChange)
allChanges = np.logical_or(allChanges, pointingChange)

# Assign a GROUP_ID for each group
groupBoundaries = np.hstack([0, np.where(allChanges)[0] + 1, allChanges.size])
groupIDs        = []
for i in range(groupBoundaries.size - 1):
    # Find the start and end indices of the group
    groupStartInd = groupBoundaries[i]
    groupEndInd   = groupBoundaries[i+1]

    # Build the gorup ID number
    groupID = i + 1

    # Count the number of images in this group
    numberOfImages = groupEndInd - groupStartInd

    # Build the list of ID numbers for THIS group and append it to the full list
    thisGroupID = numberOfImages*[groupID]
    groupIDs.extend(thisGroupID)

# Fill in the final entry
groupIDs.append(groupID)

# Store the groupID number in the reducedFileIndex
groupIDcolumn = Column(name='GROUP_ID', data=groupIDs)
reducedFileIndex.add_column(groupIDcolumn, index=2)

# Now remove any GROUPS with less than 8 images
groupIndex = reducedFileIndex.group_by('GROUP_ID')
goodGroupInds = []
groupInds = groupIndex.groups.indices
for startInd, endInd in zip(groupInds[:-1], groupInds[+1:]):
    # Count the number of images in this group and test if it's any good.
    if (endInd - startInd) >= 8:
        goodGroupInds.extend(range(startInd, endInd))


# Cull the reducedFileIndex to only include viable groups
goodGroupInds    = np.array(goodGroupInds)
reducedFileIndex = reducedFileIndex[goodGroupInds]

# Match a dither type for each group ("ABBA" or "HEX")
groupIndex = reducedFileIndex.group_by('GROUP_ID')
ditherType = []
for group in groupIndex.groups:
    # Count the number of images in this group
    numberOfImages = len(group)

    # Test if this is an ABBA or HEX dither
    if ('A' in group['AB']) and ('B' in group['AB']):
        ditherType.extend(numberOfImages*['ABBA'])
    if ('A' in group['AB']) and not ('B' in group['AB']):
        ditherType.extend(numberOfImages*['HEX'])

# Store the ditherNames number in the reducedFileIndex
ditherTypeColumn = Column(name='DITHER_TYPE', data=ditherType)
groupIndex.add_column(ditherTypeColumn, index=10)

# Identify meta-groups pointing at a single target with a single dither style.
targets = []
for group in groupIndex.groups:
    # Count the number of images in this group
    numberOfImages = len(group)

    # Get the group name
    groupName = np.unique(group['OBJECT'])[0]

    # Capitalize the group name to remove ambiguity
    groupName = groupName.upper()

    # Test it a target name occurs in this group name
    for target in targetList:
        if target in groupName:
            targets.extend(numberOfImages*[target])

            break
    else:
        import pdb; pdb.set_trace()
        raise ValueError('Gorup {} found no match in the target list'.format(groupName))

# Add the target identifications to the groupIndex table
targetColumn = Column(name='TARGET', data=targets)
groupIndex.add_column(targetColumn, index=5)

# Re-order by filename. Start by getting the sorting array
sortInds = groupIndex['FILENAME'].data.argsort()
reducedFileIndex = groupIndex[sortInds]

# Finally, add a column of "use" flags at the first index
useColumn = Column(name='USE', data=np.ones((len(reducedFileIndex),), dtype=int))
reducedFileIndex.add_column(useColumn, index=0)

# Save the index to disk.
reducedFileIndex.write(indexFile, format='ascii.csv', overwrite=True)

print('Done!')
