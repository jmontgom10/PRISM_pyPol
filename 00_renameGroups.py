import os
import sys
import pdb

# Add the AstroImage class
sys.path.append("C:\\Users\\Jordan\\Libraries\\python\\AstroImage")
from AstroImage import AstroImage

# This script will allow you to simply select a range of images and change the
# "OBJECT" keyword in their header to match the appropriate group name. Simply
# change the start and end file numbers for each grup in the dictionary.

# This is the location of all pyBDP data (index, calibration images, reduced...)
pyBDP_data    = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyBDP_data'

# This dictionary defines the night and first and last image for each group
groupDict = {'M78_V1':     {'night': 20150117, 'numbers': (513, 560)},
             'M78_V2':     {'night': 20150119, 'numbers': (241, 288)},
             'M78_R1':     {'night': 20150117, 'numbers': (561, 608)},
             'M78_R2':     {'night': 20150119, 'numbers': (289, 336)},
             'M82_V1':     {'night': 20150117, 'numbers': (657, 704)},
             'M82_V2':     {'night': 20150117, 'numbers': (705, 752)},
             'M82_V3':     {'night': 20150119, 'numbers': (385, 432)},
             'M82_V4':     {'night': 20150119, 'numbers': (433, 480)},
             'M82_V5':     {'night': 20150119, 'numbers': (577, 624)},
             'M82_R1':     {'night': 20150117, 'numbers': (609, 656)},
             'M82_R2':     {'night': 20150117, 'numbers': (753, 816)},
             'M82_R3':     {'night': 20150119, 'numbers': (337, 384)},
             'M82_R4':     {'night': 20150119, 'numbers': (481, 528)},
             'M82_R5':     {'night': 20150119, 'numbers': (529, 576)},
             'M104_V1':    {'night': 20150118, 'numbers': (517, 564)},
             'M104_V2':    {'night': 20150118, 'numbers': (661, 708)},
             'M104_V3':    {'night': 20150118, 'numbers': (709, 756)},
             'M104_V4':    {'night': 20150119, 'numbers': (625, 672)},
             'M104_V5':    {'night': 20150119, 'numbers': (769, 816)},
             'M104_R1':    {'night': 20150118, 'numbers': (565, 612)},
             'M104_R2':    {'night': 20150118, 'numbers': (613, 660)},
             'M104_R3':    {'night': 20150118, 'numbers': (757, 804)},
             'M104_R4':    {'night': 20150119, 'numbers': (673, 720)},
             'M104_R5':    {'night': 20150119, 'numbers': (721, 768)},
             'NGC1977_V1': {'night': 20150117, 'numbers': (209, 272)},
             'NGC1977_V2': {'night': 20150117, 'numbers': (337, 400)},
             'NGC1977_R1': {'night': 20150117, 'numbers': (273, 336)},
             'NGC1977_R2': {'night': 20150117, 'numbers': (401, 464)},
             'NGC2023_V1': {'night': 20150118, 'numbers': (325, 372)},
             'NGC2023_V2': {'night': 20150118, 'numbers': (421, 468)},
             'NGC2023_V3': {'night': 20150119, 'numbers': (97, 144)},
             'NGC2023_V4': {'night': 20150119, 'numbers': (193, 240)},
             'NGC2023_R1': {'night': 20150118, 'numbers': (373, 420)},
             'NGC2023_R2': {'night': 20150118, 'numbers': (469, 516)},
             'NGC2023_R3': {'night': 20150119, 'numbers': (49, 96)},
             'NGC2023_R4': {'night': 20150119, 'numbers': (145, 192)},
             'NGC7023_V1': {'night': 20150117, 'numbers': (145, 208)},
             'NGC7023_R1': {'night': 20150123, 'numbers': (85, 132)},
             'Orion_Cal_R_0117':  {'night': 20150117, 'numbers': (465, 488)},
             'Orion_Cal_V_0117':  {'night': 20150117, 'numbers': (489, 512)},
             'Taurus_Cal_V_0118': {'night': 20150118, 'numbers': (277, 300)},
             'Taurus_Cal_R_0118': {'night': 20150118, 'numbers': (301, 324)},
             'Orion_Cal_V_0119':  {'night': 20150119, 'numbers': (1, 24)},
             'Orion_Cal_R_0119':  {'night': 20150119, 'numbers': (25, 48)}}

pyBDP_reduced = os.path.join(pyBDP_data, 'pyBDP_reduced_images')

for group in groupDict.keys():
    print('Processing group {0}'.format(group))
    night    = '{0:d}'.format(groupDict[group]['night'])
    startNum, stopNum = groupDict[group]['numbers']
    # Loop through all the file numbers (be sure to include the LAST file!)
    for fileNum in range(startNum, stopNum + 1):
        # Construct the filename for this file
        strNum   = '{0:03d}'.format(fileNum)
        fileName = '.'.join([str(night), strNum, 'fits'])
        thisFile = os.path.join(pyBDP_reduced, fileName)

        if os.path.isfile(thisFile):
            # Read in the file
            thisImg = AstroImage(thisFile)

            # Modify the header to match the group name and re-write
            thisImg.header['OBJECT'] = group
            thisImg.write()

print('Done!')
