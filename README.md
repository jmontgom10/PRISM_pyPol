# PRISM_pyPol
Polarimetric reduction of astronomical \*.fits images.

# Dependencies and Setup

pyPol requires installation of the "AstroImage" class. This can be downloaded
from the jmontgom10 GitHub repository and has its own set of dependencies, which
the user should take care to install properly (see the AstroImage README File a
guide on install those packages).

I recommend using the Anaconda environment, as that comes with numpy, scipy,
astropy, and matplotlib preinstalled. If you elect not to use Anaconda, then
make sure to get those packages properly installed before proceeding to install
the AstroImage dependencies.

You will also need to install the `photutils` (and optional `astroquery` package
for photometric calibration) for Python. This can be done via anaconda

`conda install --channel https://conda.anaconda.org/astropy photutils`
`conda install -c astropy astroquery`

or pip (if you did not elect to install Anaconda)

`pip install --no-deps photutils`
`pip install astroquery`

This script package is written for Python 3 and is not compatible with Python 2.

# Procedure

The pyPol package consists of a set of scripts, each of which perform one of the
basic steps in the reduction of polarimetry CCD data. The script names are all
prepended by a number, and the scripts should be executed in the order of their
numbers, as in

```
$ python 01_buildIndex.py
```

*Nota Bene* The user must always take care to update the directory structure
information at the top of each script. Technically, the user can elect to use
whatever directory structure they like, but I recommend using a similar
directory structure to the one shown in the default values. These help keep the
intermittent data cleanly separated into logical directories.

## 01_buildIndex.py

The first step in this reduction procedure is to simply look at all the files in
the "reducedDir" directory and categorize them according to their content.
Waveband and polaroid filter rotation angle can be pulled straight out of the
image header, but some information must be provided by the user. The two
essential user-provided pieces of information are the group "target" and the
group "dither pattern."

*Use a single name to refer to each individual target*. The purpose of the
"target" information is to provided the scripts with an easy way to tell what
images should be treated as repeated measurements of the same object. For
example, the scripts cannot parse that the group names "NGC 7023_R1",
"ngc7023R2", and "NGC_7023V_1" all refer to the same target. Thus, the user
should enter an identical string (e.g. "NGC7023") for each of these groups.
Fortunately, the scripts **are** able to parse the waveband of each image, so
you should not use different target names for different wavelengths. For
example, do **not** use "NGC7023_R" to refer to R-band observations of NGC7023
and "NGC7023_V" to refer to V-band observations of NGC7023, use "NGC7023" to
refer to both, and the scripts will figure out which observations are R-band vs.
V-band. Of course, if you make a mistake, the file index will be saved as a .csv
file, so you can easily edit the file using your favorite .csv editor (excel,
open office, etc...).

The indexing script also adds a "Use" flag column to the index. Files with a 1
in this column will be used in the final polarimetry analysis while files with a
0 in this column will be omitted from later processing steps. This provides a
simple way to prevent any bad data from being included in the final polarization
estimates.

### Target Names for Calibration Data

I recommend treating calibration observations from separate nights
independently. The easiest way to handle this is to append the data to the
Target name for calibration images. Thus `Orion_Cal` targets from different
nights become `Orion_Cal_20150117` and `Orion_Cal_20150119`, etc... The
target-parser in the 05_avgPolAngImages.py will then know to treat these
separately when computing average images.

## 02_doAstrometry.py

This step can be optional depending on how you want to align your images in step
05_avgPolAngImages and beyond. If you are planning on using astrometry, then
you'll need to install Astrometry.net on your system. If you are using Linux,
then the Astrometry.net website is probably enough to get going. If you need a
bit more help, or if you're installing Astrometry.net on Windows, then I
recommend you check out the [setup
guide](https://sites.google.com/site/jmastronomy/Software/astrometry-net-setup)
I wrote expressly for that purpose.

Once you've successfully installed Astrometry.net, this step is quite
straightforward. Simply update the paths at the beginning of the script, and
execute it with Python. This should loop through all the files in your
reducedDir location and solve their astrometry (updating file headers along the
way).

## 03_repairAstrometry.py

Of course, it would be wonderful if all the files had perfect astrometry right
out of the gate, but that's not always the case. To check the self-consistency
of the solved astrometry, this script produces a histogram of the plate-scale
values for each science target and each waveband (e.g. NGC7023, R-band). Any
outliers in the histogram are identified and their astrometry information is
replaced with a substitute set of values from the other images in that group.

To execute this script, simply update the paths at the top of the file, and run it through the Python interpreter.

## 04_buildMasks.py

This script allows the user to manually generate masks for each science image.
The reflection nebula project for which the pyPol scripts were originally
developed included several targets with immensely bright stars at their center.
These bright stars produced optical ghosts on the images, and those ghosts
needed to be masked in order to accurately asses the polarization across the
nebula. If for whatever reason you also need to mask certain portions of the
your science images, then you can make those masks with this script.

This script requires the usual updating of the directories and filenames at the
top of the code, but it also requires the user to supply a list of targets for
which to build masks. Any images with a target value matching an entry in the
`targets` variable will be processed for mask building.

When the script is launched, a three pane plot of the science images will be
displayed. The center pane in this triptych is the active pane for which you are
build a mask. I have provided the following look-up table for managing the
mask-building process.

| Event       | Effect                                                         |
|-------------|----------------------------------------------------------------|
|Left Click   | Apply a circular aperture mask to the clicked region           |
|Right Click  | Delete a circular aperture mask from the clicked region        |
| 1 - 6       | Set the size of the circular aperture                          |
| Enter       | Save the current mask for the center pane to disk              |
| Backspace   | Reset the current mask to a blank slate                        |
| Left/Right  | Change the active pane to the previous/next image              |

To build a mask, simply click on the regions of the active pane which need to be
masked. The mask will be displayed in the center pane as a transparent white
outline. You can delete regions of the mask with right clicks. Once you are
satisfied with the mask you've build, you can save it to disk with a single
stroke of the Enter key. Press the left or right arrow keys to scroll through
the images for the specified target(s), and simply close the GUI plot to end the
script.

## 05_avgPolAngImages.py

This script simply loops through each target at each waveband and polaroid
filter rotation angle and computes an average image for each of these. To run
this script, simply update the directory information at the top of the script,
and run it through a python interpreter.

The following procedure is used to compute the average images.

1. The off (B) positions from the ABBA dither observational procedure are used
to generate a smoothed average background image.

2. The smoothed background images are subtracted from the on (A) positions.

3. The background subtracted on-target (A position) images are aligned with each
other.  By default, the astrometry from 02_doAstrometry is used. However, it is
possible to use a cross-correlation process to align the images before
averaging. The cross-correlation process would run *much* slower.

4. The aligned images are averaged (rejecting outliers along the way) to produce
a single output image for each polaroid-angle (in near future: aligned across
all available wavebands s that all images of that target will be aligned
regardless of the waveband.)

5. The astrometry of the averaged image is solved using Astrometry.net and saved
to disk. See my instructions for [Windows
Setup](https://sites.google.com/site/jmastronomy/Software/astrometry-net-setup)
to get the Astrometry.net solver running on your computer. A similar procedure
can be followed with much less effort in Linux. Simply install the pre-requisite
packages using `apt-get` or `yum` instead of Cygwin-setup.

## 06_polarizationCalibration.py

This script computes the polarization position angle offsets to be applied to
correct the polaroid-rotation-angle reference frame to equatorial coordinates.
It will also compute a single polarization efficiency to be used to correct the
computed polarization percentage maps.

This procedure requires the observation of polarization standard stars and their
standard polarization percentages and polarization position-angles. This
information must be supplied by the user, using the 'Target' value supplied for
these observations in the 01_buildIndex script.

## 07_finalPolarimetry.py

This script uses a cross-correlation method to align the averaged
polaroid-rotation-angle images for each target and waveband, then it performs
the necessary subtraction and division methods to compute polarization
percentage and polarization position angle maps for the extended emission in
each image.

This script should ???

## 08a_selectPhotStars.py

Automatically retrieves USNO-B1.0 data from Vizier service.

Displays stars on top of mean V- and R-band image (cropped to the region common
to *both* bands).

User selects which stars are isolated and clearly identified in the USNO-B1.0
catalog. These will be saved for use in the V- and R-band photometric
calibration.

## 08b_photometricCalibration.py

For V-band references, use the APASS data, which includes a direct measure of
the Johnson_V-band photometry across most of the sky. APASS data can be
downloaded from https://www.aavso.org/download-apass-data.

For R-band references, use the USNO-B1.0 data, which contains R1 and R2 "red
magnitude" estimates, which closely approximate the Johnson_R-band. The
USNO-B1.0 data can be retrieved from
http://www.usno.navy.mil/USNO/astrometry/optical-IR-prod/icas/fchpix.
