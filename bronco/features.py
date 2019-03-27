##########################################################################
# NSAp - Copyright (C) CEA, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Generate features based on interest masks using pyradiomics.
"""

# System import
import os
import glob

# Package import
from .pyradiomics import BroncoRadiomicsFeaturesExtractor
from radiomics import featureextractor, getTestCase
# Third party import
import nibabel
import numpy


def merge_rois(roifiles, outdir, basemaskname="merged_mask.nii.gz"):
    """ Merge the input ROIs.

    Parameters
    ----------
    roifiles: list of str
        the ROI images to be merged.
    outdir: str
        the destination folder.
    fmaskname: str
        the filename of the mask to be saved.

    Returns
    -------
    merged_file: str
        the merged ROI.
    """
    # Check inputs
    if len(roifiles) == 0:
        raise ValueError("Expect at least one ROI.")
    if not os.path.isdir(outdir):
        raise ValueError(
            "'{0}' is not a valid output directory.".format(outdir))

    # Merge ROIS
    im = nibabel.load(roifiles[0])
    ref_affine = im.affine
    merged_data = im.get_data()
    if merged_data.ndim != 3:
        raise ValueError("Expect 3d-ROIS.")
    if len(roifiles) > 1:
        for path in roifiles[1:]:
            im = nibabel.load(path)
            affine = im.affine
            if not numpy.allclose(ref_affine, affine):
                raise ValueError("ROI files have different affine matrices.")
            data = im.get_data()
            if data.ndim != 3:
                raise ValueError("Expect 3d-ROIS.")
            merged_data += data
    merged_data[merged_data > 0] = 1

    # Save the result
    merged_file = os.path.join(outdir, basemaskname)
    im = nibabel.Nifti1Image(merged_data, affine=ref_affine)
    nibabel.save(im, merged_file)

    return merged_file


def extract_features_from_roi(inputfile, maskfile, outdir, configfile=None):
    """ Extract common features using pyradiomics on a given ROI.

    Parameters
    ----------
    inputfile: str
        the input image.
    maskfile: str
        the mask image that defines the ROi of interest.
    outdir: str
        the destination folder.
    configfile: str, default None
        the radiomics configuration file.

    Returns
    -------
    features: dict
        the generated features.
    snaps: list of str
        the intermediate results.
    """

    #Set the class outdir parameter
    snapdir = os.path.join(outdir, "snaps")
    if not os.path.isdir(snapdir):
        os.mkdir(snapdir)
    BroncoRadiomicsFeaturesExtractor.outdir = snapdir

    # Create the feature extractor
    if configfile is None:
        extractor = BroncoRadiomicsFeaturesExtractor()
    else:
        extractor = BroncoRadiomicsFeaturesExtractor(configfile)

    # Extract the features
    features = extractor.execute(inputfile, maskfile)

    # Get intermediate results
    snaps = glob.glob(os.path.join(snapdir, "*.*"))

    return features, snaps


def extract_features_from_roi_212(inputfile, maskfile, outdir, configfile=None, 
    resampledpixelspacing=None, binwidth = None):
    """ Extract common features using pyradiomics on a given ROI using
        pyradiomics 2.1.2

    Parameters
    ----------
    inputfile: str
        the input image.
    maskfile: str
        the mask image that defines the ROi of interest.
    outdir: str
        the destination folder.
    configfile: str, default None
        the radiomics configuration file.
    resampledpixelspacing: str, default None
        the resampled isotropic-voxel dimension 
    binwidth: str, default None
        the bin width;

    Returns
    -------
    features: dict
        the generated features.
    """

    #Initialize the parameters
    if resampledpixelspacing != None or binwidth != None:
        params = {}
        if resampledpixelspacing != None:
            voxel_dim = [int(resampledpixelspacing)] * 3
            params['resampledPixelSpacing'] = voxel_dim
        if  binwidth != None:
            params['binwidth'] = binwidth
        extractor = featureextractor.RadiomicsFeaturesExtractor(**params)
    else:
        # Create the feature extractor
        if configfile is None:
            extractor = featureextractor.RadiomicsFeaturesExtractor()
        else:
            extractor = featureextractor.RadiomicsFeaturesExtractor(configfile)

    #Add the wavelets
    wave_params = {"wavelet": "coif1", "start_level": 0, "level":1}
    extractor.enableImageTypes(Wavelet = wave_params)

    # Extract the features
    features = extractor.execute(inputfile, maskfile)

    return features 
