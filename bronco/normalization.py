##########################################################################
# NSAp - Copyright (C) CEA, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Perform T1, T2, FLAIR bias field, intensity normalization.

Statistical normalization techniques for magnetic resonance imaging,
Neuroimage Clin. 2014; 6: 9-19.
"""

# System import
from __future__ import print_function
import os

# Package import
from bronco import ANTSWrapper

# Third party import
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pygam import PoissonGAM
import nibabel
import numpy

# Define global parameters that can be tuned if necassary
BINS = {
    "t1": 2000,
    "t2": 300,
    "flair": 300
}


def whitestripe(
        inputfile,
        roifile,
        modality,
        outdir,
        maskfile=None,
        tail_remove_fraction=0.2,
        whitestripe_width=0.05,
        verbose=0):
    """ Compute the whitestripe location and statistical characteristics.

    Parameters
    ----------
    inputfile: str
        the input volume file.
    roifile: str
        the ROI volume file used to select a brain region for model fit.
    modality: str
        the input file modlaity: 't1', 't2' or 'flair'.
    outdir: str
        the destination folder.
    maskfile: str, default None
        the mask volume file.
    tail_remove_fraction: float, default 0.2
        define a tail threshold as a fraction of the maximum intensity
        occurence in the histogram.
    whitestripe_width: float
        control the size of the whitestripe around the extracted mode.
    verbose: int, default 0
        control the verbosity level.

    Returns
    -------
    whitestripefile: str
        the whitestripe location file.
    whitestripe_mean: float
        the mean whitestripe intensity.
    whitestripe_std: float
        the std whitestripe intensities.
    snaps: list of str
        some snaps to control the whitestripe steps.
    """
    # Check input parameters
    if modality not in ("t1", "t2", "flair"):
        raise ValueError("Unsupported '{0}' modality.".format(modality))
    whitestripe_width_l = whitestripe_width
    whitestripe_width_u = whitestripe_width

    # Create snap directory if necessary
    snapdir = os.path.join(outdir, "snap")
    if not os.path.isdir(snapdir):
        os.mkdir(snapdir)

    # Welcome message
    if verbose > 0:
        print("[info] Performing '{0}' WhiteStripe...".format(
            modality))

    # Mask input image, prevent inf and nan
    input_img = nibabel.load(inputfile)
    input_data = input_img.get_data()
    if maskfile is not None:
        mask_data = nibabel.load(maskfile).get_data()
        input_data[numpy.where(mask_data == 0)] = 0
    input_data[numpy.isnan(input_data)] = 0
    input_data[numpy.isinf(input_data)] = 0

    # Get histogram mode
    roi_data = nibabel.load(roifile).get_data()
    roi_indices = numpy.where((roi_data > 0) & (input_data > 0))
    input_roi_data = input_data[roi_indices]
    hist, bin_edges = numpy.histogram(input_roi_data, bins=BINS[modality])
    bin_centers = bin_edges[:-1] + numpy.diff(bin_edges) / 2.
    if modality == "t1":
        mode, snaps = get_last_mode(
            hist,
            bin_centers,
            remove_tail=True,
            remove_fraction=tail_remove_fraction,
            spline_order=5,
            snapdir=snapdir,
            verbose=verbose)
    else:
        mode, snaps = get_largest_mode(
            hist,
            bin_centers,
            spline_order=5,
            snapdir=snapdir,
            verbose=verbose)

    # White stripe estimation
    mode_q = numpy.mean(input_roi_data < mode)
    probs = (max(mode_q - whitestripe_width_l, 0) * 100.,
             min(mode_q + whitestripe_width_u, 1) * 100.)
    whitestripe = numpy.percentile(input_roi_data, probs)
    if verbose > 0:
        print("[info] Whitestripe: {0} - {1}.".format(probs, whitestripe))
    whitestripe_ind = numpy.where(
        (input_data > whitestripe[0]) & (input_data < whitestripe[1]))
    if len(whitestripe_ind[0]) == 0:
        print("[warn] Length of whitestripe is zero, using whole brain.")
        whitestripe_ind = numpy.where(input_data > numpy.mean(input_data))
    whitestripe_data = numpy.zeros(input_data.shape, dtype=int)
    whitestripe_data[whitestripe_ind] = 1
    out_roi_indices = numpy.where((roi_data <= 0) & (input_data > 0))
    whitestripe_data[out_roi_indices] = 0
    whitestripefile = os.path.join(outdir, "whitestripe.nii.gz")
    whitestripe_img = nibabel.Nifti1Image(
        whitestripe_data, affine=input_img.affine)
    nibabel.save(whitestripe_img, whitestripefile)

    # Stats
    whitestripe_mean = numpy.mean(input_data[whitestripe_ind])
    whitestripe_std = numpy.std(input_data[whitestripe_ind])
    if verbose > 0:
        print("[info] Whitestripe mean: {0}.".format(whitestripe_mean))
        print("[info] Whitestripe std: {0}.".format(whitestripe_std))
    
    return (whitestripefile, float(whitestripe_mean), float(whitestripe_std),
            snaps)


def whitestripe_norm(
        inputfile,
        whitestripefile,
        outdir,
        maskfile=None,
        verbose=0):
    """ Compute the whitestripe normalization.

    Taking the indices from white stripe to normalize the intensity values of
    the brain.

    Parameters
    ----------
    inputfile: str
        the input volume file.
    whitestripefile: str
        the whitestripe locations file.
    outdir: str
        the destination folder.
    maskfile: str, default None
        the mask volume file.
    verbose: int, default 0
        control the verbosity level.

    Returns
    -------
    whitestripe_normfile: str
        the whitestrip normalized volume file.
    whitestripe_mean: float
        the mean whitestripe intensity.
    whitestripe_std: float
        the std whitestripe intensities.
    """
    # Welcome message
    if verbose > 0:
        print("[info] Performing WhiteStripe normalization...")

    # Mask input image, prevent inf and nan
    input_img = nibabel.load(inputfile)
    input_data = input_img.get_data()
    if maskfile is not None:
        mask_data = nibabel.load(maskfile).get_data()
        input_data[numpy.where(mask_data == 0)] = 0
    input_data[numpy.isnan(input_data)] = 0
    input_data[numpy.isinf(input_data)] = 0

    # Load whitestripe locations
    whitestripe_data = nibabel.load(whitestripefile).get_data()
    whitestripe_ind = numpy.where(whitestripe_data > 0)

    # Moment matching
    whitestripe_mean = numpy.mean(input_data[whitestripe_ind])
    whitestripe_std = numpy.std(input_data[whitestripe_ind])
    if verbose > 0:
        print("[info] Whitestripe mean: {0}.".format(whitestripe_mean))
        print("[info] Whitestripe std: {0}.".format(whitestripe_std))
    input_data = (input_data - whitestripe_mean) / whitestripe_std

    # Save result
    whitestripe_normfile = os.path.join(outdir, "whitestripe_norm.nii.gz")
    norm_img = nibabel.Nifti1Image(input_data, affine=input_img.affine)
    nibabel.save(norm_img, whitestripe_normfile)

    return (whitestripe_normfile, float(whitestripe_mean),
            float(whitestripe_std))


def get_last_mode(
        hist,
        bin_centers,
        remove_tail=True,
        remove_fraction=0.2,
        spline_order=4,
        snapdir=None,
        verbose=0):
    """ Grabs the last peak or shoulder.

    Parameters
    ----------
    hist: array
        the values of the histogram. See density and weights for a
        description of the possible semantics.
    bin_centers: array of float
        the bin centers.
    remove_tail: bool, default True
        remove tail intensitites.
    remove_fraction: float, default 0.2
        define the threshold as a fraction of the maximum intensity
        occurence in the histogram.
    spline_order: int, default 4
        the gam fitted spline order.
    snapdir: str, default None
        a folder where QC images will be generated.
    verbose: int, default 0
        control the verbosity level.

    Returns
    -------
    last_mode: int
        the last mode in the histogram.
    snaps: list of str
        some snaps to control the whitestripe steps.
    """
    # Create snaps container
    snaps = []

    # Conditionning
    hist_factor = hist.max() * 0.1
    bin_centers_factor = bin_centers.max() * 0.1
    hist = hist.astype(float) / hist_factor
    bin_centers = bin_centers / bin_centers_factor

    # Remove rare intensity tail
    if remove_tail:
        if verbose > 0:
            print("[info] Remove tail ...")
        original_hist = hist
        original_bin_centers = bin_centers
        threshold = remove_fraction * max(bin_centers)
        bin_centers = original_bin_centers[original_hist > threshold]
        hist = original_hist[original_hist > threshold]
        fig = plt.figure()
        plt.plot(original_bin_centers, original_hist, "b-")
        plt.plot(bin_centers, hist + original_hist.max(), "g-")
        plt.title("Tail removal")
        if snapdir is not None:
            snaps.append(os.path.join(snapdir, "trail_remove.png"))
            fig.savefig(snaps[-1])
        if verbose > 1:
            plt.show()
    
    # Histogram smoothing: a penalized spline smoother
    if verbose > 0:
        print("[info] Histogram smoothing ...")
    fitted_hist = gam_smooth_hist(
        hist,
        bin_centers,
        spline_order=spline_order,
        verbose=verbose)

    # Peak extraction: calculate the relative extrema of data
    extrema = argrelextrema(fitted_hist, numpy.greater)[0]
    last_mode = max(bin_centers[extrema] * bin_centers_factor)
    if verbose > 0:
        print("[info] Relative extrema: {0}.".format(extrema))
        print("[info] Last mode: {0}.".format(last_mode))
    fig = plt.figure()
    plt.plot(bin_centers * bin_centers_factor, hist * hist_factor, "b-")
    plt.plot(bin_centers * bin_centers_factor,
             fitted_hist * hist_factor, "g--")
    plt.plot(bin_centers[extrema] * bin_centers_factor,
             fitted_hist[extrema] * hist_factor, "go")
    plt.plot(bin_centers[extrema[-1]] * bin_centers_factor,
             fitted_hist[extrema[-1]] * hist_factor, "ro")
    plt.title("Last peak")
    if snapdir is not None:
        snaps.append(os.path.join(snapdir, "last_peak.png"))
        fig.savefig(snaps[-1])
    if verbose > 1:
        plt.show()
    
    return last_mode, snaps


def get_largest_mode(
        hist,
        bin_centers,
        spline_order=4,
        snapdir=None,
        verbose=0):
    """ Grabs the largest peak of the histogram.

    Parameters
    ----------
    hist: array
        the values of the histogram. See density and weights for a
        description of the possible semantics.
    bin_centers: array of float
        the bin centers.
    spline_order: int, default 4
        the gam fitted spline order.
    snapdir: str, default None
        a folder where QC images will be generated.
    verbose: int, default 0
        control the verbosity level.

    Returns
    -------
    largest_mode: int
        the largest mode in the histogram.
    snaps: list of str
        some snaps to control the whitestripe steps.
    """
    # Create snaps container
    snaps = []

    # Conditionning
    hist_factor = hist.max() * 0.1
    bin_centers_factor = bin_centers.max() * 0.1
    hist = hist.astype(float) / hist_factor
    bin_centers = bin_centers / bin_centers_factor

    # Histogram smoothing: a penalized spline smoother
    if verbose > 0:
        print("[info] Histogram smoothing ...")
    fitted_hist = gam_smooth_hist(
        hist,
        bin_centers,
        spline_order=spline_order,
        verbose=verbose)

    # Peak extraction: calculate the relative extrema of data
    extrema = argrelextrema(fitted_hist, numpy.greater)[0]
    # minima = argrelextrema(fitted_hist, numpy.less)[0]
    # zerocrossing = sorted(extema + minima)
    largest_ind = extrema[numpy.argmax(hist[extrema])]
    largest_mode = bin_centers[largest_ind] * bin_centers_factor
    if verbose > 0:
        print("[info] Relative extema: {0}.".format(extrema))
        print("[info] Largest mode: {0} - {1}.".format(
            largest_ind, largest_mode))
    fig = plt.figure()
    plt.plot(bin_centers, hist, "b-")
    plt.plot(bin_centers, fitted_hist, "g--")
    plt.plot(bin_centers[extrema], fitted_hist[extrema], "go")
    plt.plot(bin_centers[largest_ind], fitted_hist[largest_ind], "ro")
    plt.title("Largest mode")
    if snapdir is not None:
        snaps.append(os.path.join(snapdir, "largest_mode.png"))
        fig.savefig(snaps[-1])
    if verbose > 1:
        plt.show()

    return largest_mode, snaps


def gam_smooth_hist(
        hist,
        bin_centers,
        spline_order=4,
        verbose=0):
    """ Uses a generalized additive model (GAM) to smooth a histogram.

    Parameters
    ----------
    hist: array
        the values of the histogram. See density and weights for a
        description of the possible semantics.
    bin_centers: array of float
        the bin centers.
    spline_order: int, default 4
        the gam fiited spline order.

    Returns
    -------
    fitted_hist: array
        the fitted values from GAM.
    """
    gam = PoissonGAM(
        spline_order=spline_order,
        n_splines=25)
    gam = gam.gridsearch(
        bin_centers,
        hist,
        return_scores=False)
    fitted_hist = gam.predict(bin_centers)
    if verbose > 0:
        print("[info] GAM summary:")
        print(gam.summary())

    return fitted_hist


def intersect_masks(maskfiles, outdir, verbose=0):
    """ Intersect a list of mask.

    Parameters
    ----------
    maskfiles: list of str
        the list of masks to be intersected.
    outdir: str
        the destination folder.
    verbose: int, default 0
        control the verbosity level.

    Returns
    -------
    whitestripefile: str
        the whitestripe location file.
    """
    # Check inputs
    if len(maskfiles) == 0:
        raise ValueError("At least one mask file is expected.")
    
    # Intersect mask
    whitestripe_im = nibabel.load(maskfiles[0])
    whitestripe_data = whitestripe_im.get_data()
    for path in maskfiles[1:]:
        mask_data = nibabel.load(path).get_data()
        whitestripe_data[numpy.where(mask_data == 0)] = 0
    whitestripefile = os.path.join(outdir, "whitestripe.nii.gz")
    whitestripe_img = nibabel.Nifti1Image(
        whitestripe_data, affine=whitestripe_im.affine)
    nibabel.save(whitestripe_img, whitestripefile)

    return whitestripefile
    

def n4_bias_field_correction(
        inputfile,
        outdir,
        method="ants",
        maskfile=None,
        weightfile=None,
        nb_iterations=50,
        convergence_threshold=0.001,
        bspline_grid_resolution=(1, 1, 1),
        shrink_factor=1,
        bspline_order=3,
        histogram_sharpening_parameters=(0.15, 0.01, 200),
        verbose=0):
    """ Performs MRI bias correction using N4 algorithm.

    This module is based on the ITK filters or on ANTS depending of the
    input parameters.

    Parameters
    ----------
    inputfile: str
        Input image where you observe signal inhomegeneity.
    outdir: str
        The destination folder.
    method: str, default 'ants'
        The normalization software to be used: 'itk' or 'ants'.
    maskfile:  str, default None
        Binary mask that defines the structure of your interest. If the mask
        is not specified, the module will use internally Otsu thresholding
        to define this mask. Better processing results can often be obtained
        when a meaningful mask is defined.
    weightfile: str, default None
        Weight image.
    nb_iterations: int, default 50
        Maximum number of iterations at each level of resolution. Larger
        values will increase execution time, but may lead to better results.
    convergence_threshold: float, default 0.001
        Stopping criterion for the iterative bias estimation. Larger values
        will lead to smaller execution time.
    bspline_grid_resolution: int, default (1, 1, 1)
        Resolution of the initial bspline grid defined as a sequence of three
        numbers. The actual resolution will be defined by adding the bspline
        order (default is 3) to the resolution in each dimension specified
        here. For example, 1,1,1 will result in a 4x4x4 grid of control points.
        This parameter may need to be adjusted based on your input image.
        In the multi-resolution N4 framework, the resolution of the bspline
        grid at subsequent iterations will be doubled. The number of
        resolutions is implicitly defined by Number of iterations parameter
        (the size of this list is the number of resolutions).
    shrink_factor: int, default 1
        Defines how much the image should be upsampled before estimating the
        inhomogeneity field. Increase if you want to reduce the execution
        time. 1 corresponds to the original resolution. Larger values will
        significantly reduce the computation time.
    bspline_order: int, default 3
        Order of B-spline used in the approximation. Larger values will lead
        to longer execution times, may result in overfitting and poor result.
    histogram_sharpening_parameters: 3-uplate, default (0.15, 0.01, 200)
        A vector of up to three values. Non-zero values correspond to Bias
        Field Full Width at Half Maximum, Wiener filter noise, and Number of
        histogram bins.
    verbose: int, default 0
        Control the verbosity level.

    Returns
    -------
    bias_field_corrected_file: str
        Result of processing.
    bias_field_file: str
        Result of processing. 
    """
    # Check input parameters
    if method not in ("itk", "ants"):
        raise ValueError("'{0}' is not a valid normalization "
                         "software.".format(method))

    # Create a binary mask
    if maskfile is not None:
        mask_file = os.path.join(outdir, "mask.nii.gz")
        input_img = nibabel.load(inputfile)
        img = nibabel.load(maskfile)
        data = img.get_data()
        data[numpy.where(data > 0)] = 1.
        data = data.astype(numpy.uint8)
        binary_img = nibabel.Nifti1Image(data, affine=input_img.affine)
        nibabel.save(binary_img, mask_file)

    # ITK method
    if method == "itk":

        # Read the input volume
        inputImage = sitk.ReadImage(inputfile)

        # Read/compute the input volume mask
        if maskfile is not None:
            maskImage = sitk.ReadImage(mask_file)
        elif weightfile is not None:
            maskImage = sitk.ReadImage(weightfile)
        else:
            maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200 )
            mask_file = os.path.join(outdir, "mask.nii.gz")  
            sitk.WriteImage(maskImage, mask_file)

        # Shrink image/mask volume if requested
        if shrink_factor != 1:
            inputImage = sitk.Shrink(
                inputImage, [int(shrink_factor)] * inputImage.GetDimension())
            maskImage = sitk.Shrink(
                maskImage, [int(shrink_factor)] * inputImage.GetDimension())
         
        # Cast the input volume
        inputImage = sitk.Cast(inputImage, sitk.sitkFloat32 )

        # Create filter
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetBiasFieldFullWidthAtHalfMaximum(
            histogram_sharpening_parameters[0])
        corrector.SetWienerFilterNoise(histogram_sharpening_parameters[1])
        corrector.SetNumberOfHistogramBins(histogram_sharpening_parameters[2])
        corrector.SetSplineOrder(bspline_order)
        corrector.SetMaximumNumberOfIterations([int(nb_iterations)] * 4)
        corrector.SetConvergenceThreshold(convergence_threshold)
        bspline_grid_resolution = [
            bspline_order + e for e in bspline_grid_resolution]
        corrector.SetNumberOfControlPoints(bspline_grid_resolution)

        # Execute the filter
        output = corrector.Execute(inputImage, maskImage)
        
        # Write the result
        bias_field_corrected_file = os.path.join(
            outdir, "bias_corrected.nii.gz")
        bias_field_file = None
        sitk.WriteImage(output, bias_field_corrected_file)

    # ANTS method
    else:

        # Get command line parameter
        img = nibabel.load(inputfile)
        data = img.get_data()
        ndim = data.ndim
        bias_field_corrected_file = os.path.join(
            outdir, "bias_corrected.nii.gz")
        bias_field_file = os.path.join(
            outdir, "bias_field.nii.gz")

        # Create command
        bspline_grid_resolution = [
            str(e) for e in bspline_grid_resolution]
        histogram_sharpening_parameters = [
            str(e) for e in histogram_sharpening_parameters]
        cmd = [
            "N4BiasFieldCorrection",
            "-d", str(ndim),
            "-i", inputfile,
            "-s", str(shrink_factor),
            "-b", "[{0}, {1}]".format(
                "x".join(bspline_grid_resolution), bspline_order),
            "-c", "[{0}, {1}]".format(
                "x".join([str(nb_iterations)] * 4), convergence_threshold),
            "-t", "[{0}]".format(
                ", ".join(histogram_sharpening_parameters)),
            "-o", "[{0}, {1}]".format(
                bias_field_corrected_file, bias_field_file)]
        if maskfile is not None:
            cmd += ["-x", mask_file]
        if weightfile is not None:
            cmd += ["-w", weightfile]
        cmd += ["-v"]

        # Execute command
        process = ANTSWrapper(verbose=verbose)
        process(cmd)
        # TODO: remove
        # with subprocess the field map is not generated and no error is raised
        os.system(" ".join(cmd[:-1]))

    return bias_field_corrected_file, bias_field_file
