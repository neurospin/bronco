#! /usr/bin/env python
##########################################################################
# NSAp - Copyright (C) CEA, 2013 - 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
#
# Code V Frouin
##########################################################################


"""
Monkey patch on pyradiomics module.
"""

# System import
import os
import collections

# Third party import
import numpy as np
import six
import matplotlib.pyplot as plt
import SimpleITK
from radiomics.featureextractor import RadiomicsFeaturesExtractor


class BroncoRadiomicsFeaturesExtractor(RadiomicsFeaturesExtractor):
    outdir = None

    def computeFeatures(self, image, mask, inputImageName, **kwargs):

        #######################################################################
        # Original code

        featureVector = collections.OrderedDict()

        # Calculate feature classes
        for featureClassName, enabledFeatures in six.iteritems(
                self.enabledFeatures):
            # Handle calculation of shape features separately
            if featureClassName == 'shape':
                continue

            if featureClassName in self.getFeatureClassNames():
                self.logger.info('Computing %s', featureClassName)
                featureClass = self.featureClasses[featureClassName](image,
                                                                     mask,
                                                                     **kwargs)

            if enabledFeatures is None or len(enabledFeatures) == 0:
                featureClass.enableAllFeatures()
            else:
                for feature in enabledFeatures:
                    featureClass.enableFeatureByName(feature)

            featureClass.calculateFeatures()
            for (featureName, featureValue) in six.iteritems(
                    featureClass.featureValues):
                newFeatureName = "%s_%s_%s" % (inputImageName,
                                               featureClassName, featureName)
                featureVector[newFeatureName] = featureValue

            ###################################################################
            # Supplementary code to create snapshots for GCLM

            if featureClassName == "glcm":

                # Save ROI, binned ROI and mask as Nifti
                roi_image = SimpleITK.GetImageFromArray(featureClass.imageArray)
                bin_roi_image = SimpleITK.GetImageFromArray(featureClass.matrix)
                # mask_array = SimpleITK.GetImageFromArray(featureClass.maskArray)
                mask_image = featureClass.inputMask
                path_roi = os.path.join(self.outdir, "roi_%s.nii.gz" % (featureClassName))
                path_bin_roi = os.path.join(self.outdir, "binned_roi_%s.nii.gz" %  (featureClassName))
                path_mask = os.path.join(self.outdir, "mask_%s.nii.gz" %  (featureClassName))
                SimpleITK.WriteImage(roi_image, path_roi)
                SimpleITK.WriteImage(bin_roi_image, path_bin_roi)
                SimpleITK.WriteImage(mask_image, path_mask)

                # subplots: one histogram + co-occurences matrices
                nb_coocc_matrices = featureClass.P_glcm.shape[2]
                nb_subplots = 1 + nb_coocc_matrices
                fig, axes = plt.subplots(nrows=1, ncols=nb_subplots,
                                         figsize=(18, 2))
                histo_ax, matrices_axes = axes[0], axes[1:]
                fig.suptitle("GLCM matrices, image type: %s, bin width: %i"
                             % (inputImageName, featureClass.binWidth))

                # Histogram
                #bins = featureClass.binEdges # binEdges are in real data level
                bins = range(1, featureClass.coefficients['Ng']+1)
                # this hist consider all voxels in the bounding box
                # histo_ax.hist(featureClass.matrix.flatten(), bins=bins)
                # this hist consider voxel within the ROI
                histo_ax.hist(
                    featureClass.matrix[np.where(featureClass.maskArray != 0)],
                    bins=bins)
                histo_ax.tick_params(labelsize=3)
                histo_ax.set_title("%s hist" % inputImageName, fontsize=8)

                # Identify global min/max of concurrent matrices to have a
                # consistent coloration across all images
                co_min = featureClass.P_glcm.min()
                co_max = featureClass.P_glcm.max()
#                print(featureClass.P_glcm.shape )

                # Create image subplot for each matrix along with colorbar
                extent = [bins[0], bins[-1], bins[0], bins[-1]]
                for i, ax in enumerate(matrices_axes):
                    co_matrix = featureClass.P_glcm[:, :, i]
                    im = ax.imshow(co_matrix, vmin=co_min, vmax=co_max,
                                   extent=extent, cmap="Reds",
                                   interpolation='nearest')
                    ax.tick_params(labelsize=3)
                    ax.set_title("angle index: %i" % i, fontsize=6)
                    cb = plt.colorbar(im, ax=ax, orientation="horizontal")
                    cb.ax.tick_params(labelsize=3)
                fig.tight_layout()
                name_png = '%s_%s_bw%s.png' % (featureClassName,
                                               inputImageName,
                                               featureClass.binWidth)
                path_png = os.path.join(self.outdir, name_png)
                plt.savefig(path_png, dpi=300)

            if featureClassName == "glrlm":
                nb_coocc_matrices = featureClass.P_glrlm.shape[2]
                nb_subplots = 1 + nb_coocc_matrices
                fig, axes = plt.subplots(nrows=1, ncols=nb_subplots,
                                         figsize=(18, 2))
                histo_ax, matrices_axes = axes[0], axes[1:]
                fig.suptitle("GLCM matrices, image type: %s, bin width: %i"
                             % (inputImageName, featureClass.binWidth))
                # Identify global min/max of concurrent matrices to have a
                # consistent coloration across all images
                co_min = featureClass.P_glrlm.min()
                co_max = featureClass.P_glrlm.max()

                # Create image subplot for each matrix along with colorbar
                extent = [1, featureClass.P_glrlm[:,:,0].shape[1], featureClass.coefficients['Ng'], 1] 
                for i, ax in enumerate(matrices_axes):
                    co_matrix = featureClass.P_glrlm[:, :, i]
                    im = ax.imshow(co_matrix, vmin=co_min, vmax=co_max,
                                   extent=extent, cmap="Reds",
                                   interpolation='nearest')
                    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/10.)
                    ax.tick_params(labelsize=3)
                    ax.set_title("angle index: %i" % i, fontsize=6)
                    cb = plt.colorbar(im, ax=ax, orientation="horizontal")
                    cb.ax.tick_params(labelsize=3)
                fig.tight_layout()
                name_png = '%s_%s_bw%s.png' % (featureClassName,
                                               inputImageName,
                                               featureClass.binWidth)
                path_png = os.path.join(self.outdir, name_png)
                plt.savefig(path_png, dpi=300)
                
        return featureVector
