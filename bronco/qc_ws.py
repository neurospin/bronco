# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:25:17 2017

@author: vf140245
Copyrignt : CEA NeuroSpin - 2017
"""
import numpy as np
import argparse
import os.path
from glob import glob
import seaborn as sns

import nibabel as ni
from scipy.ndimage.morphology import binary_erosion
from nilearn import plotting
import matplotlib.pyplot as plt

doc = """
Command qc_ws: plot distrib of standardized NAWM

python ~/gits/bronco/bronco/qc_ws.py \
       -i /neurospin/radiomics/studies/metastasis/nsap
       -s subject_id
       -r /tmp/statistics.npy
       -t 3

To create a global distribution result : /tmp/wsq/QC_WS_{'' or 'hybrid'}_{'t1' or 'flair'}.npz
The distribution is created in dir specified with -r option
python ~/gits/bronco/bronco/qc_ws.py \
    -i, /neurospin/radiomics/studies/metastasis/nsap \
    -o, /tmp/wsq/ \
    -t, 3 \
    -r, /tmp/wsq/ \
    -m, hybrid \
    --seq, t1 \
    -p

To create the plots for one subjects.
The distribution is read from dir specified with -r option

python ~/gits/bronco/bronco/qc_ws.py \
    -i, /neurospin/radiomics/studies/metastasis/nsap \
    -o, /tmp/wsq/ \
    -r, /tmp/wsq/ \
    -s, 187962757123 \
    -m, hybrid \
    --seq, t1
"""
parser = argparse.ArgumentParser(description=doc)
parser.add_argument(
    "-v", "--verbose", dest="verbose", type=int, choices=[0, 1, 2], default=0,
    help="increase the verbosity level: 0 silent, [1, 2] verbose.")
parser.add_argument(
    "-m", "--mode", dest="mode", type=str, choices=['mono', 'hybrid'],
    required=True, default=0,
    help="Choose WS mode mono or hybrid.")
parser.add_argument(
    "--seq", dest="seq", type=str, choices=['t1', 'flair'],
    required=True, default=0,
    help="Choose WS targeted image t1 or flair.")
parser.add_argument(
    "-i", "--in", dest="indir", metavar="FILE",
    help="Entry directory to parse")
parser.add_argument(
    "-r", "--ref", dest="refstat", metavar="FILE",
    help="Entry file to read/write reference statistics")
parser.add_argument(
    "-p", "--preprocess", dest="preprocess", action='store_true',
    help="parse subject available to build the DB of distributions")
parser.add_argument(
    "-s", "--subject", dest="subject", default=None,
    help="Subject id")
parser.add_argument(
    "-t", "--iterations", dest="iterations", default=0, metavar="int",
    help="Num of iterations")
parser.add_argument(
    "-o", "--out", dest="outfile", metavar="FILE",
    help="Image file that will contain the snap result")


def erode_mask(mask_image, iterations=1):
    """ Erode a binary mask file.

    Parameters
    ----------
    mask_image: Nifti image
        the mask to erode.
    iterations: int (optional, default 1)
        the number of path for the erosion.
    white_thresh: float (optional, default 1.)
        threshold to apply to mask_image.

    Returns
    -------
    erode_mask: Nifti image
        the eroded binary Nifti image.
    """
    # specific case
    if iterations == 0:
        return mask_image
    # Generate structural element
    structuring_element = np.array(
        [[[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]],
         [[0, 1, 0],
          [1, 1, 1],
          [0, 1, 0]],
         [[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]]])

    # Erode source mask
    source_data = mask_image.get_data()
    erode_data = binary_erosion(source_data, iterations=iterations,
                                structure=structuring_element)
    erode_data = erode_data.astype(source_data.dtype)
    erode_mask = ni.Nifti1Image(erode_data, mask_image.get_affine())

    return erode_mask

# test existence of a list of files
def files_exist(files):
    retval = True
    for f in files:
        if not os.path.exists(f):
            retval = False
            print f
            break

    return retval

def main(spyder_inter):
    # read args
    if spyder_inter:
        args = parser.parse_args(
            ['-i', '/neurospin/radiomics/studies/metastasis/nsap',
              '-o', '/tmp/wsq/',
              '-t', '3',
              '-r', '/tmp/wsq/',
              '-s', '187962757123',
              '-m', 'hybrid',
              '--seq', 't1',
              #'-p',
              ])
    else:
        args = parser.parse_args()

    # init from args
    subjects = [os.path.basename(fn)
                    for fn in glob(os.path.join(args.indir, 'brain', '*'))
                        if (os.path.basename(fn)[0].isdigit())]
    sel_s = args.subject
    print sel_s
    if args.mode == 'mono':
        extension_dir = args.seq
    else:
        extension_dir = args.mode + '_' + args.seq

    # needed init
    white_thresh = 0.9
    gray_thresh = 0.9
    white_before_hist = []
    white_after_hist = []
    gray_before_hist = []
    gray_after_hist = []
    list_subjects = []
    log_error_subjects = []

    # create the distributions from all subjects
    if args.preprocess:
        for i, s in enumerate(subjects):
            # for code testing
            if (sel_s is not None) and (sel_s not in s):
                continue

            # Progression
            print s, ' ', (i+1), '/', len(subjects)
            # files to consider
            before_ws = os.path.join(
                            args.indir,
                            'whitestripe_{}'.format(extension_dir),
                            '{}'.format(s),
                            'n4', 'bias_corrected.nii.gz')
            after_ws = os.path.join(
                            args.indir,
                            'whitestripe_{}'.format(extension_dir),
                            '{}'.format(s),
                            'whitestripe', 'whitestripe_norm.nii.gz')
            background = os.path.join(
                            args.indir, '..', 'base',
                            '{}'.format(s), 'anat',
                            '{}_enh-gado_T1w.nii.gz'.format(s))
            mask_gray = os.path.join(
                            args.indir, 'segmentation', '{}'.format(s),
                            'fast', 'brain_pve_1.nii.gz')
            mask_white = os.path.join(
                            args.indir, 'segmentation', '{}'.format(s),
                            'fast', 'brain_pve_2.nii.gz')
            if not files_exist([before_ws, after_ws, background,
                                mask_gray, mask_white]):
                print '{}: files missing'.format(s)
                log_error_subjects.append('{}: files missing'.format(s))
                continue

            # load data
            before_ws_data = ni.load(before_ws).get_data()
            after_ws_data = ni.load(after_ws).get_data()
            # load masks
            mask_white_img = erode_mask(ni.load(mask_white),
                                        iterations=int(args.iterations))
            mask_gray_img = erode_mask(ni.load(mask_gray),
                                       iterations=1)
            # get histograms
            white_before_hist.append(
                    before_ws_data[mask_white_img.get_data() > white_thresh])
            white_after_hist.append(
                    after_ws_data[mask_white_img.get_data() > white_thresh])
            gray_before_hist.append(
                    before_ws_data[mask_gray_img.get_data() > gray_thresh])
            gray_after_hist.append(
                    after_ws_data[mask_gray_img.get_data() > gray_thresh])
            list_subjects.append(s)

        # save data
        print 'Saving ', args.refstat+'QC_WS_{}.npz'.format(extension_dir)
        np.savez(args.refstat+'QC_WS_{}.npz'.format(extension_dir),
                 white_before_hist=white_before_hist,
                 white_after_hist=white_after_hist,
                 gray_before_hist=gray_before_hist,
                 gray_after_hist=gray_after_hist,
                 list_subjects=list_subjects
                 )
        # save error_log
        fn = args.outfile + 'QC_WS_{}_error_subjects.txt'.format(extension_dir)
        print 'Saving ', fn
        with open(fn, 'w') as f:
            f.write('\n'.join(log_error_subjects))
            f.write('\n')

    # Plot individual subject stats
    else:
        # load the DB of distributions
        print 'Reading distribution DB'
        refstat = np.load(args.refstat + 'QC_WS_{}.npz'.format(extension_dir))
        white_before_hist = refstat['white_before_hist']
        white_after_hist = refstat['white_after_hist']
        gray_before_hist = refstat['gray_before_hist']
        gray_after_hist = refstat['gray_after_hist']
        list_subjects = refstat['list_subjects'].tolist()
        print '......................... done'

        # start plottings
        for i, s in enumerate(subjects):
            # for code testing
            if (sel_s is not None) and (sel_s not in s):
                continue

            # Progression
            print s, ' ', (i+1), '/', len(subjects)

            if s not in list_subjects:
                print '{}: files missing'.format(s)
                continue

            # images overlay
            t1 = os.path.join(
                            args.indir,
                            'whitestripe_t1',
                            '{}'.format(s),
                            'n4', 'bias_corrected.nii.gz')
            flair = os.path.join(
                            args.indir,
                            'whitestripe_flair',
                            '{}'.format(s),
                            'n4', 'bias_corrected.nii.gz')
            mask_gray = os.path.join(
                            args.indir, 'segmentation', '{}'.format(s),
                            'fast', 'brain_pve_1.nii.gz')
            mask_white = os.path.join(
                            args.indir, 'segmentation', '{}'.format(s),
                            'fast', 'brain_pve_2.nii.gz')
            nadwm = os.path.join(    # non apparently damaged white matter
                            args.indir,
                            'whitestripe_{}'.format(extension_dir),
                            '{}'.format(s),
                            'whitestripe', 'whitestripe.nii.gz')
            t1_img = ni.load(t1)
            flair_img = ni.load(flair)
            mask_gray_img = ni.load(mask_gray)
            mask_white_img = ni.load(mask_white)
            nadwm_img = ni.load(nadwm)

            # do the plots
            fig = plt.figure("T1 and T2 overlays", figsize=(20, 10))
            fig.subplots_adjust(hspace=0.01, bottom=0.01, left=0.01,
                                right=0.99, top=0.99)

            fig_id = plt.subplot(4, 1, 1)
            plotting.plot_roi(roi_img=mask_white_img, bg_img=t1_img,
                              cmap=plotting.cm.red_transparent,
                              title="Overlay WM on T1 anatomy",
                              display_mode='z', alpha=0.6,
                              cut_coords=20, axes=fig_id)
            fig_id = plt.subplot(4, 1, 2)
            plotting.plot_roi(roi_img=mask_white_img, bg_img=flair_img,
                              cmap=plotting.cm.red_transparent,
                              title="Overlay WM on FLAIR",
                              display_mode='z', alpha=0.6,
                              cut_coords=20, axes=fig_id)
            fig_id = plt.subplot(4, 1, 3)
            plotting.plot_roi(roi_img=nadwm_img, bg_img=t1_img,
                              cmap=plotting.cm.red_transparent,
                              title="Overlay striped voxels on T1 anatomy",
                              display_mode='z', alpha=.99,
                              cut_coords=20, axes=fig_id)
            fig_id = plt.subplot(4, 1, 4)
            d = plotting.plot_roi(roi_img=nadwm_img, bg_img=flair_img,
                                  cmap=plotting.cm.red_transparent,
                                  title="Overlay striped voxels on FLAIR",
                                  display_mode='z', alpha=.99,
                                  cut_coords=20, axes=fig_id)
            file_name = args.outfile +\
                'ws_{}_overlays_{}'.format(extension_dir, s) + '.png'
            print 'Saving ', file_name
            d.savefig(file_name, dpi=300)
            d.close()

            # get index of the current subjects in the list
            ind = list_subjects.index(s)
            # kde plots
            sns.plt.figure()
            sns.set_style('whitegrid')
            sns.set_color_codes("pastel")
            ax = sns.plt.subplot(2, 2, 1)
            title = 'WM orig {}'.format(s)
            ax.set_title(title)
            ax.set_autoscalex_on(False)
            ax.set_xlim([0, 4000])
            for h in white_before_hist:
                sns.kdeplot(np.array(h[h > 0.]), bw=0.5, color='#BBBBBB')
            h = white_before_hist[ind]
            sns.kdeplot(np.array(h[h > 0.]), bw=0.5, color='r')
            ax = sns.plt.subplot(2, 2, 3)
            title = 'WM WScorr  {}'.format(s)
            ax.set_title(title)
            ax.set_autoscalex_on(False)
            ax.set_xlim([-20, 20])
            for h in white_after_hist:
                sns.kdeplot(np.array(h), bw=0.5, color='#BBBBBB')
            h = white_after_hist[ind]
            sns.kdeplot(np.array(h), bw=0.5, color='r')

            ax = sns.plt.subplot(2, 2, 2)
            title = 'GM orig {}'.format(s)
            ax.set_title(title)
            ax.set_autoscalex_on(False)
            ax.set_xlim([0, 4000])
            for h in gray_before_hist:
                sns.kdeplot(np.array(h[h > 0.]), bw=0.5, color='#BBBBBB')
            h = gray_before_hist[ind]
            sns.kdeplot(np.array(h[h > 0.]), bw=0.5, color='r')
            ax = sns.plt.subplot(2, 2, 4)
            title = 'GM WScorr {}'.format(s)
            ax.set_title(title)
            ax.set_autoscalex_on(False)
            ax.set_xlim([-40, 20])
            for h in gray_after_hist:
                sns.kdeplot(np.array(h), bw=0.5, color='#BBBBBB')
            h = gray_after_hist[ind]
            sns.kdeplot(np.array(h), bw=0.5, color='r')
            file_name = args.outfile + \
                'ws_{}_distributions_{}.png'.format(extension_dir, s)
            print 'Saving ', file_name
            sns.plt.savefig(file_name, dpi=300)
            sns.plt.close('all')


if __name__ == "__main__":
    spyder_inter = False
    ws_hist = main(spyder_inter)
