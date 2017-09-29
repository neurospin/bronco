#! /usr/bin/env python

# System import
from nilearn import plotting
import nibabel
import argparse
import os
from clindmri.registration.fsl import flirt
from glob import glob

doc = """

Create QC plots (from nilearn) about FLAIR to T1-enh FLIRT

Command:
========

python qc_coreg.py -f AxT2.nii.gz -t AxT1enhanced.nii.gz  -o  qc_coreg

"""


# Parsing
def is_dir(dirarg):
    """ Type for argparse - checks that output dir exists.
    """
    if not os.path.isdir(dirarg):
        raise argparse.ArgumentError(
            "The dir '{0}' does not exist!".format(dirarg))
    return dirarg

parser = argparse.ArgumentParser(description=doc)
parser.add_argument(
    "-v", "--verbose", dest="verbose", type=int, choices=[0, 1, 2], default=0,
    help="increase the verbosity level: 0 silent, [1, 2] verbose.")
parser.add_argument(
    "-f", "--flair", dest="flair", metavar="FILE",
    help="Image to register")
parser.add_argument(
    "-t", "--t1gado", dest="t1gado", metavar="FILE",
    help="Target/Reference image file")
parser.add_argument(
    "-o", "--out", dest="outfile", metavar="FILE",
    help="Image file that will contain the qc")
args = parser.parse_args()


# transform pathnames
outfile = args.outfile
flair = args.flair
t1gado = args.t1gado
omatfile = os.path.splitext(outfile)[0]
outfileAxi = '{}_qc_axi.png'.format(omatfile)
outfileSag = '{}_qc_sag.png'.format(omatfile)



# QC :  pdf sheet
bg = nibabel.load(flair)
anat = nibabel.load(t1gado)
# image axial
display = plotting.plot_anat(bg, title="T1 Gado contours", 
                             display_mode = 'z',
                             cut_coords = 10)
display.add_edges(anat)
display.savefig(outfileAxi)
display.close()
# mage coronal
display = plotting.plot_anat(bg, title="T1 Gado contours", 
                             display_mode = 'x',
                             cut_coords = 10)
display.add_edges(anat)
display.savefig(outfileSag)
display.close()