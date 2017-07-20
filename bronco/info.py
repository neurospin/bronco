##########################################################################
# NSAp - Copyright (C) CEA, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Module current version
version_major = 0
version_minor = 0
version_micro = 0

# Expected by setup.py: string of form "X.Y.Z"
__version__ = "{0}.{1}.{2}".format(version_major, version_minor, version_micro)

# The tested ants version
ANTS_RELEASE = "2.2.0"

# Expected by setup.py: the status of the project
CLASSIFIERS = ["Development Status :: 5 - Production/Stable",
               "Environment :: Console",
               "Environment :: X11 Applications :: Qt",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering",
               "Topic :: Utilities"]

# Project descriptions
description = """[bronco]
This package provides common scripts:

* bronco_xxx
"""
long_description = """
======================
bronco
======================

Python codes for radiomics.
"""

# Main setup parameters
NAME = "bronco"
ORGANISATION = "CEA"
MAINTAINER = "xxx"
MAINTAINER_EMAIL = "xxx"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/neurospin/bronco"
DOWNLOAD_URL = "https://github.com/neurospin/bronco"
LICENSE = "CeCILL-B"
CLASSIFIERS = CLASSIFIERS
AUTHOR = "bronco developers"
AUTHOR_EMAIL = "xxx"
PLATFORMS = "OS Independent"
ISRELEASE = True
VERSION = __version__
PROVIDES = ["bronco"]
REQUIRES = [
    "numpy>=1.6.1",
    "scipy>=0.9.0",
    "nibabel>=1.1.0",
    "pyradiomics>=1.2.0"
]
EXTRA_REQUIRES = {}
