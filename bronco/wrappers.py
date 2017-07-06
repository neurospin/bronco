##########################################################################
# NSAp - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import os
import re
import pwd
import json
import warnings
import subprocess


# Module import
from .info import ANTS_RELEASE
from .exceptions import ANTSRuntimeError
from .exceptions import ANTSDependencyError
from .exceptions import ANTSConfigurationError


class ANTSWrapper(object):
    """ Parent class for the wrapping of ANTS functions.
    """
    def __init__(self, env=None, verbose=0):
        """ Initialize the ANTSWrapper class by setting properly the
        environment.

        Parameters
        ----------
        env: dict, default None
            the current environment in which the ANTS command will be executed.
            Default None, the current environment.
        verbose: int, default 0
            control the verbosity level.
        """
        self.cmd = None
        self.version = None
        self.verbose = verbose
        self.environment = env or os.environ
        self._ants_version_check()

    def __call__(self, cmd, cwdir=None):
        """ Run the ANTS command.

        Parameters
        ----------
        cmd: list of str
            the ANTS command to execute.
        cwdir: str, default None
            the working directory that will be passed to the subprocess.
        """
        # Welcome message
        if self.verbose > 0:
            print("[info] Executing '{0}'...".format(" ".join(cmd)))

        # Check ANTS has been configured so the command can be found
        self.cmd = cmd
        process = subprocess.Popen(["which", self.cmd[0]],
                                   env=self.environment,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        self.stdout, self.stderr = process.communicate()
        self.exitcode = process.returncode
        if self.exitcode != 0:
            raise ANTSConfigurationError(self.cmd[0])

        # Execute the command
        process = subprocess.Popen(self.cmd,
                                   env=self.environment,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   cwd=cwdir)
        self.stdout, self.stderr = process.communicate()
        self.exitcode = process.returncode

        # Raise exception of exitcode is not zero
        if self.exitcode != 0:
            raise ANTSRuntimeError(
                self.cmd[0], " ".join(self.cmd[1:]), self.stderr + self.stdout)

    def _ants_version_check(self):
        """ Check that a tested ANTS version is installed.
        """
        # Check ANTS version
        cmd = ["dpkg", "-s", "ants"]
        process = subprocess.Popen(cmd,
                                   env=self.environment,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        self.stdout, self.stderr = process.communicate()
        self.exitcode = process.returncode
        if self.exitcode != 0:
            raise ANTSRuntimeError(
                cmd[0], " ".join(cmd[1:]), self.stderr + self.stdout)
        versions = re.findall("Version: .*$", self.stdout, re.MULTILINE)
        self.version = versions[0].replace("Version: ", "")
        if not self.version.startswith(ANTS_RELEASE):
            message = ("Installed '{0}' version of ANTS "
                       "not tested. Currently supported version "
                       "is '{1}'.".format(self.version, ANTS_RELEASE))
            warnings.warn(message)

