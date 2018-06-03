# -*- coding: utf-8 -*-

"""@file svnversion.py
Produces a compact version identifier for the working copy

@author Chris Mirabito (mirabito@mit.edu)
"""
import subprocess


def svnversion():
    """Produces a compact version identifier for the working copy
    @return SVN version identifier (from @c stdout of the subprocess)
    """
    process = subprocess.Popen('svnversion',
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    (stdout, _) = process.communicate()
    return stdout
