# -*- coding: utf-8 -*-
"""@package zipdir
Packages and compresses (archives) files

@author: Chris Mirabito (mirabito@mit.edu)
"""
import os
import zipfile


def zipdir(path, zip_path):
    """Packages and compresses (archives) the contents of path into zip_path
    @param path Path string
    @param zip_path Zip path string
    """
    parent_dir = os.path.dirname(path)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zfile:
        for root, dirs, files in os.walk(path):
            for dirname in dirs:
                abs_path = os.path.join(root, dirname)
                rel_path = abs_path.replace(parent_dir + os.sep, '')
                zfile.write(abs_path, rel_path)
            for filename in files:
                abs_path = os.path.join(root, filename)
                rel_path = abs_path.replace(parent_dir + os.sep, '')
                zfile.write(abs_path, rel_path)
