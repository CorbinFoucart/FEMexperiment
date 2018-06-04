#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools for plotting over triangulations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def colorbar_data_normed(data, cmap):
    """ creates a ScalarMappable object normalized to data max and min
    usage:
        m = colorbar_data_normed(vh, cmo.balance)
        plt.colorbar(m)
    """
    norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
    m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array([])
    return m
