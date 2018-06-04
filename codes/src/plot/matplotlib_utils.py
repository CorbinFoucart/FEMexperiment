#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools for matplotlib interaction, colorbars, common figure layouts etc
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

def add_shared_colorbar(fig, cbar_scaling_data, cmap,
        cbar_ax=[0.92, 0.15, 0.02, 0.7], right_adjust=0.9, **kwargs):
    """ add a shared colorbar to fig
    @param fig  the object on which to draw the colorbar
    @param cbar_scaling_data  the data with which to scale the colorbar
    @param cmap  colormap
    @param cbar_ax  Add an axes at position rect [left, bottom, width, height]
        where all quantities are in fractions of figure width and height.
    @param right adjust  percent of fig width for plots, rest for cbar
    @param kwargs  kwargs to fig.colorbar call
    """
    fig.subplots_adjust(right=right_adjust)
    cbar_axis = fig.add_axes(cbar_ax)
    m = colorbar_data_normed(cbar_scaling_data, cmap)
    fig.colorbar(m, cax=cbar_axis, **kwargs)
    return fig
