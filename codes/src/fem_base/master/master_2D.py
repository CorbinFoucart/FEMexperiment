#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# define the master elements in 2D
MASTER_ELEMENT_VERTICES = {
    'TRIANGLE': ((-1, -1), (1, -1), (-1, 1)),
    'QUAD': ((-1, -1), (1, -1), (1, 1), (-1, 1))
}

