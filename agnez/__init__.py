# -*- coding: utf-8 -*-

__author__ = 'Eder Santana'
__email__ = 'edercsjr+git@gmail.com'
__version__ = '0.1.0'

import seaborn as sns
sns.set_style('dark')
sns.set_palette('muted')
try:
    import IPython
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
except:
    pass

from .grid import *
from .embedding import *
from .video import *
from .utils import *
