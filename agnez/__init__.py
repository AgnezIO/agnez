# -*- coding: utf-8 -*-

__author__ = 'Eder Santana'
__email__ = 'edercsjr+git@gmail.com'
__version__ = '0.1.0'

import seaborn as sns
sns.set_style('dark')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

from weight import *
from inputs import *
from output import *
