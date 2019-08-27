# -*- coding: utf-8 -*

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
"""
LWA Software Library

Provided packages:
  * lsl.common
  * lsl.reader
  * lsl.writer
  * lsl.correlator
  * lsl.statistics
  * lsl.sim
  * lsl.imaging
  * lsl.misc

Provided modules:
  * lsl.astro
  * lsl.catalog
  * lsl.skymap
  * lsl.transform

See the individual package descriptions for more information.
"""

from lsl import version

__version__ = '0.8'
__revision__ = '$Rev$'
__author__ = "Jayce Dowell"
