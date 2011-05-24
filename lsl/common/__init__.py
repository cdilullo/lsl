# -*- coding: utf-8 -*

"""
lsl.common - Common information for the LSL package.  Including:
 * warns    - depercation and experimental feature warnings, 
 * paths    - paths to the module and its data directories, 
 * dp       - information about the DP system, 
 * stations - information about LWA stations, and 
 * sdm      - read in and interpret SDM files.
 """

import warns
import progress

import paths
import dp
import stations
import sdm
