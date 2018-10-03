# -*- coding: utf-8 -*-

"""Unit test for lsl.reader.ldp module"""

import os
import unittest

from lsl.common.paths import DATA_BUILD
from lsl.reader import ldp
from lsl.reader import errors


__revision__ = "$Rev$"
__version__  = "0.1"
__author__    = "Jayce Dowell"


tbwFile = os.path.join(DATA_BUILD, 'tests', 'tbw-test.dat')
tbnFile = os.path.join(DATA_BUILD, 'tests', 'tbn-test.dat')
drxFile = os.path.join(DATA_BUILD, 'tests', 'drx-test.dat')
drspecFile = os.path.join(DATA_BUILD, 'tests', 'drspec-test.dat')

tbfFile = os.path.join(DATA_BUILD, 'tests', 'tbf-test.dat')


class ldp_adp_tests(unittest.TestCase):
    """A unittest.TestCase collection of unit tests for the lsl.reader
    modules."""
    
    ### TBF ###
    
    def test_ldp_tbf(self):
        """Test the LDP interface for a TBF file."""
        
        f = ldp.TBFFile(tbfFile)
        
        # File info
        self.assertEqual(f.get_info("sample_rate"), 25e3)
        self.assertEqual(f.get_info("data_bits"), 4)
        self.assertEqual(f.get_info("nframes"), 5)
        
        self.assertEqual(f.sample_rate, 25e3)
        self.assertEqual(f.data_bits, 4)
        self.assertEqual(f.nframes, 5)
        
        # Read a frame
        frame = f.read_frame()
        
        # Get the remaining frame count
        self.assertEqual(f.get_remaining_frame_count(), f.get_info("nframes")-1)
        self.assertEqual(f.nframes_remaining, f.get_info("nframes")-1)
        
        # Reset
        f.reset()
        
        # Close it out
        f.close()
        
    def test_ldp_tbf_nocheck(self):
        """Test the LDP interface for a TBF file."""
        
        f = ldp.TBFFile(tbfFile, ignore_timetag_errors=True)
        
        # File info
        self.assertEqual(f.get_info("sample_rate"), 25e3)
        self.assertEqual(f.get_info("data_bits"), 4)
        self.assertEqual(f.get_info("nframes"), 5)
        
        self.assertEqual(f.sample_rate, 25e3)
        self.assertEqual(f.data_bits, 4)
        self.assertEqual(f.nframes, 5)
        
        # Read a frame
        frame = f.read_frame()
        
        # Get the remaining frame count
        self.assertEqual(f.get_remaining_frame_count(), f.get_info("nframes")-1)
        self.assertEqual(f.nframes_remaining, f.get_info("nframes")-1)
        
        # Reset
        f.reset()
        
        # Close it out
        f.close()
        
    ### File Type Discovery ###
    
    def test_ldp_discover_tbw(self):
        """Test the LDP LWA1DataFile function of TBW."""
        # TBW
        self.assertRaises(RuntimeError, ldp.LWASVDataFile, tbwFile)
        
    def test_ldp_discover_tbn(self):
        """Test the LDP LWASVDataFile function of TBN."""
        # TBN
        f = ldp.LWASVDataFile(tbnFile)
        self.assertEqual(type(f), ldp.TBNFile)
        
    def test_ldp_discover_drx(self):
        """Test the LDP LWASVDataFile function of DRX."""
        # DRX
        f = ldp.LWASVDataFile(drxFile)
        self.assertEqual(type(f), ldp.DRXFile)
        
    def test_ldp_discover_drspec(self):
        """Test the LDP LWASVDataFile function of DR Spectrometer."""
        # DR Spectrometer
        f = ldp.LWASVDataFile(drspecFile)
        self.assertEqual(type(f), ldp.DRSpecFile)
        
    def test_ldp_discover_tbf(self):
        """Test the LDP LWASVDataFile function of TBF."""
        # TBF
        f = ldp.LWASVDataFile(tbfFile)
        self.assertEqual(type(f), ldp.TBFFile)
        
    def tearDown(self):
        """Cleanup"""
        for handler in list(ldp._open_ldp_files.handlers):
            handler.close()


class ldp_adp_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the lsl.reader.ldp 
    unit tests."""
    
    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(ldp_adp_tests)) 


if __name__ == '__main__':
    unittest.main()
