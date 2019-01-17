# -*- coding: utf-8 -*-

# Python3 compatiability
import sys
if sys.version_info > (3,):
    xrange = range
    
"""
Unit test for the lsl.common.idf module.
"""

import os
import pytz
import ephem
import tempfile
import unittest
from datetime import datetime, timedelta
try:
    import cStringIO as StringIO
except ImprotError:
    import StringIO

from lsl.common.paths import DATA_BUILD
from lsl.common import idf
from lsl.common.stations import lwa1, lwasv


__revision__ = "$Rev$"
__version__  = "0.1"
__author__    = "Jayce Dowell"


drxFile = os.path.join(DATA_BUILD, 'tests', 'drx-idf.txt')
solFile = os.path.join(DATA_BUILD, 'tests', 'sol-idf.txt')
jovFile = os.path.join(DATA_BUILD, 'tests', 'jov-idf.txt')
sdfFile = os.path.join(DATA_BUILD, 'tests', 'drx-sdf.txt')


class idf_tests(unittest.TestCase):
    """A unittest.TestCase collection of unit tests for the lsl.common.idf
    module."""
    
    testPath = None

    def setUp(self):
        """Create the temporary file directory."""

        self.testPath = tempfile.mkdtemp(prefix='test-idf-', suffix='.tmp')
        
    ### General ###
    
    def test_flat_projects(self):
        """Test single session/scans SDFs."""
        
        obs = idf.Observer('Test Observer', 99)
        targ = idf.DRX('Target', 'Target', '2019/1/1 00:00:00', '00:00:10', 0.0, 90.0, 40e6, 50e6, 6)
        sess = idf.Run('Test Session', 1, scans=targ)
        proj = idf.Project(obs, 'Test Project', 'COMTST', runs=sess)
        out = proj.render()
        
    def test_ucf_username(self):
        """Test setting the UCF username for auto-copy support."""
        
        obs = idf.Observer('Test Observer', 99)
        targ = idf.DRX('Target', 'Target', '2019/1/1 00:00:00', '00:00:10', 0.0, 90.0, 40e6, 50e6, 6)
        sess = idf.Run('Test Session', 1, scans=targ)
        sess.set_data_return_method('UCF')
        sess.set_ucf_username('test')
        proj = idf.Project(obs, 'Test Project', 'COMTST', runs=sess)
        out = proj.render()
        self.assertTrue(out.find('ucfuser:test') >= 0)
        
        obs = idf.Observer('Test Observer', 99)
        targ = idf.DRX('Target', 'Target', '2019/1/1 00:00:00', '00:00:10', 0.0, 90.0, 40e6, 50e6, 6)
        sess = idf.Run('Test Session', 1, scans=targ, comments='This is a comment')
        sess.set_data_return_method('UCF')
        sess.set_ucf_username('test/dir1')
        proj = idf.Project(obs, 'Test Project', 'COMTST', runs=sess)
        out = proj.render()
        self.assertTrue(out.find('ucfuser:test/dir1') >= 0)
        
    ### DRX - TRK_RADEC ###
    
    def test_drx_parse(self):
        """Test reading in a TRK_RADEC SDF file."""
        
        project = idf.parse_idf(drxFile)
        
        # Basic file structure
        self.assertEqual(len(project.runs), 1)
        self.assertEqual(len(project.runs[0].scans), 1)
        
        # Correlator setup
        self.assertEqual(project.runs[0].corr_channels, 256)
        self.assertAlmostEqual(project.runs[0].corr_inttime, 0.1, 6)
        self.assertEqual(project.runs[0].corr_basis, 'linear')
        
        # Observational setup - 1
        self.assertEqual(project.runs[0].scans[0].mode, 'TRK_RADEC')
        self.assertEqual(project.runs[0].scans[0].mjd,  58490)
        self.assertEqual(project.runs[0].scans[0].mpm,  74580000)
        self.assertEqual(project.runs[0].scans[0].dur,  7200000)
        self.assertEqual(project.runs[0].scans[0].freq1, 766958446)
        self.assertEqual(project.runs[0].scans[0].freq2, 1643482384)
        self.assertEqual(project.runs[0].scans[0].filter,   6)
        self.assertAlmostEqual(project.runs[0].scans[0].ra, 19.991210200, 6)
        self.assertAlmostEqual(project.runs[0].scans[0].dec, 40.733916000, 6)
        
    def test_drx_update(self):
        """Test updating TRK_RADEC values."""
        
        project = idf.parse_idf(drxFile)
        project.runs[0].scans[0].set_start("MST 2011 Feb 23 17:00:15")
        project.runs[0].scans[0].set_duration(timedelta(seconds=15))
        project.runs[0].scans[0].set_frequency1(75e6)
        project.runs[0].scans[0].set_frequency2(76e6)
        project.runs[0].scans[0].set_ra(ephem.hours('5:30:00'))
        project.runs[0].scans[0].set_dec(ephem.degrees('+22:30:00'))
        
        self.assertEqual(project.runs[0].scans[0].mjd,  55616)
        self.assertEqual(project.runs[0].scans[0].mpm,  15000)
        self.assertEqual(project.runs[0].scans[0].dur,  15000)
        self.assertEqual(project.runs[0].scans[0].freq1, 1643482384)
        self.assertEqual(project.runs[0].scans[0].freq2, 1665395482)
        self.assertAlmostEqual(project.runs[0].scans[0].ra, 5.5, 6)
        self.assertAlmostEqual(project.runs[0].scans[0].dec, 22.5, 6)
        
    def test_drx_write(self):
        """Test writing a TRK_RADEC SDF file."""
        
        project = idf.parse_idf(drxFile)
        # Fix for LWA-SV only going up to filter code 6
        for obs in project.runs[0].scans:
            obs.filter = 6
        out = project.render()
        
    def test_drx_errors(self):
        """Test various TRK_RADEC SDF errors."""
        
        project = idf.parse_idf(drxFile)
        
        # Bad interferometer
        project.runs[0].stations = [lwa1,]
        self.assertFalse(project.validate())
        
        # Bad correlator channel count
        project.runs[0].corr_channels = 129
        self.assertFalse(project.validate())
        
        # Bad correlator integration time
        project.runs[0].corr_inttime = 1e-6
        self.assertFalse(project.validate())
        
        # Bad correlator output polarization basis
        project.runs[0].corr_basis = 'cats'
        self.assertFalse(project.validate())
        
        # Bad filter
        project.runs[0].scans[0].filter = 7
        self.assertFalse(project.validate())
        
        # Bad frequency
        project.runs[0].scans[0].filter = 6
        project.runs[0].scans[0].frequency1 = 90.0e6
        project.runs[0].scans[0].update()
        self.assertFalse(project.validate())
        
        project.runs[0].scans[0].frequency1 = 38.0e6
        project.runs[0].scans[0].frequency2 = 90.0e6
        project.runs[0].scans[0].update()
        self.assertFalse(project.validate())
        
        # Bad duration
        project.runs[0].scans[0].frequency2 = 38.0e6
        project.runs[0].scans[0].duration = '96:00:00.000'
        project.runs[0].scans[0].update()
        self.assertFalse(project.validate())
        
        # Bad pointing
        project.runs[0].scans[0].duration = '00:00:01.000'
        project.runs[0].scans[0].dec = -72.0
        project.runs[0].scans[0].update()
        self.assertFalse(project.validate())
        
    ### DRX - TRK_SOL ###
    
    def test_sol_parse(self):
        """Test reading in a TRK_SOL SDF file."""
        
        project = idf.parse_idf(solFile)
        
        # Basic file structure
        self.assertEqual(len(project.runs), 1)
        self.assertEqual(len(project.runs[0].scans), 1)
        
        # Correlator setup
        self.assertEqual(project.runs[0].corr_channels, 512)
        self.assertAlmostEqual(project.runs[0].corr_inttime, 0.5, 6)
        self.assertEqual(project.runs[0].corr_basis, 'stokes')
        
        # Observational setup - 1
        self.assertEqual(project.runs[0].scans[0].mode, 'TRK_SOL')
        self.assertEqual(project.runs[0].scans[0].mjd,  58490)
        self.assertEqual(project.runs[0].scans[0].mpm,  74580000)
        self.assertEqual(project.runs[0].scans[0].dur,  7200000)
        self.assertEqual(project.runs[0].scans[0].freq1, 766958446)
        self.assertEqual(project.runs[0].scans[0].freq2, 1643482384)
        self.assertEqual(project.runs[0].scans[0].filter,   6)
        
    def test_sol_update(self):
        """Test updating TRK_SOL values."""
        
        project = idf.parse_idf(solFile)
        project.runs[0].scans[0].set_start("MST 2011 Feb 23 17:00:15")
        project.runs[0].scans[0].set_duration(timedelta(seconds=15))
        project.runs[0].scans[0].set_frequency1(75e6)
        project.runs[0].scans[0].set_frequency2(76e6)
        
        self.assertEqual(project.runs[0].scans[0].mjd,  55616)
        self.assertEqual(project.runs[0].scans[0].mpm,  15000)
        self.assertEqual(project.runs[0].scans[0].dur,  15000)
        self.assertEqual(project.runs[0].scans[0].freq1, 1643482384)
        self.assertEqual(project.runs[0].scans[0].freq2, 1665395482)
        
    def test_sol_write(self):
        """Test writing a TRK_SOL SDF file."""
        
        project = idf.parse_idf(solFile)
        # Fix for LWA-SV only going up to filter code 6
        for obs in project.runs[0].scans:
            obs.filter = 6
        out = project.render()
        
    def test_sol_errors(self):
        """Test various TRK_SOL SDF errors."""
        
        project = idf.parse_idf(solFile)
        
        # Bad interferometer
        project.runs[0].stations = [lwa1,]
        self.assertFalse(project.validate())
        
        # Bad correlator channel count
        project.runs[0].corr_channels = 129
        self.assertFalse(project.validate())
        
        # Bad correlator integration time
        project.runs[0].corr_inttime = 1e-6
        self.assertFalse(project.validate())
        
        # Bad correlator output polarization basis
        project.runs[0].corr_basis = 'cats'
        self.assertFalse(project.validate())
        
        # Bad filter
        project.runs[0].scans[0].filter = 7
        self.assertFalse(project.validate())
        
        # Bad frequency
        project.runs[0].scans[0].filter = 6
        project.runs[0].scans[0].frequency1 = 90.0e6
        project.runs[0].scans[0].update()
        self.assertFalse(project.validate())
        
        project.runs[0].scans[0].frequency1 = 38.0e6
        project.runs[0].scans[0].frequency2 = 90.0e6
        project.runs[0].scans[0].update()
        self.assertFalse(project.validate())
        
        # Bad duration
        project.runs[0].scans[0].frequency2 = 38.0e6
        project.runs[0].scans[0].duration = '96:00:00.000'
        project.runs[0].scans[0].update()
        self.assertFalse(project.validate())
        
    ### DRX - TRK_JOV ###
    
    def test_jov_parse(self):
        """Test reading in a TRK_JOV SDF file."""
        
        project = idf.parse_idf(jovFile)
        
        # Basic file structure
        self.assertEqual(len(project.runs), 1)
        self.assertEqual(len(project.runs[0].scans), 1)
        
        # Correlator setup
        self.assertEqual(project.runs[0].corr_channels, 512)
        self.assertAlmostEqual(project.runs[0].corr_inttime, 0.1, 6)
        self.assertEqual(project.runs[0].corr_basis, 'circular')
        
        # Observational setup - 1
        self.assertEqual(project.runs[0].scans[0].mode, 'TRK_JOV')
        self.assertEqual(project.runs[0].scans[0].mjd,  58490)
        self.assertEqual(project.runs[0].scans[0].mpm,  74580000)
        self.assertEqual(project.runs[0].scans[0].dur,  7200000)
        self.assertEqual(project.runs[0].scans[0].freq1, 766958446)
        self.assertEqual(project.runs[0].scans[0].freq2, 1643482384)
        self.assertEqual(project.runs[0].scans[0].filter,   6)
        
    def test_jov_update(self):
        """Test updating TRK_JOV values."""
        
        project = idf.parse_idf(jovFile)
        project.runs[0].scans[0].set_start("MST 2011 Feb 23 17:00:15")
        project.runs[0].scans[0].set_duration(timedelta(seconds=15))
        project.runs[0].scans[0].set_frequency1(75e6)
        project.runs[0].scans[0].set_frequency2(76e6)
        
        self.assertEqual(project.runs[0].scans[0].mjd,  55616)
        self.assertEqual(project.runs[0].scans[0].mpm,  15000)
        self.assertEqual(project.runs[0].scans[0].dur,  15000)
        self.assertEqual(project.runs[0].scans[0].freq1, 1643482384)
        self.assertEqual(project.runs[0].scans[0].freq2, 1665395482)
        
    def test_jov_write(self):
        """Test writing a TRK_JOV SDF file."""
        
        project = idf.parse_idf(jovFile)
        # Fix for LWA-SV only going up to filter code 6
        for obs in project.runs[0].scans:
            obs.filter = 6
        out = project.render()
        
    def test_jov_errors(self):
        """Test various TRK_JOV SDF errors."""
        
        project = idf.parse_idf(jovFile)
        
        # Bad interferometer
        project.runs[0].stations = [lwa1,]
        self.assertFalse(project.validate())
        
        # Bad correlator channel count
        project.runs[0].corr_channels = 129
        self.assertFalse(project.validate())
        
        # Bad correlator integration time
        project.runs[0].corr_inttime = 1e-6
        self.assertFalse(project.validate())
        
        # Bad correlator output polarization basis
        project.runs[0].corr_basis = 'cats'
        self.assertFalse(project.validate())
        
        # Bad filter
        project.runs[0].scans[0].filter = 7
        self.assertFalse(project.validate())
        
        # Bad frequency
        project.runs[0].scans[0].filter = 6
        project.runs[0].scans[0].frequency1 = 90.0e6
        project.runs[0].scans[0].update()
        self.assertFalse(project.validate())
        
        project.runs[0].scans[0].frequency1 = 38.0e6
        project.runs[0].scans[0].frequency2 = 90.0e6
        project.runs[0].scans[0].update()
        self.assertFalse(project.validate())
        
        # Bad duration
        project.runs[0].scans[0].frequency2 = 38.0e6
        project.runs[0].scans[0].duration = '96:00:00.000'
        project.runs[0].scans[0].update()
        self.assertFalse(project.validate())
        
    ### Misc. ###
    
    def test_generate_sdfs(self):
        """Test generated SDFs from the IDF."""
        
        project = idf.parse_idf(drxFile)
        sdfs = project.generate_sdfs()
        
    def test_auto_update(self):
        """Test project auto-update on render."""
        
        # Part 1 - frequency and duration
        project = idf.parse_idf(drxFile)
        project.runs[0].scans[0].frequency1 = 75e6
        project.runs[0].scans[0].duration = '00:01:31.000'
        
        # Fix for LWA-SV only going up to filter code 6
        for obs in project.runs[0].scans:
            obs.filter = 6
        fh = open(os.path.join(self.testPath, 'idf.txt'), 'w')
        fh.write(project.render())
        fh.close()
        
        project = idf.parse_idf(os.path.join(self.testPath, 'idf.txt'))
        self.assertEqual(project.runs[0].scans[0].freq1, 1643482384)
        self.assertEqual(project.runs[0].scans[0].dur, 91000)
        
        # Part 2 - frequency and duration (timedelta)
        project = idf.parse_idf(drxFile)
        project.runs[0].scans[0].frequency1 = 75e6
        project.runs[0].scans[0].duration = timedelta(minutes=1, seconds=31, microseconds=1000)
        
        # Fix for LWA-SV only going up to filter code 6
        for obs in project.runs[0].scans:
            obs.filter = 6
        fh = open(os.path.join(self.testPath, 'idf.txt'), 'w')
        fh.write(project.render())
        fh.close()
        
        project = idf.parse_idf(os.path.join(self.testPath, 'idf.txt'))
        self.assertEqual(project.runs[0].scans[0].freq1, 1643482384)
        self.assertEqual(project.runs[0].scans[0].dur, 91001)
        
        # Part 3 - frequency and start time
        project = idf.parse_idf(drxFile)
        project.runs[0].scans[0].frequency2 = 75e6
        project.runs[0].scans[0].start = "MST 2011 Feb 23 17:00:15"
        
        # Fix for LWA-SV only going up to filter code 6
        for obs in project.runs[0].scans:
            obs.filter = 6
        fh = open(os.path.join(self.testPath, 'idf.txt'), 'w')		
        fh.write(project.render())
        fh.close()
        
        project = idf.parse_idf(os.path.join(self.testPath, 'idf.txt'))
        self.assertEqual(project.runs[0].scans[0].freq2, 1643482384)
        self.assertEqual(project.runs[0].scans[0].mjd,  55616)
        self.assertEqual(project.runs[0].scans[0].mpm,  15000)
        
        # Part 4 - frequency and start time (timedelta)
        project = idf.parse_idf(drxFile)
        _MST = pytz.timezone('US/Mountain')
        project.runs[0].scans[0].frequency2 = 75e6
        project.runs[0].scans[0].start = _MST.localize(datetime(2011, 2, 23, 17, 00, 30, 1000))
        
        # Fix for LWA-SV only going up to filter code 6
        for obs in project.runs[0].scans:
            obs.filter = 6
        fh = open(os.path.join(self.testPath, 'idf.txt'), 'w')		
        fh.write(project.render())
        fh.close()
        
        project = idf.parse_idf(os.path.join(self.testPath, 'idf.txt'))
        self.assertEqual(project.runs[0].scans[0].freq2, 1643482384)
        self.assertEqual(project.runs[0].scans[0].mjd,  55616)
        self.assertEqual(project.runs[0].scans[0].mpm,  30001)
        
    def test_set_stations(self):
        """Test the set stations functionlity."""
        
        project = idf.parse_idf(drxFile)
        project.runs[0].set_stations([lwasv, lwa1])
        # Fix for LWA-SV only going up to filter code 6
        for obs in project.runs[0].scans:
            obs.filter = 6
        self.assertTrue(project.validate())
        
    def test_is_valid(self):
        """Test whether or not is_valid works."""
        
        self.assertTrue(idf.is_valid(drxFile))
        self.assertTrue(idf.is_valid(solFile))
        self.assertTrue(idf.is_valid(jovFile))
        
    def test_is_not_valid(self):
        """Test whether or not is_valid works on LWA1 and IDF files."""
        
        self.assertFalse(idf.is_valid(sdfFile))
        
    def test_username(self):
        """Test setting auto-copy parameters."""
        
        project = idf.parse_idf(drxFile)
        project.runs[0].set_data_return_method('UCF')
        project.runs[0].set_ucf_username('jdowell')
        # Fix for LWA-SV only going up to filter code 6
        for obs in project.runs[0].scans:
            obs.filter = 6
        out = project.render()
        
        self.assertTrue(out.find('Requested data return method is UCF') > 0)
        self.assertTrue(out.find('ucfuser:jdowell') > 0)
        
        fh = open(os.path.join(self.testPath, 'idf.txt'), 'w')		
        fh.write(out)
        fh.close()
        
        project = idf.parse_idf(os.path.join(self.testPath, 'idf.txt'))
        out = project.render()
        
        self.assertTrue(out.find('Requested data return method is UCF') > 0)
        self.assertTrue(out.find('ucfuser:jdowell') > 0)
        
    def tearDown(self):
        """Remove the test path directory and its contents"""

        tempFiles = os.listdir(self.testPath)
        for tempFile in tempFiles:
            os.unlink(os.path.join(self.testPath, tempFile))
        os.rmdir(self.testPath)
        self.testPath = None


class idf_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the lsl.common.idf units 
    tests."""
    
    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(idf_tests)) 


if __name__ == '__main__':
    unittest.main()
