"""
Module that contains all of the relevant class to build up a representation 
of a interferometer definition file.  The hierarchy of classes is:
 * Project - class that holds all of the information about the project (including
             the observer) and one or more runs.  Technically, a ID file has 
             only one run but this approach allows for the generation of 
             multiple SD files from a single Project object.
 * Observer - class that hold the observer's name and numeric ID
 * Run - class that holds all of the scans associated and the associated
         correlator setup for a with a particular interferometer run
 * Scan - class that hold information about a particular scan.  It
          includes a variety of attributes that are used to convert human-
          readable inputs to SDF data values.  The scan class is 
          further subclasses into:
           - DRX - class for general DRX scan, with sub-classes:
            * Solar - class for solar tracking
            * Jovian - class for Jovian tracking
    
Most class contain 'validate' attribute functions that can be used to determine if the 
project/run/scan are valid or not given the constraints of
the ADP system.

In addition to providing the means for creating interferometer definition files from 
scratch, this module also includes a simple parser for ID files.

.. versionadded:: 1.2.4
"""

# Python2 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info < (3,):
    range = xrange
    
import os
import re
import copy
import math
import pytz
import ephem
import weakref
from textwrap import fill as tw_fill
from datetime import datetime, timedelta

from lsl.transform import Time
from lsl.astro import utcjd_to_unix, MJD_OFFSET, DJD_OFFSET
from lsl.astro import date as astroDate, get_date as astroGetDate
from lsl.common.color import colorfy

from lsl.common.mcsADP import mjdmpm_to_datetime
from lsl.common.adp import freq_to_word, word_to_freq, fC
from lsl.common.stations import LWAStation, get_full_stations, lwa1
from lsl.reader.drx import FILTER_CODES as DRXFilters
from lsl.reader.drx import FRAME_SIZE as DRXSize
from lsl.common import sdfADP as sdf

from lsl.misc import telemetry
telemetry.track_module()


__version__ = '0.1'
__all__ = ['Observer', 'ProjectOffice', 'Project', 'Run', 'Scan', 'DRX', 'Solar', 'Jovian', 'parse_idF',  'get_scan_start_stop', 'is_valid', '__version__', '__revision__', '__all__']


_UTC = pytz.utc
_DRSUCapacityTB = 10

_MAX_ALT_PHASE_CENTERS = 10


class Observer(object):
    """Class to hold information about an observer."""
    
    def __init__(self, name, id, first=None, last=None):
        self.name = name
        self.first = first
        self.last = last
        self.id = int(id)
        
    def __str__(self):
        return "%s (#%i)" % (self.name, self.id)
        
    def __repr__(self):
        return "<%s.%s name='%s', id=%i>" % (self.__class__.__module__,
                                             self.__class__.__name__,
                                             self.name, self.id)
        
    def join_name(self):
        if self.first != '':
            self.name = ', '.join([self.last, self.first])
        else:
            self.name = self.last
        
    def split_name(self):
        try:
            self.last, self.first = self.name.split(', ', 1)
        except ValueError:
            self.last = self.name
            self.first = ''


class ProjectOffice(object):
    """Class to hold comments from the LWA object office.  This class isn't really 
    needed to create ID files, but it is helpful for parsing ID files."""
    
    def __init__(self, project=None, runs=None, scans=None):
        self.project = project
        if runs is None:
            self.runs = []
        else:
            self.runs = runs
        if scans is None:
            self.scans = []
        else:
            self.scans = scans
            
    def __repr__(self):
        return "<%s.%s runs=%s, scans=%s>" % (self.__class__.__module__,
                                              self.__class__.__name__,
                                              repr(self.runs), repr(self.scans))


class Project(object):
    """
    Class to hold all the information about a specific interferometer run for a 
    project/proposal.
    """
    
    def __init__(self, observer, name, id, runs=None, comments=None, project_office=None):
        if not isinstance(observer, Observer):
            raise TypeError("Expected 'observer' to be an Observer")
        self.observer = observer
        self.name = name
        self.id = id
        self.comments = comments
        self.runs = sdf._TypedParentList(Run, self)
        if runs is not None:
            if isinstance(runs, (list, tuple)):
                self.runs.extend(runs)
            else:
                self.runs.append(runs)
        if project_office is None:
            self.project_office = ProjectOffice()
        else:
            if not isinstance(project_office, ProjectOffice):
                raise TypeError("Expected 'project_office' to be a ProjectOffice")
            self.project_office = project_office
            
    def __str__(self):
        return "%s: %s with %i run(s) for %s" % (self.id, self.name, len(self.runs), str(self.observer))
        
    def update(self):
        """Update the various runs that are part of this project."""
        
        for ses in self.runs:
            ses.update()
            
    def validate(self, verbose=False):
        """Examine all of the runs and all of their scans to check
        for validity.  If everything is valid, return True.  Otherwise, return
        False."""
        
        self.update()
        
        failures = 0
        runCount = 1
        if len(self.id) > 8:
            if verbose:
                print("[%i] Project ID is too long" % (os.getpid(),))
            failures += 1
            
        for run in self.runs:
            if verbose:
                print("[%i] Validating run %i" % (os.getpid(), runCount))
            if not run.validate(verbose=verbose):
                failures += 1
                
            runCount += 1
            
        if failures == 0:
            return True
        else:
            return False
            
    @staticmethod
    def _render_file_size(size):
        """Convert a file size in bytes to a easy-to-use string."""
        
        units = 'B'
        if size >= 1024**4:
            size /= 1024.0**4
            units = 'TB'
        elif size >= 1024**3:
            size /= 1024.0**3
            units = 'GB'
        elif size >= 1024**2:
            size /= 1024.0**2
            units = 'MB'
        elif size >= 1024**1:
            size /= 1024.0**1
            units = 'kB'
            
        return "%.2f %s" % (size, units)
        
    @staticmethod
    def _render_bandwidth(filter, FILTER_CODES):
        """Convert a filter number to an easy-to-use string."""
        
        if FILTER_CODES[filter] > 1e6:
            return "%.3f MHz" % (FILTER_CODES[filter]/1e6,)
        elif FILTER_CODES[filter] > 1e3:
            return "%.3f kHz" % (FILTER_CODES[filter]/1e3,)
        else:
            return "%.3f Hz" % (FILTER_CODES[filter],)
            
    def append(self, newRun):
        """Add a new run to the list of runs."""
        
        self.runs.append(newRun)
        
    def render(self, run=0, verbose=False):
        """Create a run definition file that corresponds to the specified 
        run.  Returns the ID file's contents as a string."""
        
        if not self.validate(verbose=verbose) :
            raise RuntimeError("Invalid run/scan parameters.  Aborting.")
        if run >= len(self.runs):
            raise IndexError("Invalid run index")
        
        self.runs[run].update()
        self.runs[run].scans.sort()
        for obs in self.runs[run].scans:
            obs.dur = obs.get_duration()
            
        ses = self.runs[run]
        try:
            # Try to pull out the project office comments about the run
            pos = self.project_office.runs[run]
        except:
            pos = None
        try:
            # Try to pull out the project office comments about the scans
            poo = self.project_office.scans[run]
        except:
            poo = []
        # Enforce that the number of project office scan comments match the
        # actual number of scans
        while (len(ses.scans) - len(poo)) > 0:
            poo.append(None)
            
        # Combine the run comments together in an intelligent fashion
        ## Observer comments
        if ses.ucfuser is not None:
            clean = ''
            if ses.comments:
                clean = sdf._usernameRE.sub('', ses.comments)
            ses.comments = 'ucfuser:%s' % ses.ucfuser
            if len(clean) > 0:
                ses.comments += ';;%s' % clean
        ## Project office comments, including the data return method
        if pos != 'None' and pos is not None:
            pos = 'Requested data return method is %s;;%s' % (ses.data_return_method, pos)
            
        ## PI Information
        output = ""
        output = "%sPI_ID            %s\n" % (output, self.observer.id)
        output = "%sPI_NAME          %s\n" % (output, self.observer.name)
        output = "%s\n" % output
        
        ## Project Information
        output = "%sPROJECT_ID       %s\n" % (output, self.id)
        output = "%sPROJECT_TITLE    %s\n" % (output, self.name)
        output = "%sPROJECT_REMPI    %s\n" % (output, self.comments[:4090] if self.comments else 'None provided')
        output = "%sPROJECT_REMPO    %s\n" % (output, self.project_office.project)
        output = "%s\n" % output
        
        ## Run Information
        output = "%sRUN_ID           %s\n" % (output, ses.id)
        output = "%sRUN_TITLE        %s\n" % (output, 'None provided' if ses.name is None else ses.name)
        output = "%sRUN_STATIONS     %s\n" % (output, ','.join([station.id for station in ses.stations]))
        output = "%sRUN_CHANNELS     %i\n" % (output, ses.corr_channels)
        output = "%sRUN_INTTIME      %.3f\n" % (output, ses.corr_inttime)
        output = "%sRUN_BASIS        %s\n" % (output, ses.corr_basis)
        output = "%sRUN_REMPI        %s\n" % (output, ses.comments[:4090] if ses.comments else 'None provided')
        output = "%sRUN_REMPO        %s\n" % (output, "Requested data return method is %s" % ses.data_return_method if pos == 'None' or pos is None else pos[:4090])
        output = "%s\n" % output
                    
        ## Scans
        for i,obs in enumerate(ses.scans):
            obsID = i + 1
            
            output = "%sSCAN_ID          %i\n" % (output, obsID)
            output = "%sSCAN_TARGET      %s\n" % (output, obs.target)
            output = "%sSCAN_INTENT      %s\n" % (output, obs.intent)
            output = "%sSCAN_REMPI       %s\n" % (output, obs.comments[:4090] if obs.comments else 'None provided')
            output = "%sSCAN_REMPO       %s\n" % (output, "Estimated raw data volume for this scan is %s per station; %s total" % (self._render_file_size(obs.dataVolumeStation), self._render_file_size(obs.dataVolume)) if poo[i] == 'None' or poo[i] == None else poo[i])
            output = "%sSCAN_START_MJD   %i\n" % (output, obs.mjd)
            output = "%sSCAN_START_MPM   %i\n" % (output, obs.mpm)
            output = "%sSCAN_START       %s\n" % (output, obs.start.strftime("%Z %Y/%m/%d %H:%M:%S") if isinstance(obs.start, datetime) else obs.start)
            output = "%sSCAN_DUR         %i\n" % (output, obs.dur)
            output = "%sSCAN_DUR+        %s\n" % (output, obs.duration)
            output = "%sSCAN_MODE        %s\n" % (output, obs.mode)
            if obs.mode == 'TRK_RADEC':
                output = "%sSCAN_RA          %.9f\n" % (output, obs.ra)
                output = "%sSCAN_DEC         %+.9f\n" % (output, obs.dec)
                if obs.pm[0] != 0.0 or obs.pm[1] != 0.0:
                    output = "%sSCAN_PM_RA       %+.1f\n" % (output, obs.pm[0])
                    output = "%sSCAN_PM_DEC      %+.1f\n" % (output, obs.pm[1])
            output = "%sSCAN_FREQ1       %i\n" % (output, obs.freq1)
            output = "%sSCAN_FREQ1+      %.9f MHz\n" % (output, obs.frequency1/1e6)
            output = "%sSCAN_FREQ2       %i\n" % (output, obs.freq2)
            output = "%sSCAN_FREQ2+      %.9f MHz\n" % (output, obs.frequency2/1e6)
            output = "%sSCAN_BW          %i\n" % (output, obs.filter)
            output = "%sSCAN_BW+         %s\n" % (output, self._render_bandwidth(obs.filter, obs.FILTER_CODES))
            ## Alternate phase centers
            if len(obs.alt_phase_centers) > 0:
                output = "%sSCAN_ALT_N             %i\n" % (output, len(obs.alt_phase_centers))
                for i,phase_center in enumerate(obs.alt_phase_centers):
                    output = "%sSCAN_ALT_TARGET[%i]    %s\n" % (output, i+1, phase_center.target)  
                    output = "%sSCAN_ALT_INTENT[%i]    %s\n" % (output, i+1, phase_center.intent) 
                    output = "%sSCAN_ALT_RA[%i]        %.9f\n" % (output, i+1, phase_center.ra)  
                    output = "%sSCAN_ALT_DEC[%i]       %+.9f\n" % (output, i+1, phase_center.dec)
                    if phase_center.pm[0] != 0.0 or phase_center.pm[1] != 0.0:
                        output = "%sSCAN_ALT_PM_RA[%i]       %+.1f\n" % (output, i+1, phase_center.pm[0])
                        output = "%sSCAN_ALT_PM_DEC[%i]      %+.1f\n" % (output, i+1, phase_center.pm[1])
                        
            ## ASP filter setting
            if obs.aspFlt != -1:
                output = "%sSCAN_ASP_FLT     %i\n" % (output, obs.aspFlt)
            ## DRX gain
            if obs.gain != -1:
                output = "%sSCAN_DRX_GAIN    %i\n" % (output, obs.gain)
            output = "%s\n" % output
            
        return output
        
    def writeto(self, filename, run=0, verbose=False, overwrite=False):
        """Create a run definition file that corresponds to the specified 
        run and write it to the provided filename."""
        
        if os.path.exists(filename) and not overwrite:
            raise RuntimeError("'%s' already exists" % filename)
            
        output = self.render(run=run, verbose=verbose)
        with open(filename, 'w') as fh:
            fh.write(output)
            
    def generate_sdfs(self, starting_session_id=1, run=0, verbose=False):
        """Convert the ID file into a collection of `lsl.common.sdfADP.Project` instances
        that can be used to write SD files."""
        
        if not self.validate(verbose=verbose) :
            raise RuntimeError("Invalid run/scan parameters.  Aborting.")
        if run >= len(self.runs):
            raise IndexError("Invalid run index")
        
        self.runs[run].update()
        self.runs[run].scans.sort()
        for obs in self.runs[run].scans:
            obs.dur = obs.get_duration()
            
        ses = self.runs[run]
        try:
            # Try to pull out the project office comments about the run
            pos = self.project_office.runs[run]
        except:
            pos = None
        try:
            # Try to pull out the project office comments about the scans
            poo = self.project_office.scans[run]
        except:
            poo = []
        # Enforce that the number of project office scan comments match the
        # actual number of scans
        while (len(ses.scans) - len(poo)) > 0:
            poo.append(None)
            
        # Build the SDFs
        ## Setup the common information
        start = mjdmpm_to_datetime(ses.scans[0].get_mjd(), ses.scans[0].get_mpm())
        new_observer = sdf.Observer(copy.deepcopy(self.observer.name), copy.deepcopy(self.observer.id))
        ## Go!
        sdfs = []
        for i,station in enumerate(ses.stations):
            ### Session
            session = sdf.Session("%s - %s (%i of %i)" % (ses.name, station.id, i+1, len(ses.stations)), 
                                  starting_session_id, observations=[], station=station)
            session.set_drx_beam(1)
            session.set_ucf_username('eLWA/%s_%s_%s_%04i' % (self.id, start.strftime('%y%m%d'), start.strftime('%H%M'), ses.id))
            session.set_data_return_method('UCF')
            
            ## Project Office
            new_projoff = sdf.ProjectOffice(project=copy.deepcopy(self.project_office.project), 
                                            sessions=copy.deepcopy([self.project_office.runs[run],]), 
                                            observations=copy.deepcopy([self.project_office.scans[run],]))
            
            ## Observations
            for o,obs in enumerate(ses.scans):
                obs_start = mjdmpm_to_datetime(obs.mjd, obs.mpm)
                if isinstance(obs, DRX):
                    ### Apply the proper motion to generate the SDFs
                    delta_epoch = (obs.mjd + obs.mpm/1000.0/86400.0 - 51544.5) / 365.25
                    ra = obs.ra + delta_epoch * obs.pm[0]/math.cos(obs.dec*math.pi/180)/1000.0/3600.0/15.0
                    dec = obs.dec + delta_epoch * obs.pm[1]/1000.0/3600.0
                    comments = ''
                    if comments is not None:
                        comments += obs.comments
                    if obs.pm[0] != 0.0 or obs.pm[1] != 0.0:
                        comments = comments+";;Applied proper motion of %+.1f mas/yr in RA and %+.1f mas/yr in dec" % (obs.pm[0], obs.pm[1])
                        
                    new_obs = sdf.DRX(obs.intent, obs.target, _UTC.localize(obs_start), obs.duration, 
                                      ra, dec, 
                                      obs.frequency1, obs.frequency2, obs.filter, 
                                      gain=obs.gain, max_snr=False, comments=comments)
                elif isinstance(obs, Solar):
                    new_obs = sdf.Solar(obs.intent, obs.target, _UTC.localize(obs_start), obs.duration, 
                                        obs.frequency1, obs.frequency2, obs.filter, 
                                        gain=obs.gain, max_snr=False, comments=obs.comments)
                elif isinstance(obs, Jovian):
                    new_obs = sdf.Jovian(obs.intent, obs.target, _UTC.localize(obs_start), obs.duration, 
                                         obs.frequency1, obs.frequency2, obs.filter, 
                                         gain=obs.gain, max_snr=False, comments=obs.comments)
                else:
                    raise RuntimeError("This should never happen")
                session.append(new_obs)
                
                npc = len(obs.alt_phase_centers)
                for p,phase_center in enumerate(reversed(obs.alt_phase_centers)):
                    cid = npc - p
                    alt_t, alt_i, alt_r, alt_d = phase_center.target, phase_center.intent, phase_center.ra, phase_center.dec
                    alt_pr, alt_pd = phase_center.pm
                    
                    ### Apply the proper motion to generate the SDFs
                    delta_epoch = (obs.mjd + obs.mpm/1000.0/86400.0 - 51544.5) / 365.25
                    alt_r = alt_r + delta_epoch * alt_pr/math.cos(alt_d*math.pi/180)/1000.0/3600.0/15.0
                    alt_d = alt_d + delta_epoch * alt_pd/1000.0/3600.0
                    
                    try:
                        new_projoff.observations[0][o]
                    except IndexError:
                        new_projoff.observations[0][o] = None
                    if new_projoff.observations[0][o] is None:
                        new_projoff.observations[0][o] = ''
                    new_projoff.observations[0][o] = "alttarget%i:%s;;altintent%i:%s;;altra%i:%.9f;;altdec%i:%+.9f;;%s" % (cid, alt_t, cid, alt_i, cid, alt_r, cid, alt_d, new_projoff.observations[0][o])
                    
            ## Project
            project = sdf.Project(new_observer, "%s - %s (%i of %i)" % (self.name, station.id, i+1, len(ses.stations)), 
                                  copy.deepcopy(self.id), sessions=[session,], comments=copy.deepcopy(self.comments), 
                                  project_office=new_projoff)
            if project.project_office.sessions[0] is None:
               project.project_office.sessions[0] = ''
            project.project_office.sessions[0] = "corrchannels:%i;;corrinttime:%.3f;;corrbasis:%s;;origuser:%s;;origreturn:%s;;%s" % (ses.corr_channels, ses.corr_inttime, ses.corr_basis, ses.ucfuser, ses.data_return_method.lower(), project.project_office.sessions[0])
            
            ## Save an increment the session ID
            sdfs.append(project)
            starting_session_id += 1
            
        # Done
        return sdfs


class Run(object):
    """Class to hold all of the scans in an interferometer run."""
    
    def __init__(self, name, id, scans=None, data_return_method='DRSU', comments=None, corr_channels=256, corr_inttime=1.0, corr_basis='linear', stations=get_full_stations()):
        self.name = name
        self.id = int(id)
        self.scans = sdf._TypedParentList(Scan, self)
        if scans is not None:
            if isinstance(scans, (tuple, list)):
                self.scans.extend(scans)
            else:
                self.scans.append(scans)
        self.data_return_method = data_return_method
        self.ucfuser = None
        self.comments = comments
        
        self.corr_inttime = corr_inttime
        self.corr_channels = corr_channels
        self.corr_basis = corr_basis
        self.stations = sdf._TypedParentList(LWAStation, None, stations)
        
    def __str__(self):
        return "%i: %s with %i scans and correlator setup:\n  channels: %i\n  int. time: %f\n  basis: %s\n  stations: %s" % (self.id, self.name, len(self.scans), self.corr_channels, self.corr_inttime, self.corr_basis, " ".join([s.id for s in self.stations]))
        
    def set_stations(self, stations):
        """
        Update the stations used by the project for source computations.
        """
        
        self.stations = sdf._TypedParentList(LWAStation, None, stations)
        self.update()
        
    def append(self, newScan):
        """Add a new Scan to the list of scans."""
        
        self.scans.append(newScan)
        
    def set_correlator_channels(self, value):
        """Set the number of spectrometer channels to generate, 0 to disable."""
        
        self.corr_channels = int(value)
        
    def set_correlator_inttime(self, value):
        """Set the number of spectrometer FFT integrations to use, 0 to disable."""
        
        self.corr_inttime = float(value)
        
    def set_correlator_basis(self, value):
        """Set the correlator output polarization basis."""
        
        if value.lower() not in ('linear', 'circular', 'stokes'):
            raise ValueError("Unknown polarization basis: %s" % value)
        self.corr_basis = value
        
    def set_data_return_method(self, method):
        """Set the data return method for the run.  Valid values are: UCF, DRSU, and 
        'USB Harddrives'."""
        
        if method not in ('UCF', 'DRSU', 'USB Harddrives'):
            raise ValueError("Unknown data return method: %s" % method)
            
        self.data_return_method = method
        
    def set_ucf_username(self, username):
        """Set the username to use for UCF data copies."""
        
        self.ucfuser = username
        
    def update(self):
        """Update the various scans in the run."""
        
        for obs in self.scans:
            obs.update()
            
    def validate(self, verbose=False):
        """Examine all of the scans associated with the run to check
        for validity.  If everything is valid, return True.  Otherwise, return
        False."""
        
        self.update()
        
        failures = 0
        totalData = 0.0
        if self.id < 1 or self.id > 9999:
            if verbose:
                print("[%i] Error: Invalid run ID number '%i'" % (os.getpid(), self.id))
            failures += 1
            
        if len(self.stations) < 2:
            if verbose:
                print("[%i] Error: Need at least two stations to form an interferometer" % (os.getpid(),))
            failures += 1
        station_count = {}
        for station in self.stations:
            try:
                station_count[station.id] += 1
            except KeyError:
                station_count[station.id] = 1
        for station in station_count:
            if station_count[station] != 1:
                if verbose:
                    print("[%i] Error: Station '%s' is included %i times" % (os.getpid(), station, station_count[station]))
                failures += 1
        if self.corr_inttime < 0.1 or self.corr_inttime > 10.0:
            if verbose:
                print("[%i] Error: Invalid correlator integration time '%.3f s'" % (os.getpid(), self.corr_inttime))
            failures += 1
        if self.corr_channels < 16 or self.corr_channels > 32768 or self.corr_channels % 2:
            if verbose:
                print("[%i] Error: Invalid correlator channel count '%i'" % (os.getpid(), self.corr_channels))
            failures += 1
        if self.corr_basis.lower() not in (None, '', 'linear', 'circular', 'stokes'):
            if verbose:
                print("[%i] Error: Invalid correlator output polarization basis '%s'" % (os.getpid(), self.corr_basis))
            failures += 1
            
        scanCount = 1
        for obs in self.scans:
            if verbose:
                print("[%i] Validating scan %i" % (os.getpid(), scanCount))
                
            if not obs.validate(verbose=verbose):
                failures += 1
            totalData += obs.dataVolume
                
            if scanCount > 1:
                if obs.filter != self.scans[scanCount-2].filter:
                    if verbose:
                        print("[%i] Error: Filter code changes at scan %i" % (os.getpid(), scanCount))
                    failures += 1
                    
            scanCount += 1
            
        # Make sure that the scans don't overlap
        sObs = self.scans
        
        for i in range(len(sObs)):
            maxOverlaps = 1
            overlaps = []
            nOverlaps = 0

            for j in range(len(sObs)):
                if verbose and i != j:
                    print("[%i] Checking for overlap between scans %i and %i" % (os.getpid(), i+1, j+1))

                cStart = int(sObs[j].mjd)*24*3600*1000 + int(sObs[j].mpm)
                cStop = cStart + int(sObs[j].dur)
                pStart = int(sObs[i].mjd)*24*3600*1000 + int(sObs[i].mpm)
                pStop = pStart + int(sObs[i].dur)

                if pStart >= cStart and pStart < cStop:
                    nOverlaps += 1
                    
                    if i != j:
                        overlaps.append(j)
            
            if nOverlaps > maxOverlaps:
                if verbose:
                    print("[%i] Error: Scan %i overlaps with %s" % (os.getpid(), i+1, ','.join(["%i" % (j+1) for j in overlaps])))
                failures += 1
            
        if totalData >= (len(self.stations)*_DRSUCapacityTB*1024**4):
            if verbose:
                print("[%i] Error: Total data volume for run exceeds per-station %i TB DRSU limit" % (os.getpid(), _DRSUCapacityTB,))
            failures += 1
        
        if failures == 0:
            return True
        else:
            return False
            
    def __eq__(self, other):
        if isinstance(other, Run):
            self.observations.sort()
            other.observations.sort()
            
            startSelf = self.observations[0].mjd + self.observations[0].mpm / (1000.0*3600.0*24.0)
            startOther = other.observations[0].mjd + other.observations[0].mpm / (1000.0*3600.0*24.0)
            return startSelf == startOther
        else:
            raise TypeError("Unsupported type: '%s'" % type(other).__name__)
            
    def __ne__(self, other):
        if isinstance(other, Run):
            self.observations.sort()
            other.observations.sort()
            
            startSelf = self.observations[0].mjd + self.observations[0].mpm / (1000.0*3600.0*24.0)
            startOther = other.observations[0].mjd + other.observations[0].mpm / (1000.0*3600.0*24.0)
            return startSelf != startOther
        else:
            raise TypeError("Unsupported type: '%s'" % type(other).__name__)
            
    def __gt__(self, other):
        if isinstance(other, Run):
            self.observations.sort()
            other.observations.sort()
            
            startSelf = self.observations[0].mjd + self.observations[0].mpm / (1000.0*3600.0*24.0)
            startOther = other.observations[0].mjd + other.observations[0].mpm / (1000.0*3600.0*24.0)
            return startSelf > startOther
        else:
            raise TypeError("Unsupported type: '%s'" % type(other).__name__)
            
    def __ge__(self, other):
        if isinstance(other, Run):
            self.observations.sort()
            other.observations.sort()
            
            startSelf = self.observations[0].mjd + self.observations[0].mpm / (1000.0*3600.0*24.0)
            startOther = other.observations[0].mjd + other.observations[0].mpm / (1000.0*3600.0*24.0)
            return startSelf >= startOther
        else:
            raise TypeError("Unsupported type: '%s'" % type(other).__name__)
            
    def __lt__(self, other):
        if isinstance(other, Run):
            self.observations.sort()
            other.observations.sort()
            
            startSelf = self.observations[0].mjd + self.observations[0].mpm / (1000.0*3600.0*24.0)
            startOther = other.observations[0].mjd + other.observations[0].mpm / (1000.0*3600.0*24.0)
            return startSelf < startOther
        else:
            raise TypeError("Unsupported type: '%s'" % type(other).__name__)
            
    def __le__(self, other):
        if isinstance(other, Run):
            self.observations.sort()
            other.observations.sort()
            
            startSelf = self.observations[0].mjd + self.observations[0].mpm / (1000.0*3600.0*24.0)
            startOther = other.observations[0].mjd + other.observations[0].mpm / (1000.0*3600.0*24.0)
            return startSelf <= startOther
        else:
            raise TypeError("Unsupported type: '%s'" % type(other).__name__)


class Scan(object):
    """
    Class to hold the specifics of a scans.  It currently
    handles TRK_RADEC, TRK_SOL, and TRK_JOV.
    """
    
    _parent = None
    
    id = 1
    FILTER_CODES = DRXFilters

    def __init__(self, target, intent, start, duration, mode, ra, dec, frequency1, frequency2, filter, gain=-1, pm=[0.0, 0.0], comments=None):
        self.target = target
        self.intent = intent
        self.ra = float(ra) * (12.0/math.pi if type(ra).__name__ == 'Angle' else 1.0)
        self.dec = float(dec)* (180.0/math.pi if type(dec).__name__ == 'Angle' else 1.0)
        if pm is None:
            pm = [0.0, 0.0]
        self.pm = [pm[0], pm[1]]
        self.start = start
        if isinstance(duration, timedelta):
            # Make sure the number of microseconds agree with milliseconds
            us = int(round(duration.microseconds/1000.0))*1000
            duration = timedelta(days=duration.days, seconds=duration.seconds, microseconds=us)
        self.duration = str(duration)
        self.mode = mode
        self.beamDipole = None
        self.frequency1 = float(frequency1)
        self.frequency2 = float(frequency2)
        self.filter = int(filter)
        self.comments = comments
        
        self.mjd = None
        self.mpm = None
        self.dur = None
        self.freq1 = None
        self.freq2 = None
        self.dataVolumeStation = None
        self.dataVolume = None
        
        self.aspFlt = -1
        
        self.gain = int(gain)
        
        self.alt_phase_centers = []
        
        self.update()
        
    def __str__(self):
        """Return a nice string to describe the scan."""
        
        return "%s Scan of '%s':\n Intent %s\n Start %s\n Duration %s\n Filter: %i\n Frequency: %.3f; %.3f\n RA: %.6f hr @ %+.3f mas/yr\n Dec. %.6f d @ +%.3f mas/yr\n" % (self.mode, self.target, self.intent, self.start, self.duration, self.filter, self.frequency1, self.frequency2, self.ra, self.pm[0], self.dec, self.pm[1])
        
    def update(self):
        """Update the computed parameters from the string values."""
        
        stations = [lwa1,]
        if self._parent is not None:
            stations = self._parent.stations
            
        # If we have a datetime instance, make sure we have an integer
        # number of milliseconds
        if isinstance(self.start, datetime):
            us = self.start.microsecond
            us = int(round(us/1000.0))*1000
            self.start = self.start.replace(microsecond=us)
        self.duration = str(self.duration)
        
        self.mjd = self.get_mjd()
        self.mpm = self.get_mpm()
        self.dur = self.get_duration()
        self.freq1 = self.get_frequency1()
        self.freq2 = self.get_frequency2()
        self.dataVolumeStation = self.estimate_bytes()
        self.dataVolume = self.dataVolumeStation*len(stations)
        
        # Update the associated alternate phase centers
        for phase_center in self.alt_phase_centers:
            phase_center.update()
            
    def set_start(self, start):
        """Set the scan start time."""
        
        self.start = start
        self.update()
        
    def get_mjd(self):
        """Return the modified Julian Date corresponding to the date/time of the
        self.start string."""
        
        utc = sdf.parse_time(self.start)		## TODO:  We need to get the station informaiton here somehow
        utc = Time(utc, format=Time.FORMAT_PY_DATE)
        return int(utc.utc_mjd)

    def get_mpm(self):
        """Return the number of milliseconds between the date/time specified in the
        self.start string and the previous UT midnight."""
        
        utc = sdf.parse_time(self.start)		## TODO:  We need to get the station informaiton here somehow
        utcMidnight = datetime(utc.year, utc.month, utc.day, 0, 0, 0, tzinfo=_UTC)
        diff = utc - utcMidnight
        return int(round((diff.seconds + diff.microseconds/1000000.0)*1000.0))

    def get_duration(self):
        """Parse the self.duration string with the format of HH:MM:SS.SSS to return the
        number of milliseconds in that period."""
        
        fields = self.duration.split(':')
        if len(fields) == 3:
            out = int(fields[0])*3600.0
            out += int(fields[1])*60.0
            out += float(fields[2])
        elif len(fields) == 2:
            out = int(fields[0])*60.0
            out += float(fields[1])
        else:
            out = float(fields[0])
            
        return int(round(out*1000.0))

    def set_duration(self, duration):
        """Set the scan duration."""
        
        if isinstance(duration, timedelta):
            # Make sure the number of microseconds agree with milliseconds
            us = int(round(duration.microseconds/1000.0))*1000
            duration = timedelta(days=duration.days, seconds=duration.seconds, microseconds=us)
        self.duration = str(duration)
        self.update()
        
    def get_frequency1(self):
        """Return the number of "tuning words" corresponding to the first frequency."""
        
        freq1 = freq_to_word(self.frequency1)
        self.frequency1 = word_to_freq(freq1)
        return freq1
        
    def set_frequency1(self, frequency1):
        """Set the frequency in Hz corresponding to tuning 1."""
        
        self.frequency1 = float(frequency1)
        self.update()
        
    def get_frequency2(self):
        """Return the number of "tuning words" corresponding to the second frequency."""
        
        freq2 = freq_to_word(self.frequency2)
        self.frequency2 = word_to_freq(freq2)
        return freq2
        
    def set_frequency2(self, frequency2):
        """Set the frequency in Hz correpsonding to tuning 2."""
        
        self.frequency2 = float(frequency2)
        self.update()
        
    def add_alt_phase_center(self, target_or_apc, intent=None, ra=None, dec=None, pm=None):
        """Add an alternate phase center to the scan."""
        
        if isinstance(target_or_apc, AlternatePhaseCenter):
            apc = target_or_apc
        else:
            apc = AlternatePhaseCenter(target_or_apc, intent, ra, dec, pm=pm)
        self.alt_phase_centers.append(apc)
        
    def estimate_bytes(self):
        """Estimate the data volume for the specified type and duration of 
        scans.  For DRX:
        
            bytes = duration * sample_rate / 4096 * 4128 bytes * 2 tunings * 2 pols.
        """
        
        try:
            nFrames = self.get_duration()/1000.0 * self.FILTER_CODES[self.filter] / 4096
        except KeyError:
            nFrames = 0
        nBytes = nFrames * DRXSize * 4
        return nBytes
        
    def get_fixed_body(self):
        """Return an ephem.Body object corresponding to where the scan is 
        pointed.  None if the scan mode is TBN."""
        
        pnt = ephem.FixedBody()
        pnt.name = self.target
        pnt._ra = self.ra / 12.0 * math.pi
        pnt._dec = self.dec / 180.0 * math.pi
        pnt._pmra = self.pm[0]
        pnt._pmdec = self.pm[1]
        pnt._epoch = ephem.J2000
        return pnt
        
    def compute_visibility(self):
        """Return the fractional visibility of the target during the scan 
        period."""
        
        stations = [lwa1,]
        if self._parent is not None:
            stations = self._parent.stations
            
        vis_list = []
        for station in stations:
            lwa = station.get_observer()
            pnt = self.get_fixed_body()
            
            vis = 0
            cnt = 0
            dt = 0.0
            while dt <= self.dur/1000.0:
                lwa.date = self.mjd + (self.mpm/1000.0 + dt)/3600/24.0 + MJD_OFFSET - DJD_OFFSET
                pnt.compute(lwa)
                
                cnt += 1
                if pnt.alt > 0:
                    vis += 1
                    
                dt += 300.0
                
            vis_list.append(float(vis)/float(cnt))
        return min(vis_list)
        
    def validate(self, verbose=False):
        """Evaluate the scan and return True if it is valid, False otherwise."""
        
        self.update()
        
        stations = [lwa1,]
        if self._parent is not None:
            stations = self._parent.stations
        tuning_min = max([station.interface.get_module('backend').DRX_TUNING_WORD_MIN for station in stations])
        tuning_max = min([station.interface.get_module('backend').DRX_TUNING_WORD_MAX for station in stations])
        
        failures = 0
        # Basic - Intent, duration, frequency, and filter code values
        if self.intent.lower() not in ('fluxcal', 'phasecal', 'target', 'dummy'):
            if verbose:
                print("[%s] Error: Invalid scan intent '%s'" % (os.getpid(), self.intent))
            failures += 1
        if self.dur < 1:
            if verbose:
                print("[%i] Error: Specified a duration of length zero" % os.getpid())
            failures += 1
        if self.freq1 < tuning_min or self.freq1 > tuning_max:
            if verbose:
                print("[%i] Error: Specified frequency for tuning 1 is outside of LWA tuning range" % os.getpid())
            failures += 1
        if (self.freq2 < tuning_min or self.freq2 > tuning_max) and self.freq2 != 0:
            if verbose:
                print("[%i] Error: Specified frequency for tuning 2 is outside of LWA tuning range" % os.getpid())
            failures += 1
        if self.filter not in [1, 2, 3, 4, 5, 6, 7]:
            if verbose:
                print("[%i] Error: Invalid filter code '%i'" % (os.getpid(), self.filter))
            failures += 1
            
        # Advanced - Target Visibility
        if self.ra < 0 or self.ra >= 24:
            if verbose:
                print("[%i] Error: Invalid value for RA '%.6f'" % (os.getpid(), self.ra))
            failures += 1
        if self.dec < -90 or self.dec > 90:
            if verbose:
                print("[%i] Error: Invalid value for dec. '%+.6f'" % (os.getpid(), self.dec))
            failures += 1
        if self.compute_visibility() < 1.0:
            if verbose:
                print("[%i] Error: Target is only above the horizon for %.1f%% of the scan for %s" % (os.getpid(), self.compute_visibility(station=station)*100.0, station.id))
            failures += 1
            
        # Advanced - alternate phase centers
        if len(self.alt_phase_centers) > _MAX_ALT_PHASE_CENTERS:
            if verbose:
                print("[%i] Error: too many alternate phase centers defined" % os.getpid())
            failures += 1
        for j,phase_center in enumerate(self.alt_phase_centers):
            if not phase_center.validate(verbose=verbose):
                if verbose:
                    print("[%i] Error: invalid alternate phase center %i" % (os.getpid(), j+1))
                failures += 1
                
            ## Closeness to pointing center
            pnt = self.get_fixed_body()
            alt_pnt = phase_center.get_fixed_body()
            
            lwa = stations[0].get_observer()
            lwa.date = self.mjd + (self.mpm/1000.0 + self.dur/1000.0/2.0)/3600/24.0 + MJD_OFFSET - DJD_OFFSET
            pnt.compute(lwa)
            alt_pnt.compute(lwa)
            
            alt_sep = ephem.separation(pnt, alt_pnt) * 180/math.pi
            
            beam = 2.0*74e6/max([self.frequency1, self.frequency2])
            if alt_sep > beam/2.0:
                if verbose:
                    print("[%i] Error: alternate phase center %i is %.1f degrees from pointing center" % (os.getpid(), j+1, alt_sep))
                failures += 1
                
        # Advanced - Data Volume
        if self.dataVolumeStation >= (_DRSUCapacityTB*1024**4):
            if verbose:
                print("[%i] Error: Data volume exceeds %i TB DRSU limit" % (os.getpid(), _DRSUCapacityTB))
            failures += 1
            
        # Any failures indicates a bad scan
        if failures == 0:
            return True
        else:
            return False
            
    def __eq__(self, other):
        if isinstance(other, Scan):
            startSelf = self.mjd + self.mpm / (1000.0*3600.0*24.0)
            startOther = other.mjd + other.mpm / (1000.0*3600.0*24.0)
            return startSelf == startOther
        else:
            raise TypeError("Unsupported type: '%s'" % type(other).__name__)
            
    def __ne__(self, other):
        if isinstance(other, Scan):
            startSelf = self.mjd + self.mpm / (1000.0*3600.0*24.0)
            startOther = other.mjd + other.mpm / (1000.0*3600.0*24.0)
            return startSelf != startOther
        else:
            raise TypeError("Unsupported type: '%s'" % type(other).__name__)
            
    def __gt__(self, other):
        if isinstance(other, Scan):
            startSelf = self.mjd + self.mpm / (1000.0*3600.0*24.0)
            startOther = other.mjd + other.mpm / (1000.0*3600.0*24.0)
            return startSelf > startOther
        else:
            raise TypeError("Unsupported type: '%s'" % type(other).__name__)
            
    def __ge__(self, other):
        if isinstance(other, Scan):
            startSelf = self.mjd + self.mpm / (1000.0*3600.0*24.0)
            startOther = other.mjd + other.mpm / (1000.0*3600.0*24.0)
            return startSelf >= startOther
        else:
            raise TypeError("Unsupported type: '%s'" % type(other).__name__)
            
    def __lt__(self, other):
        if isinstance(other, Scan):
            startSelf = self.mjd + self.mpm / (1000.0*3600.0*24.0)
            startOther = other.mjd + other.mpm / (1000.0*3600.0*24.0)
            return startSelf < startOther
        else:
            raise TypeError("Unsupported type: '%s'" % type(other).__name__)
            
    def __le__(self, other):
        if isinstance(other, Scan):
            startSelf = self.mjd + self.mpm / (1000.0*3600.0*24.0)
            startOther = other.mjd + other.mpm / (1000.0*3600.0*24.0)
            return startSelf <= startOther
        else:
            raise TypeError("Unsupported type: '%s'" % type(other).__name__)


class DRX(Scan):
    """
    Required Arguments:
     * scan target
     * scan intent
     * scan start date/time (UTC YYYY/MM/DD HH:MM:SS.SSS string or timezone-
       aware datetime instance)
     * scan duration (HH:MM:SS.SSS string or timedelta instance)
     * scan RA in hours, J2000.0 or ephem.hours instance
     * scan Dec in degrees, J2000.0 or ephem.hours instance
     * scan tuning frequency 1 (Hz)
     * scan tuning frequency 1 (Hz)
     * integer filter code
    
    Optional Keywords:
     * comments - comments about the scan
    """
    
    alt_phase_centers = []
    
    def __init__(self, target, intent, start, duration, ra, dec, frequency1, frequency2, filter, gain=-1, pm=[0.0, 0.0], comments=None):
        Scan.__init__(self, target, intent, start, duration, 'TRK_RADEC', ra, dec, frequency1, frequency2, filter, gain=gain, pm=pm, comments=comments)
        
    def set_ra(self, ra):
        """Set the pointing RA."""
        
        self.ra = float(ra) * (12.0/math.pi if type(ra).__name__ == 'Angle' else 1.0)
        
    def set_dec(self, dec):
        """Set the pointing Dec."""
        
        self.dec = float(dec)* (180.0/math.pi if type(dec).__name__ == 'Angle' else 1.0)
        
    def set_pm(self, ra_dec):
        """Set the proper motion of the target in mas/yr."""
        
        self.pm[0] = ra_dec[0]
        self.pm[1] = ra_dec[1]


class Solar(Scan):
    """Sub-class of DRX specifically for Solar DRX scans.   It features a
    reduced number of parameters needed to setup the scan.
    
    Required Arguments:
     * scan target
     * scan intent
     * scan start date/time (UTC YYYY/MM/DD HH:MM:SS.SSS string or timezone-
       aware datetime instance)
     * scan duration (HH:MM:SS.SSS string or timedelta instance)
     * scan tuning frequency 1 (Hz)
     * scan tuning frequency 1 (Hz)
     * integer filter code
    
    Optional Keywords:
     * comments - comments about the scan
    """
    
    def __init__(self, target, intent, start, duration, frequency1, frequency2, filter, gain=-1, comments=None):
        Scan.__init__(self, target, intent, start, duration, 'TRK_SOL', 0.0, 0.0, frequency1, frequency2, filter, gain=gain, comments=comments)
        
    def get_fixed_body(self):
        """Return an ephem.Body object corresponding to where the scan is 
        pointed.  None if the scan mode is TBN."""
        
        return ephem.Sun()


class Jovian(Scan):
    """Sub-class of DRX specifically for Jovian DRX scans.   It features a
    reduced number of parameters needed to setup the scan.
    
    Required Arguments:
     * scan target
     * scan intent
     * scan start date/time (UTC YYYY/MM/DD HH:MM:SS.SSS string or timezone-
       aware datetime instance)
     * scan duration (HH:MM:SS.SSS string or timedelta instance)
     * scan tuning frequency 1 (Hz)
     * scan tuning frequency 1 (Hz)
     * integer filter code
    
    Optional Keywords:
     * comments - comments about the scan
    """
    
    def __init__(self, target, intent, start, duration, frequency1, frequency2, filter, gain=-1, comments=None):
        Scan.__init__(self, target, intent, start, duration, 'TRK_JOV', 0.0, 0.0, frequency1, frequency2, filter, gain=gain, comments=comments)

    def get_fixed_body(self):
        """Return an ephem.Body object corresponding to where the scan is 
        pointed.  None if the scan mode is TBN."""
        
        return ephem.Jupiter()


class AlternatePhaseCenter(object):
    """Class to hold an alternate phase center for a scan."""
    
    def __init__(self, target, intent, ra, dec, pm=[0.0, 0.0]):
        self.target = target
        self.intent = intent
        self.ra = float(ra) * (12.0/math.pi if type(ra).__name__ == 'Angle' else 1.0)
        self.dec = float(dec)* (180.0/math.pi if type(dec).__name__ == 'Angle' else 1.0)
        if pm is None:
            pm = [0.0, 0.0]
        self.pm = [pm[0], pm[1]]
        
    def set_ra(self, ra):
        """Set the pointing RA."""
        
        self.ra = float(ra) * (12.0/math.pi if type(ra).__name__ == 'Angle' else 1.0)
        
    def set_dec(self, dec):
        """Set the pointing Dec."""
        
        self.dec = float(dec)* (180.0/math.pi if type(dec).__name__ == 'Angle' else 1.0)
        
    def set_pm(self, ra_dec):
        """Set the proper motion of the target in mas/yr."""
        
        self.pm[0] = ra_dec[0]
        self.pm[1] = ra_dec[1]
        
    def update(self):
        """Update the computed parameters from the string values."""
        
        pass
        
    def get_fixed_body(self):
        """Return an ephem.Body object corresponding to where the scan is 
        pointed.  None if the scan mode is TBN."""
        
        pnt = ephem.FixedBody()
        pnt.name = self.target
        pnt._ra = self.ra / 12.0 * math.pi
        pnt._dec = self.dec / 180.0 * math.pi
        pnt._pmra = self.pm[0]
        pnt._pmdec = self.pm[1]
        pnt._epoch = ephem.J2000
        return pnt
    
    def validate(self, verbose=False):
        """Basic validation of the pointing, that's it."""
        
        failures = 0
        
        ## Intent
        if self.intent.lower() not in ('fluxcal', 'phasecal', 'target'):
            if verbose:
                print("[%s] Error: Invalid alternate phase center intent '%s'" % (os.getpid(), self.intent))
            failures += 1
            
        ## Pointing
        if self.ra < 0 or self.ra >= 24:
            if verbose:
                print("[%i] Error: Invalid alternate phase center value for RA '%.6f'" % (os.getpid(), self.ra))
            failures += 1
        if self.dec < -90 or self.dec > 90:
            if verbose:
                print("[%i] Error: Invalid alternate phase center value for dec. '%+.6f'" % (os.getpid(), self.dec))
            failures += 1
            
        # Any failures indicates a bad scan
        if failures == 0:
            return True
        else:
            return False


def _parse_create_scan_object(obs_temp, alt_temps=[], verbose=False):
    """Given a obs_temp dictionary of scan parameters, return a complete Scan object 
    corresponding to those values."""
    
    # If the scan ID is 0, do nothing.
    if obs_temp['id'] == 0:
        return None
        
    # Create a time string for the start time in UTC.  This is a little tricky 
    # because of the rounding to the nearest millisecond which has to be done
    # to the datetime object.
    start = Time(obs_temp['mjd'] + obs_temp['mpm'] / 1000.0 / 3600.0 / 24.0, format='MJD').utc_py_date
    start += timedelta(microseconds=(int(round(start.microsecond/1000.0)*1000.0)-start.microsecond))
    utcString = start.strftime("UTC %Y %m %d %H:%M:%S.%f")
    utcString = utcString[:-3]
    
    # Build up a string representing the scan duration.
    try:
        dur = obs_temp['duration']
        dur = float(dur) / 1000.0
        durString = '%02i:%02i:%06.3f' % (dur/3600.0, (dur%3600.0)/60.0, dur%60.0)
    except:
        pass
        
    # Convert the frequencies from "tuning words" to Hz
    f1 = word_to_freq(obs_temp['freq1'])
    f2 = word_to_freq(obs_temp['freq2'])
    
    # Get the mode and run through the various cases
    mode = obs_temp['mode']
    if verbose:
        print("[%i] Scan %i is mode %s" % (os.getpid(), obs_temp['id'], mode))
        
    if mode == 'TRK_RADEC':
        obsOut = DRX(obs_temp['target'], obs_temp['intent'], utcString, durString, obs_temp['ra'], obs_temp['dec'], f1, f2, obs_temp['filter'], gain=obs_temp['gain'], pm=obs_temp['pm'], comments=obs_temp['comments'])
    elif mode == 'TRK_SOL':
        obsOut = Solar(obs_temp['target'], obs_temp['intent'], utcString, durString, f1, f2, obs_temp['filter'], gain=obs_temp['gain'], comments=obs_temp['comments'])
    elif mode == 'TRK_JOV':
        obsOut = Jovian(obs_temp['target'], obs_temp['intent'], utcString, durString, f1, f2, obs_temp['filter'], gain=obs_temp['gain'], comments=obs_temp['comments'])
    else:
        raise RuntimeError("Invalid mode encountered: %s" % mode)
        
    # Add in the alternate phase centers
    if obs_temp['nAlt'] != len(alt_temps):
        raise RuntimeError("Mis-match between SCAN_ALT_N and the number of alternate phase centers")
    for alt_temp in alt_temps:
        obsOut.add_alt_phase_center(alt_temp['target'], alt_temp['intent'], alt_temp['ra'], alt_temp['dec'], pm=alt_temp['pm'])
        
    # Set the ASP/FEE values
    obsOut.aspFlt = copy.deepcopy(obs_temp['aspFlt'])
    
    # Force the scan to be updated
    obsOut.update()
    
    # Return the newly created Scan object
    return obsOut


def parse_idf(filename, verbose=False):
    """
    Given a filename, read the file's contents into the IDF instance and return
    that instance.
    """
    
    # Create the keyword regular expression to deal with various indicies included 
    # in the keywords
    kwdRE = re.compile(r'(?P<keyword>[A-Z_0-9\+]+)(\[(?P<id1>[0-9]+?)\])?(\[(?P<id2>[0-9]+?)\])?(\[(?P<id3>[0-9]+?)\])?(\[(?P<id4>[0-9]+?)\])?')
    
    # Create the metatag regular expression to deal with spectrometer mode settings
    metaRE = re.compile(r'\{.*\}')
    
    # Create empty objects to get things started.  Values will get filled in as they
    # are found in the file
    po = ProjectOffice()
    observer = Observer('observer_name', 0)
    project = Project(observer, 'project_name', 'project_id', project_office=po)
    run = Run('run_name', 0, scans=[])
    project.runs = [run,]
    project.project_office.runs = []
    project.project_office.scans = [[],]
    
    obs_temp = {'id': 0, 'target': '', 'intent': '', 'ra': 0.0, 'dec': 0.0, 'pm':[0.0, 0.0], 'start': '', 'duration': '', 'mode': '', 
                'freq1': 0, 'freq2': 0, 'filter': 0, 'comments': None, 'nAlt': 0, 'gain': -1, 
                'aspFlt': -1}
    alt_temp = {'id': 0, 'target': '', 'intent': '', 'ra': 0.0, 'dec': 0.0, 'pm':[0.0, 0.0]}
    alt_temps = []
    
    # Loop over the file
    with open(filename, 'r') as fh:
        for line in fh:
            # Trim off the newline character and skip blank lines
            line = line.replace('\n', '')
            if len(line) == 0 or line.isspace():
                continue
            
            # Split into a keyword, value pair and run it through the regular expression
            # to deal with any indicies present
            try:
                keywordSection, value = line.split(None, 1)
            except:
                continue
            
            mtch = kwdRE.match(keywordSection)
            keyword = mtch.group('keyword')
        
            ids = [-1, -1, -1, -1]
            for i in range(4):
                try:
                    ids[i] = int(mtch.group('id%i' % (i+1)))
                except TypeError:
                    pass
                
            # Skip over the observer comment lines (denoted by a plus sign at the end) 
            # of the keyword
            if keyword[-1] == '+':
                continue
            
            # Observer Info
            if keyword == 'PI_ID':
                project.observer.id = int(value)
                continue
            if keyword == 'PI_NAME':
                project.observer.name = value
                project.observer.split_name()
                continue
            
            # Project/Proposal Info
            if keyword == 'PROJECT_ID':
                project.id = value
                continue
            if keyword == 'PROJECT_TITLE':
                project.name = value
                continue
            if keyword == 'PROJECT_REMPI':
                project.comments = value
                continue
            if keyword == 'PROJECT_REMPO':
                project.project_office.project = value
                continue
            
            # Run Info
            if keyword == 'RUN_ID':
                project.runs[0].id = int(value)
                continue
            if keyword == 'RUN_TITLE':
                project.runs[0].name = value
                continue
            if keyword == 'RUN_STATIONS':
                use_stations = []
                possible = get_full_stations()
                for field in value.split(','):
                    field = field.strip().rstrip()
                    if field.lower() == 'all':
                        use_stations = copy.deepcopy(possible)
                        break
                    else:
                        for station in possible:
                            if station.id == field:
                                use_stations.append(station)
                                break
                project.runs[0].stations  = use_stations
            if keyword == 'RUN_CHANNELS':
                project.runs[0].corr_channels = int(value)
                continue
            if keyword == 'RUN_INTTIME':
                project.runs[0].corr_inttime = float(value)
                continue
            if keyword == 'RUN_BASIS':
                project.runs[0].corr_basis = value
                continue
            if keyword == 'RUN_REMPI':
                mtch = sdf._usernameRE.search(value)
                if mtch is not None:
                    project.runs[0].ucfuser = mtch.group('username')
                    if mtch.group('subdir') is not None:
                        project.runs[0].ucfuser = os.path.join(project.runs[0].ucfuser, mtch.group('subdir'))
                project.runs[0].comments = sdf._usernameRE.sub('', value)
                continue
            if keyword == 'RUN_REMPO':
                project.project_office.runs.append(None)
                parts = value.split(';;', 1)
                first = parts[0]
                try:
                    second = parts[1]
                except IndexError:
                    second = ''
                
                if first[:31] == 'Requested data return method is':
                    # Catch for project office comments that are data return related
                    project.runs[0].data_return_method = first[32:]
                    project.project_office.runs[0] = second
                else:
                    # Catch for standard (not data related) project office comments
                    project.project_office.runs[0] = value
                continue
            
            # Scan Info
            if keyword == 'SCAN_ID':
                if obs_temp['id'] != 0:
                    project.runs[0].scans.append( _parse_create_scan_object(obs_temp, alt_temps=alt_temps, verbose=verbose) )
                    alt_temp = {'id': 0, 'target': '', 'intent': '', 'ra': 0.0, 'dec': 0.0, 'pm':[0.0, 0.0]}
                    alt_temps = []
                obs_temp['id'] = int(value)
                project.project_office.scans[0].append( None )
            
                if verbose:
                    print("[%i] Started scan %i" % (os.getpid(), int(value)))
                
                continue
            if keyword == 'SCAN_TARGET':
                obs_temp['target'] = value
                continue
            if keyword == 'SCAN_INTENT':
                obs_temp['intent'] = value
                continue
            if keyword == 'SCAN_REMPI':
                obs_temp['comments'] = value
                continue
            if keyword == 'SCAN_REMPO':
                project.project_office.scans[0][-1] = value
                continue
            if keyword == 'SCAN_START_MJD':
                obs_temp['mjd'] = int(value)
                continue
            if keyword == 'SCAN_START_MPM':
                obs_temp['mpm'] = int(value)
                continue
            if keyword == 'SCAN_DUR':
                obs_temp['duration'] = int(value)
                continue
            if keyword == 'SCAN_MODE':
                obs_temp['mode'] = value
                continue
            if keyword == 'SCAN_RA':
                obs_temp['ra'] = float(value)
                continue
            if keyword == 'SCAN_DEC':
                obs_temp['dec'] = float(value)
                continue
            if keyword == 'SCAN_PM_RA':
                obs_temp['pm'][0] = float(value)
                continue
            if keyword == 'SCAN_PM_DEC':
                obs_temp['pm'][1] = float(value)
                continue
            if keyword == 'SCAN_FREQ1':
                obs_temp['freq1'] = int(value)
                continue
            if keyword == 'SCAN_FREQ2':
                obs_temp['freq2'] = int(value)
                continue
            if keyword == 'SCAN_BW':
                obs_temp['filter'] = int(value)
                continue
            
            # Alternate phase centers
            if keyword == 'SCAN_ALT_N':
                obs_temp['nAlt'] = int(value)
                continue
            if keyword == 'SCAN_ALT_TARGET':
                if len(alt_temps) == 0:
                    alt_temps.append( copy.deepcopy(alt_temp) )
                    alt_temps[-1]['id'] = ids[0]
                    alt_temps[-1]['target'] = value
                else:
                    if alt_temps[-1]['id'] != ids[0]:
                        alt_temps.append( copy.deepcopy(alt_temps[-1]) )
                        alt_temps[-1]['id'] = ids[0]
                    alt_temps[-1]['target'] = value
                continue
            if keyword == 'SCAN_ALT_INTENT':
                if len(alt_temps) == 0:
                    alt_temps.append( copy.deepcopy(alt_temp) )
                    alt_temps[-1]['id'] = ids[0]
                    alt_temps[-1]['intent'] = value
                else:
                    if alt_temps[-1]['id'] != ids[0]:
                        alt_temps.append( copy.deepcopy(alt_temp) )
                        alt_temps[-1]['id'] = ids[0]
                    alt_temps[-1]['intent'] = value
                continue
            if keyword == 'SCAN_ALT_RA':
                if len(alt_temps) == 0:
                    alt_temps.append( copy.deepcopy(alt_temp) )
                    alt_temps[-1]['id'] = ids[0]
                    alt_temps[-1]['ra'] = float(value)
                else:
                    if alt_temps[-1]['id'] != ids[0]:
                        alt_temps.append( copy.deepcopy(alt_temp) )
                        alt_temps[-1]['id'] = ids[0]
                    alt_temps[-1]['ra'] = float(value)
                continue
            if keyword == 'SCAN_ALT_DEC':
                if len(alt_temps) == 0:
                    alt_temps.append( copy.deepcopy(alt_temp) )
                    alt_temps[-1]['id'] = ids[0]
                    alt_temps[-1]['dec'] = float(value)
                else:
                    if alt_temps[-1]['id'] != ids[0]:
                        alt_temps.append( copy.deepcopy(alt_temp) )
                        alt_temps[-1]['id'] = ids[0]
                    alt_temps[-1]['dec'] = float(value)
                continue
            if keyword == 'SCAN_ALT_PM_RA':
                if len(alt_temps) == 0:
                    alt_temps.append( copy.deepcopy(alt_temp) )
                    alt_temps[-1]['id'] = ids[0]
                    alt_temps[-1]['pm'][0] = float(value)
                else:
                    if alt_temps[-1]['id'] != ids[0]:
                        alt_temps.append( copy.deepcopy(alt_temp) )
                        alt_temps[-1]['id'] = ids[0]
                    alt_temps[-1]['pm'][0] = float(value)
                continue
            if keyword == 'SCAN_ALT_PM_DEC':
                if len(alt_temps) == 0:
                    alt_temps.append( copy.deepcopy(alt_temp) )
                    alt_temps[-1]['id'] = ids[0]
                    alt_temps[-1]['pm'][1] = float(value)
                else:
                    if alt_temps[-1]['id'] != ids[0]:
                        alt_temps.append( copy.deepcopy(alt_temp) )
                        alt_temps[-1]['id'] = ids[0]
                    alt_temps[-1]['pm'][1] = float(value)
                continue
            # Run wide settings at the end of the scans
            if keyword == 'SCAN_ASP_FLT':
                obs_temp['aspFlt'] = int(value)
                continue
            if keyword == 'SCAN_DRX_GAIN':
                obs_temp['gain'] = int(value)
                continue
            
            # Keywords that might indicate this is a SDF
            if keyword in ('SESSION_ID', 'SESSION_DRX_BEAM'):
                raise RuntimeError("Invalid keyword encountered: %s" % keyword)
            
        # Create the final scan
        if obs_temp['id'] != 0:
            project.runs[0].scans.append( _parse_create_scan_object(obs_temp, alt_temps=alt_temps, verbose=verbose) )
            
    # Return the project
    return project


def get_scan_start_stop(obs):
    """
    Given a scan, get the start and stop times (returned as a two-
    element tuple of UTC datetime instances).
    """
    
    # UNIX timestamp for the start
    tStart = utcjd_to_unix(obs.mjd + MJD_OFFSET)
    tStart += obs.mpm / 1000.0
    
    # UNIX timestamp for the stop
    tStop = tStart +  obs.dur / 1000.0
    
    # Conversion to a timezone-aware datetime instance
    tStart = _UTC.localize( datetime.utcfromtimestamp(tStart) )
    tStop  = _UTC.localize( datetime.utcfromtimestamp(tStop ) )
    
    # Make sure we have an integer number of milliseconds
    ## Start
    us = tStart.microsecond
    us = int(round(us/1000.0))*1000
    tStart = tStart.replace(microsecond=us)
    ## Stop
    us = tStop.microsecond
    us = int(round(us/1000.0))*1000
    tStop = tStop.replace(microsecond=us)
    
    # Return
    return tStart, tStop


def is_valid(filename, verbose=False):
    """
    Given a filename, see if it is valid IDF file or not.
    """
    
    passes = 0
    failures = 0
    try:
        proj = parse_idf(filename)
        passes += 1
        if verbose:
            print(colorfy("Parser - {{%green OK}}"))
            
        valid = proj.validate()
        if valid:
            passes += 1
            if verbose:
                print(colorfy("Validator - {{%green OK}}"))
        else:
            failures += 1
            if verbose:
                print(colorfy("Validator - {{%red {{%bold FAILED}}}}"))
                
    except IOError as e:
        raise e
    except:
        failures += 1
        if verbose:
            print(colorfy("Parser - {{%red {{%bold FAILED}}}}"))
            
    if verbose:
        print("---")
        print("%i passed / %i failed" % (passes, failures))
        
    return False if failures else True
