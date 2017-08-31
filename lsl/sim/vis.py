# -*- coding: utf-8 -*-

# Python3 compatiability
from __future__ import print_function
import sys
if sys.version_info > (3,):
	xrange = range
	long = int

"""
Module for generating simulated arrays and visibility data.  The chief 
functions of this module are:

buildSimArray
  given a station object, a list of stands, and a list of frequencies, build 
  a AIPY AntennaArray-like object.  This module can also generate AntennaArray 
  objects with positional errors by setting the 'PosError' keyword to a 
  positive value.

buildSimData
  given a SimArray and a list of aipy.src sources, build up a collection of 
  visibilities for a given set of Julian dates

scaleData
  given a dictionary of simulated visibilities from buildSimData, apply 
  antenna-based gains and delays to the visibilities

shiftData
  given a dictionary of simulated visibilities from buildSimData, shift the uvw 
  coordinates of the visibilities.
  .. note::

	This only changes the uvw values and does not phase-shift the data.

The format of the data dictionaries mentioned above is:

primary keys
  The primary keys store the major aspects of the visiblity data, e.g., 
  frequency coverage, baseline pairs, uvw coordinates, etc.  Valid keys are:
    * *freq* - list of frequencies used in Hz
    * *isMasked* - whether or not the visibility data have been masked 
      (numpy.compress'd)
    * *bls* - list of baselines in (stand 1, stand2) format
    * *uvw* - list of uvw coordinates as 3-element numpy arrays
    * *vis* - list of visibility numpy arrays
    * *wgt* - list of weight arrays with the same length as the visilitity arrays
    * *msk* - list of mask arrays used for the data.  1 = masked, 0 = valid
    * *jd*  - list of Julian dates associated with each list element

secondary keys
  The bls, uvw, vis, wgt, msk, and jd primary keys also have secondary keys that 
  indicate which polarizations are being stored.  Valid keys are:
    * *xx*
    * *yy*
    * *xy*
    * *yx*

In addition to simulation functions, this module includes buildGriddedImage
which takes a dictionary of visibilities and returns and aipy.im.ImgW object.

.. versionchanged:: 0.3.0
	This module was formerly called lsl.sim.sim
	
.. versionchanged:: 0.5.0
	Moved buildGriddedImage to the :mod:`lsl.imaging.utils` module.

.. versionchanged:: 1.0.1
	Switched over to a new C-based simulation package
	
.. versionchanged:: 1.0.2
	Added in a function, addBaselineNoise, to help bring noise into the 
	simulations
	
.. versionchanged:: 1.0.3
	Moved the calculateSEFD function into lsl.misc.rfutils
	Changed the meaning of the ForceGaussian parameter of the buildSimArray()
	function to be the Gaussian full width at half maximum in degrees
"""

import os
import aipy
import copy
import math
import ephem
import numpy
from scipy.interpolate import interp1d

from lsl import astro
from lsl.common.paths import data as dataPath
from lsl.correlator import uvUtils
from lsl.common.stations import lwa1
from lsl.common.constants import c as speedOfLight

from _simfast import FastVis

__version__ = '0.6'
__revision__ = '$Rev$'
__all__ = ['srcs', 'RadioEarthSatellite', 'BeamAlm', 'Antenna', 'AntennaArray', 
		 'buildSimArray', 'buildSimData', 'scaleData', 'shiftData', 'addBaselineNoise', 
		 '__version__', '__revision__', '__all__']


# A dictionary of bright sources in the sky to use for simulations
srcs = aipy.src.get_catalog(srcs=['Sun', 'Jupiter', 'cas', 'crab', 'cyg', 'her', 'sgr', 'vir'])


class RadioEarthSatellite(object):
	"""
	Implement a aipy.amp.RadioBody-lime simulation object for an Earth-
	orbiting satellite using a two-line element set.
	"""
	
	def __init__(self, tle, tfreq, tpower=0.0, tbw=1.0e6, ionref=(0.,0.)):
		"""
		Initialize the class using:
		 * a three-element list of strings that specify the two-line element
		   set of the satellite,
		 * a list of frequencies (in GHz) where the satellite transmits, 
		 * the transmitter power in W, and
		 * the transmission bandwidth in Hz.
		 
		.. note::
			For calculating the power received on the ground this class 
			assumes that the signal is emitted isotropically.
		"""
		
		# Location
		self.Body = ephem.readtle(*tle)
		self.src_name = self.Body.name
		
		# Transmitter - tuning, power, and bandwidth
		try:
			self.tfreq = list(tfreq)
		except TypeError:
			self.tfreq = [tfreq,]
		self.tpower = tpower
		self.tbw = tbw
		
		# Ionospheric distortion
		self.ionref = list(ionref)
		
		# Shape (unresolved)
		self.srcshape = list((0.,0.,0.))
		
	def __getattr__(self, nm):
		"""
		First try to access attribute from this class, but if that fails, 
		try to get it from the underlying PyEphem object.
		"""
		
		try:
			return object.__getattr__(self, nm)
		except AttributeError:
			return self.Body.__getattribute__(nm)
			
	def compute(self, observer):
		"""
		Update coordinates relative to the provided observer.  Must be
		called at each time step before accessing information.
		"""
		
		self.Body.compute(observer)
		self.update_jys(observer.get_afreqs())
		
	def get_crds(self, crdsys, ncrd=3):
		"""
		Return the coordinates of this location in the desired coordinate
		system ('eq','top') in the current epoch.  If ncrd=2, angular
		coordinates (ra/dec or az/alt) are returned, and if ncrd=3,
		xyz coordinates are returned.
		"""
		
		assert(crdsys in ('eq','top'))
		assert(ncrd in (2,3))
		if crdsys == 'eq':
			if ncrd == 2:
				return (self.ra, self.dec)
			return aipy.coord.radec2eq((self.ra, self.dec))
		else:
			if ncrd == 2:
				return (self.az, self.alt)
			return aipy.coord.azalt2top((self.az, self.alt))
			
	def update_jys(self, afreqs):
		"""
		Update fluxes relative to the provided observer.  Must be
		called at each time step before accessing information.
		"""
		
		# Setup
		r = self.Body.range				# m
		v = self.Body.range_velocity		# m/s
		self.jys = numpy.zeros_like(afreqs)
		
		# Compute the flux coming from the satellite assuming isotropic emission
		self._jys = self.tpower / (4*numpy.pi*r**2) / self.tbw		# W / m^2 / Hz
		self._jys /= 10**-26								# Jy
		
		try:
			dFreq = afreqs[1]-afreqs[0]
		except IndexError:
			dFreq = afreqs[0,1]-afreqs[0,0]
		for f in self.tfreq:
			## Apply the Doppler shift
			fPrime = f / (1 + v/speedOfLight)
			
			## Figure out which frequency bin is within one channel of the
			## shifted signal.  If it's close, set it to a 
			diff = numpy.abs(fPrime - afreqs)*1e9
			good = numpy.where( diff <= self.tbw/2.0 )
			self.jys[good] = self._jys * min([1.0, dFreq*1e9/self.tbw])
				
	def get_jys(self):
		"""
		Return the fluxes vs. freq that should be used for simulation.
		"""
		
		return self.jys


class BeamAlm(aipy.amp.BeamAlm):
	"""
	AIPY-based representation of a beam model where each pointing has a 
	response defined as a polynomial in frequency, and the spatial 
	distributions of these coefficients decomposed into spherical 
	harmonics.
	
	This differs from the AIPY version in that the response() method 
	accepts two and three-dimensions arrays of topocentric coordinates, 
	similar to what aipy.img.ImgW.get_top() produces, and computes the 
	beam response at all points.
	"""
	
	def __init__(self, freqs, lmax=8, mmax=8, deg=7, nside=64, coeffs={}):
		"""
		AIPY __init__() function.
		
		lmax = maximum spherical harmonic term
		mmax = maximum spherical harmonic term in the z direction
		deg = order of polynomial to used for mapping response of each pointing
		nside = resolution of underlying HealpixMap to use
		coeffs = dictionary of polynomial term (integer) and corresponding Alm 
		coefficients (see healpix.py doc).
		"""
		
		aipy.phs.Beam.__init__(self, freqs)
		self.alm = [aipy.healpix.Alm(lmax, mmax) for i in range(deg+1)]
		self.hmap = [aipy.healpix.HealpixMap(nside, scheme='RING', interp=True)
			for a in self.alm]
		for c in coeffs:
			if c < len(self.alm):
				self.alm[-1-c].set_data(coeffs[c])
		self._update_hmap()
		
	def __responsePrimitive(self, top):
		"""
		Copy of the original aipy.amp.BeamAlm.response function.
		
		Return beam response across active band for specified topocentric 
		coordinates (x=E,y=N,z=UP). x,y,z may be multiple coordinates.  
		Returns 'x' pol (rotate pi/2 for 'y').
		"""
		
		top = [aipy.healpix.mk_arr(c, dtype=numpy.double) for c in top]
		px,wgts = self.hmap[0].crd2px(*top, interpolate=1)
		poly = numpy.array([numpy.sum(h.map[px] * wgts, axis=-1) for h in self.hmap])
		rv = numpy.polyval(poly, numpy.reshape(self.afreqs, (self.afreqs.size, 1)))
		return rv
		
	def response(self, top):
		"""
		Return beam response across active band for specified topocentric 
		coordinates (x=E,y=N,z=UP). x,y,z may be multiple coordinates.  
		Returns 'x' pol (rotate pi/2 for 'y').
		
		.. note::
			This function also accepts two and three-dimensions arrays of 
			topocentric coordinates, similar to what aipy.img.ImgW.get_top() 
			produces, and computes the beam response at all points
		"""
		
		test = numpy.array(top)
		x,y,z = top
		
		if len(test.shape) == 1:
			temp = self.__responsePrimitive((x,y,z))
			
		elif len(test.shape) == 2:
			temp = numpy.zeros((self.afreqs.size,)+test.shape[1:])
			for i in xrange(temp.shape[1]):
				temp[:,i] = numpy.squeeze(self.__responsePrimitive((x[i],y[i],z[i])))
				
		elif len(test.shape) == 3:
			temp = numpy.zeros((self.afreqs.size,)+test.shape[1:])
			for i in xrange(temp.shape[1]):
				for j in xrange(temp.shape[2]):
					temp[:,i,j] = numpy.squeeze(self.__responsePrimitive((x[i,j],y[i,j],z[i,j])))
					
		else:
			raise ValueError("Cannot compute response for %s" % str(test.shape))
			
		return temp


class Beam2DGaussian(aipy.amp.Beam2DGaussian):
	"""
	AIPY-based representation of a 2-D Gaussian beam pattern, with default 
	setting for a flat beam.  The width parameters denote the 'sigma' of
	the Gaussian in radians.
	
	This differs from the AIPY version in that the response() method 
	accepts two and three-dimensions arrays of topocentric coordinates, 
	similar to what aipy.img.ImgW.get_top() produces, and computes the 
	beam response at all points.
	
	.. versionchanged:: 1.0.3
		Clarified what 'xwidth' and 'ywidth' are.
	"""
	
	def __init__(self, freqs, xwidth=numpy.Inf, ywidth=numpy.Inf):
		"""
		AIPY __init__() function.
		
		xwidth = angular width (radians) in EW direction
		ywidth = angular width (radians) in NS direction
		"""
		
		aipy.phs.Beam.__init__(self, freqs)
		self.xwidth, self.ywidth = xwidth, ywidth
		
	def __responsePrimitive(self, top):
		"""
		Copy of the original aipy.amp.Beam2DGaussian.response function.
		
		Return beam response across active band for specified topocentric 
		coordinates: (x=E,y=N,z=UP). x,y,z may be arrays of multiple 
		coordinates.  Returns 'x' linear polarization (rotate pi/2 for 'y').
		"""
		
		x,y,z = top
		x,y = numpy.arcsin(x)/self.xwidth, numpy.arcsin(y)/self.ywidth
		resp = numpy.sqrt(numpy.exp(-(x**2 + y**2)/2.0))
		resp = numpy.resize(resp, (self.afreqs.size, resp.size))
		return resp
		
	def response(self, top):
		"""
		Return beam response across active band for specified topocentric 
		coordinates (x=E,y=N,z=UP). x,y,z may be multiple coordinates.  
		Returns 'x' pol (rotate pi/2 for 'y').
		
		.. note::
			This function also accepts two and three-dimensions arrays of 
			topocentric coordinates, similar to what aipy.img.ImgW.get_top() 
			produces, and computes the beam response at all points
		"""
		
		test = numpy.array(top)
		x,y,z = top
		
		if len(test.shape) == 1:
			temp = self.__responsePrimitive((x,y,z))
			
		elif len(test.shape) == 2:
			temp = numpy.zeros((self.afreqs.size,)+test.shape[1:])
			for i in xrange(temp.shape[1]):
				temp[:,i] = numpy.squeeze(self.__responsePrimitive((x[i],y[i],z[i])))
				
		elif len(test.shape) == 3:
			temp = numpy.zeros((self.afreqs.size,)+test.shape[1:])
			for i in xrange(temp.shape[1]):
				for j in xrange(temp.shape[2]):
					temp[:,i,j] = numpy.squeeze(self.__responsePrimitive((x[i,j],y[i,j],z[i,j])))
					
		else:
			raise ValueError("Cannot compute response for %s" % str(test.shape))
			
		return temp


class BeamPolynomial(aipy.amp.BeamPolynomial):
	"""
	AIPY-based representation of a Gaussian beam model whose width varies 
	with azimuth angle and with frequency.
	
	This differs from the AIPY version in that the response() method 
	accepts two and three-dimensions arrays of topocentric coordinates, 
	similar to what aipy.img.ImgW.get_top() produces, and computes the 
	beam response at all points.
	"""
	def __init__(self, freqs, poly_azfreq=numpy.array([[.5]])):
		"""
		AIPY __init__() function.
		
		poly_azfreq = a 2D polynomial in cos(2*n*az) for first axis and 
		in freq**n for second axis.
		"""
		
		self.poly = poly_azfreq
		aipy.phs.Beam.__init__(self, freqs)
		self.poly = poly_azfreq
		self._update_sigma()
		
	def __responsePrimitive(self, top):
		"""
		Copy of the original aipy.amp.Beam2DGaussian.response function.
		
		Return beam response across active band for specified topocentric 
		coordinates (x=E,y=N,z=UP). x,y,z may be multiple coordinates.  
		Returns 'x' pol (rotate pi/2 for 'y').
		"""
		
		az,alt = aipy.coord.top2azalt(top)
		zang = numpy.pi/2 - alt
		if zang.size == 1:
			zang = numpy.array([zang]); zang.shape = (1,)
			az = numpy.array([az])
			az.shape = (1,)
		a = 2 * numpy.arange(self.poly.shape[0], dtype=numpy.float)
		a.shape = (1,) + a.shape
		az.shape += (1,)
		zang.shape += (1,)
		a = numpy.cos(numpy.dot(az, a))
		a[:,0] = 0.5
		s = numpy.dot(a, self.sigma)
		return numpy.sqrt(numpy.exp(-(zang/s)**2)).transpose()
		
	def response(self, top):
		"""
		Return beam response across active band for specified topocentric 
		coordinates (x=E,y=N,z=UP). x,y,z may be multiple coordinates.  
		Returns 'x' pol (rotate pi/2 for 'y').
		
		.. note::
			This function also accepts two and three-dimensions arrays of 
			topocentric coordinates, similar to what aipy.img.ImgW.get_top() 
			produces, and computes the beam response at all points
		"""
		
		test = numpy.array(top)
		x,y,z = top
		
		if len(test.shape) == 1:
			temp = self.__responsePrimitive((x,y,z))
			
		elif len(test.shape) == 2:
			temp = numpy.zeros((self.afreqs.size,)+test.shape[1:])
			for i in xrange(temp.shape[1]):
				temp[:,i] = numpy.squeeze(self.__responsePrimitive((x[i],y[i],z[i])))
				
		elif len(test.shape) == 3:
			temp = numpy.zeros((self.afreqs.size,)+test.shape[1:])
			for i in xrange(temp.shape[1]):
				for j in xrange(temp.shape[2]):
					temp[:,i,j] = numpy.squeeze(self.__responsePrimitive((x[i,j],y[i,j],z[i,j])))
					
		else:
			raise ValueError("Cannot compute response for %s" % str(test.shape))
			
		return temp


class Beam(aipy.amp.Beam):
	"""
	AIPY-based representation of a flat (gain=1) antenna beam pattern.
	
	This differs from the AIPY version in that the response() method 
	accepts two and three-dimensions arrays of topocentric coordinates, 
	similar to what aipy.img.ImgW.get_top() produces, and computes the 
	beam response at all points.
	"""
	
	def __responsePrimitive(self, top):
		"""
		Copy of the original aipy.amp.Beam.response function.
		
		Return the (unity) beam response as a function of position.
		"""
		
		x,y,z = numpy.array(top)
		return numpy.ones((self.afreqs.size, x.size))
		
	def response(self, top):
		"""
		Return beam response across active band for specified topocentric 
		coordinates (x=E,y=N,z=UP). x,y,z may be multiple coordinates.  
		Returns 'x' pol (rotate pi/2 for 'y').
		
		.. note::
			This function also accepts two and three-dimensions arrays of 
			topocentric coordinates, similar to what aipy.img.ImgW.get_top() 
			produces, and computes the beam response at all points
		"""
		
		test = numpy.array(top)
		x,y,z = top
		
		if len(test.shape) == 1:
			temp = self.__responsePrimitive((x,y,z))
			
		elif len(test.shape) == 2:
			temp = numpy.zeros((self.afreqs.size,)+test.shape[1:])
			for i in xrange(temp.shape[1]):
				temp[:,i] = numpy.squeeze(self.__responsePrimitive((x[i],y[i],z[i])))
				
		elif len(test.shape) == 3:
			temp = numpy.zeros((self.afreqs.size,)+test.shape[1:])
			for i in xrange(temp.shape[1]):
				for j in xrange(temp.shape[2]):
					temp[:,i,j] = numpy.squeeze(self.__responsePrimitive((x[i,j],y[i,j],z[i,j])))
					
		else:
			raise ValueError("Cannot compute response for %s" % str(test.shape))
			
		return temp


class Antenna(aipy.amp.Antenna):
	"""
	Modification to the aipy.amp.Antenna class to also store the stand ID 
	number in the Antenna.stand attribute.  This also add a getBeamShape 
	attribute that pulls in the old vis.getBeamShape function.
	"""

	def __init__(self, x, y, z, beam, phsoff=[0.,0.], bp_r=numpy.array([1]), bp_i=numpy.array([0]), amp=1, pointing=(0.,numpy.pi/2,0), stand=0, **kwargs):
		"""
		New init function that include stand ID number support.  From aipy.amp.Antenna:
		  * x,y z = antenna coordinates in equatorial (ns) coordinates
		  * beam = Beam object (implements response() function)
		  * phsoff = polynomial phase vs. frequency.  Phs term that is linear
		    with freq is often called 'delay'.
		  * bp_r = polynomial (in freq) modeling real component of passband
		  * bp_i = polynomial (in freq) modeling imaginary component of passband
		  * amp = overall multiplicative scaling of gain
		  * pointing = antenna pointing (az,alt).  Default is zenith.
		"""
		
		aipy.phs.Antenna.__init__(self, x,y,z, beam=beam, phsoff=phsoff)
		self.set_pointing(*pointing)
		self.bp_r = bp_r
		self.bp_i = bp_i
		self.amp = amp
		self.stand = stand
		self._update_gain()
		
	def bm_response(self, top, pol='x'):
		"""
		Return response of beam for specified polarization.
		
		.. note::
			This differs from the AIPY implementation in that the LWA X-pol.
			is oriented N-S, not E-W.
			
		.. note::
			This function also accepts two and three-dimensions arrays of 
			topocentric coordinates, similar to what img.ImgW.get_top() 
			produces, and computes the beam response at all points.
		"""
		top = numpy.array(top)
		
		def robustDot(a, b):
			"""
			Dot product that operations on multi-dimensional coordinate sets.
			"""
			
			if len(b.shape) == 1:
				temp = numpy.dot(a, b)
				
			elif len(b.shape) == 2:
				temp = numpy.zeros_like(b)
				for i in xrange(b.shape[1]):
					temp[:,i] = numpy.dot(a, b[:,i])
					
			elif len(b.shape) == 3:
				temp = numpy.zeros_like(b)
				for i in xrange(b.shape[1]):
					for j in xrange(top.shape[2]):
						temp[:,i,j] = numpy.dot(a, b[:,i,j])
						
			else:
				raise ValueError("Cannot dot a (%s) with b (%s)" % (str(a.shape), str(b.shape)))
			
			return temp
			
		top = {'y':robustDot(self.rot_pol_x, top), 
			  'x':robustDot(self.rot_pol_y, top), 
			  'l':robustDot(self.rot_pol_x, top), 
			  'r':robustDot(self.rot_pol_y, top)}[pol]
		x,y,z = top
		
		return self.beam.response((x,y,z))
		
	def get_beam_shape(self, pol='x'):
		"""
		Return a 360 by 90 by nFreqs numpy array showing the beam pattern of a
		particular antenna in the array.  The first two dimensions of the output 
		array contain the azimuth (from 0 to 359 degrees in 1 degree steps) and 
		altitude (from 0 to 89 degrees in 1 degree steps).
		"""
		
		# Build azimuth and altitude arrays.  Be sure to convert to radians
		az = numpy.zeros((360,90))
		for i in range(360):
			az[i,:] = i*numpy.pi/180.0
		alt = numpy.zeros((360,90))
		for i in range(90):
			alt[:,i] = i*numpy.pi/180.0
			
		# The beam model is computed in terms of topocentric coordinates, so make that
		# conversion right quick using the aipy.coord module.
		xyz = aipy.coord.azalt2top(numpy.concatenate([[az],[alt]]))
		
		# I cannot figure out how to do this all at once, so loop through azimuth/
		# altitude pairs
		resp = numpy.zeros((360,90,len(self.beam.freqs)))
		for i in range(360):
			for j in range(90):
				resp[i,j,:] = numpy.squeeze( self.bm_response(numpy.squeeze(xyz[:,i,j]), pol=pol) )
				
		return resp


class AntennaArray(aipy.amp.AntennaArray):
	"""
	Modification to the aipy.ant.AntennaArray class to add a function to 
	retrieve the stands stored in the AntennaArray.ants attribute.  Also add 
	a function to set the array time from a UNIX timestamp.
	
	.. versionchanged:: 1.0.1
		Added an option to set the ASP filter for simulation proposes.  
		This updates the bandpasses used by AIPY to include the antenna
		impedance mis-match and the mean ARX response.
	"""

	def get_stands(self):
		"""
		Return a numpy array listing the stands found in the AntennaArray 
		object.
		"""

		stands = []
		for ant in self.ants:
			stands.append(ant.stand)
		
		return numpy.array(stands)

	def set_unixtime(self, timestamp):
		"""
		Set the array time using a UNIX timestamp (epoch 1970).
		"""
		
		self.set_jultime(astro.unix_to_utcjd(timestamp))
		
	def set_asp_filter(self, filter='split'):
		"""
		Update the bandpasses for the antennas to include the effect of 
		the antenna impedance mis-match (IMM) and the mean LWA1 ARX 
		response.
		
		Valid filters are:
		  * split
		  * full
		  * reduced
		  * none
		  
		None is a special case where both the IMM and ARX response are 
		removed, i.e., the bandpass is unity for all frequencies.
		
		.. versionadded:: 1.0.1
		"""
		
		# Get the frequencies we need to estimate the bandpass for
		freqs = self.get_afreqs()*1e9
		if len(freqs.shape) == 2:
			freqs = freqs[0,:]
			
		if filter == 'none' or filter is None:
			# Build up the bandpass - of ones
			resp = numpy.ones(freqs.size)
			
		else:
			# Load in the LWA1 antennas so we can grab some data
			ants = lwa1.getAntennas()
			
			# Antenna impedance mis-match
			immf, immr = ants[0].response(dB=False)
			immr /= immr.max()
			immIntp = interp1d(immf, immr, kind='cubic', bounds_error=False)
			
			# Mean ARX filter response
			arxf, arxr = ants[0].arx.response(dB=False)
			for i in xrange(1, len(ants)):
				arxfi, arxri = ants[i].arx.response(dB=False)
				arxr += arxri
			arxr /= arxr.max()
			arxIntp = interp1d(arxf, arxr, kind='cubic', bounds_error=False)
			
			# Build up the bandpass
			resp = numpy.ones(freqs.size)
			resp *= immIntp(freqs)
			resp *= arxIntp(freqs)
			
		# Update the AIPY passbands
		for i in xrange(len(self.ants)):
			self[i].amp = resp
			self[i].update()
			
	def get_baseline_fast(self, i, j, src='z', map=None):
		"""Return the baseline corresponding to i,j in various coordinate 
		projections: src='e' for current equatorial, 'z' for zenith 
		topocentric, 'r' for unrotated equatorial, or a RadioBody for
		projection toward that source - fast."""
		bl = self[j] - self[i]
		
		if type(src) == str:
			if src == 'e':
				return numpy.dot(self._eq2now, bl)
			elif src == 'z':
				return numpy.dot(self._eq2zen, bl)
			elif src == 'r':
				return bl
			else:
				raise ValueError('Unrecognized source:' + src)
		try:
			if src.alt < 0:
				raise PointingError('%s below horizon' % src.src_name)
			m = src.map
		except AttributeError:
			if map is None:
				ra,dec = aipy.coord.eq2radec(src)
				m = aipy.coord.eq2top_m(self.sidereal_time() - ra, dec)
			else:
				m = map
		return numpy.dot(m, bl).transpose()
		
	def gen_uvw_fast(self, i, j, src='z', w_only=False, map=None):
		"""Compute uvw coordinates of baseline relative to provided RadioBody, 
		or 'z' for zenith uvw coordinates.  If w_only is True, only w (instead
		of (u,v,w) will be returned) - fast."""
		
		x,y,z = self.get_baseline_fast(i,j, src=src, map=map)
		
		afreqs = self.get_afreqs()
		afreqs = numpy.reshape(afreqs, (1,afreqs.size))
		if len(x.shape) == 0:
			if w_only:
				return z*afreqs
			else:
				return numpy.array([x*afreqs, y*afreqs, z*afreqs])
				
		#afreqs = numpy.reshape(afreqs, (1,afreqs.size))
		x.shape += (1,); y.shape += (1,); z.shape += (1,)
		
		if w_only:
			out = numpy.dot(z,afreqs)
		else:
			out = numpy.array([numpy.dot(x,afreqs), numpy.dot(y,afreqs), numpy.dot(z,afreqs)])
			
		return out
		
	def gen_phs_fast(self, src, i, j, mfreq=.150, ionref=None, srcshape=None, resolve_src=False, u=None, v=None, w=None):
		"""Return phasing that is multiplied to data to point to src - fast."""
		if ionref is None:
			try:
				ionref = src.ionref
			except AttributeError:
				pass
		if ionref is not None or resolve_src:
			if u is None or v is None or w is None:
				u,v,w = self.gen_uvw_fast(i,j,src=src)
		else:
			if w is None:
				w = self.gen_uvw_fast(i,j,src=src, w_only=True)
		if ionref is not None:
			w += self.refract(u, v, mfreq=mfreq, ionref=ionref)
		o = self.get_phs_offset(i,j)
		phs = numpy.exp(-1j*2*numpy.pi*(w + o))
		if resolve_src:
			if srcshape is None:
				try:
					srcshape = src.srcshape
				except AttributeError:
					pass
		if srcshape is not None:
			phs *= self.resolve_src(u, v, srcshape=srcshape)
		return phs.squeeze()
		
	def sim(self, i, j, pol='xx'):
		"""
		Simulate visibilities for the specified (i,j) baseline and 
		polarization.  sim_cache() must be called at each time step before 
		this will return valid results.
		
		This function differs from aipy.amp.AntennaArray.sim in the fact that
		*ionref* and *srcshape* are both None in the call to gen_phs and that
		*resolve_src* is set to False.
		"""
		
		assert(pol in ('xx','yy','xy','yx'))
		
		if self._cache is None:
			raise RuntimeError('sim_cache() must be called before the first sim() call at each time step.')
		elif self._cache == {}:
			return numpy.zeros_like(self.passband(i,j))
			
		s_eqs = self._cache['s_eqs']
		s_map = self._cache['s_map']
		w = self.gen_uvw_fast(i, j, src=s_eqs, map=s_map, w_only=True)
		I_sf = self._cache['jys']
		Gij_sf = self.passband(i,j)
		Bij_sf = self.bm_response(i, j, pol=pol)
		if len(Bij_sf.shape) == 2:
			Gij_sf = numpy.reshape(Gij_sf, (1, Gij_sf.size))
			
		# Get the phase of each src vs. freq, also does resolution effects
		E_sf = numpy.conjugate( self.gen_phs_fast(s_eqs, i, j, mfreq=self._cache['mfreq'], resolve_src=False, w=w) )
		try:
			E_sf.shape = I_sf.shape
		except(AttributeError):
			pass
			
		# Combine and sum over sources
		GBIE_sf = Gij_sf * Bij_sf * I_sf * E_sf
		Vij_f = GBIE_sf.sum(axis=0)
		
		return Vij_f


def buildSimArray(station, antennas, freq, jd=None, PosError=0.0, ForceFlat=False, ForceGaussian=False, verbose=False):
	"""
	Build a AIPY AntennaArray for simulation purposes.  Inputs are a station 
	object defined from the lwa_common module, a numpy array of stand 
	numbers, and a numpy array of frequencies in either Hz of GHz.  Optional 
	inputs are a Julian Date to set the array to and a positional error terms 
	that perturbs each of the stands in x, y, and z.  The output of this 
	module is an AIPY AntennaArray object.
	
	The shape of the antenna response is either flat (gain of 1 in all 
	directions), modeled by a 2-D Gaussian with the specified full width at
	half maximum in degrees, or modeled by a collection of spherical 
	harmonics that are polynomials in frequency.  The spherical harmonics 
	are used if the file 'beam_shape.npz' is found in the current directory.
	
	.. versionchanged:: 1.0.3
		Changed the meaning of the ForceGaussian parameters so that the
		Gaussian full width at half maximum in degrees is passed in.
		
	.. versionchanged:: 1.0.1
		Moved the simulation code over from AIPY to the new _simFast module.  
		This should be much faster but under the caveats that the bandpass
		and antenna gain patterns are the same for all antennas.  This 
		should be a reasonable assumption for large-N arrays.
		
		Added an option to use a 2-D Gaussian beam pattern via the ForceGaussian
		keyword.
		
	.. versionchanged:: 0.4.0
		Switched over to passing in Antenna instances generated by the
		:mod:`lsl.common.station` module instead of a list of stand ID numbers.
	"""
	
	# If the frequencies are in Hz, we need to convert to GHz
	freqs = freq.copy()
	if freqs.min() > 1e6:
		freqs /= 1.0e9
		
	# If the beam Alm coefficient file is present, build a more realistic beam 
	# response.  Otherwise, assume a flat beam
	if ForceGaussian:
		try:
			xw, yw = ForceGaussian
			xw, yw = float(xw), float(yw)
		except (TypeError, ValueError) as e:
			xw = float(ForceGaussian)
			yw = 1.0*xw
			
		# FWHM to sigma
		xw /= 2.0*numpy.sqrt(2.0*numpy.log(2.0))
		yw /= 2.0*numpy.sqrt(2.0*numpy.log(2.0))
		
		# Degrees to radians
		xw *= numpy.pi/180
		yw *= numpy.pi/180
		
		if verbose:
			print("Using a 2-D Gaussian beam with sigmas %.1f by %.1f degrees" % (xw*180/numpy.pi, yw*180/numpy.pi))
		beam = Beam2DGaussian(freqs, xw, yw)
		
	elif ForceFlat:
		if verbose:
			print("Using flat beam model")
		beam = Beam(freqs)
		
	else:
		if os.path.exists(os.path.join(dataPath, 'beam-shape.npz')):
			dd = numpy.load(os.path.join(dataPath, 'beam-shape.npz'))
			coeffs = dd['coeffs']
			
			deg = coeffs.shape[0]-1
			lmax = int((math.sqrt(1+8*coeffs.shape[1])-3)/2)
			beamShapeDict = {}
			for i in range(deg+1):
				beamShapeDict[i] = numpy.squeeze(coeffs[-1-i,:])
			try:
				dd.close()
			except AttributeError:
				pass
				
			if verbose:
				print("Using Alm beam model with %i-order freq. polynomial and %i-order sph. harmonics" % (deg, lmax))
			beam = BeamAlm(freqs, lmax=lmax, mmax=lmax, deg=deg, nside=128, coeffs=beamShapeDict)
		else:
			if verbose:
				print("Using flat beam model")
			beam = Beam(freqs)
			
	if PosError != 0:
		print("WARNING:  Creating array with positional errors between %.3f and %.3f m" % (-PosError, PosError))

	# Build an array of AIPY Antenna objects
	ants = []
	for antenna in antennas:
		top = numpy.array([antenna.stand.x, antenna.stand.y, antenna.stand.z])
		top += (2*PosError*numpy.random.rand(3)-PosError)	# apply a random positional error if needed
		top.shape = (3,)
		eq = numpy.dot( aipy.coord.top2eq_m(0.0, station.lat), top )
		eq *= 100	# m -> cm
		eq /= aipy.const.c	# cm -> s
		eq *= 1e9	# s -> ns
		
		delayCoeff = numpy.zeros(2)
		
		amp = 0*antenna.cable.gain(freq*1e9) + 1
		
		ants.append( Antenna(eq[0], eq[1], eq[2], beam, phsoff=delayCoeff, amp=amp, stand=antenna.stand.id) )

	# Combine the array of antennas with the array's location to generate an
	# AIPY AntennaArray object
	simAA = AntennaArray(ants=ants, location=station.getAIPYLocation())
	
	# Set the Julian Data for the AntennaArray object if it is provided.  The try...except
	# clause is used to deal with people who may want to pass an array of JDs in rather than
	# just one.  If one isn't provided, use the date set for the input 'station'.
	if jd is not None:
		try:
			junk = len(jd)
		except:
			simAA.set_jultime(jd)
		else:
			simAA.set_jultime(jd[0])
	else:
		simAA.date = station.date
		
	return simAA


def __buildSimData(aa, srcs, pols=['xx', 'yy', 'xy', 'yx'], jd=None, chan=None, phaseCenter='z', baselines=None, mask=None, verbose=False, count=None, max=None, flatResponse=False, resolve_src=False):
	"""
	Helper function for buildSimData so that buildSimData can be called with 
	a list of Julian Dates and reconstruct the data appropriately.
	
	.. versionchanged:: 1.0.1
		* Moved the simulation code over from AIPY to the new _simFast module.  
		  This should be much faster but under the caveats that the bandpass
		  and antenna gain patterns are the same for all antennas.  This 
		  should be a reasonable assumption for large-N arrays.
		* Added a 'flatResponse' keyword to make it easy to toggle on and off
		  the spectral and spatial response of the array for the simulation
		* Added a 'resolve_src' keyword to turn on source resolution effects
	"""
	
	rawFreqs = aa.get_afreqs()
	
	if len(rawFreqs.shape) == 2:
		nFreq = rawFreqs.shape[1]
	else:
		nFreq = rawFreqs.size
	if chan is None:
		chanMin = 0
		chanMax = -1
	else:
		chanMin = chan[0]
		chanMax = chan[-1]
		
	# Update the JD if necessary
	if jd is not None:
		if verbose:
			if count is not None and max is not None:
				print("Setting Julian Date to %.5f (%i of %i)" % (jd, count, max))
			else:
				print("Setting Julian Date to %.5f" % jd)
		aa.set_jultime(jd)
	else:
		jd = aa.get_jultime()
	Gij_sf = aa.passband(0,1)
	def Bij_sf(xyz, pol):
		Bi = aa[0].bm_response(xyz, pol=pol[0]).transpose()
		Bj = aa[1].bm_response(xyz, pol=pol[1]).transpose()
		Bij = numpy.sqrt( Bi*Bj.conj() )
		return Bij.squeeze()
	if flatResponse:
		Gij_sf *= 0.0
		Gij_sf += 1.0
		
	# Compute the source parameters
	srcs_tp = []
	srcs_eq = []
	srcs_ha = []
	srcs_dc = []
	srcs_jy = []
	srcs_fq = []
	srcs_sh = []
	if verbose:
		print("Sources Used for Simulation:")
	for name,src in srcs.iteritems():
		## Update the source's coordinates
		src.compute(aa)
		
		## Remove sources below the horizon
		srcTop = src.get_crds(crdsys='top', ncrd=3)
		srcAzAlt = aipy.coord.top2azalt(srcTop)
		if srcAzAlt[1] < 0:
			if verbose:
				print("  %s: below horizon" % name)
			continue
			
		## Topocentric coordinates for the gain pattern calculations
		srcTop.shape = (1,3)
		srcs_tp.append( srcTop )
		
		## RA/dec -> equatorial 
		srcEQ = src.get_crds(crdsys='eq', ncrd=3)
		srcEQ.shape = (3,1)
		srcs_eq.append( srcEQ )
		
		## HA/dec
		srcRA,srcDec = aipy.coord.eq2radec(srcEQ)
		srcHA = aa.sidereal_time() - srcRA
		srcs_ha.append( srcHA )
		srcs_dc.append( srcDec )
		
		## Source flux over the bandpass - corrected for the bandpass
		jys = src.get_jys()
		jys.shape = (1,nFreq)
		jys = Gij_sf * jys
		srcs_jy.append( jys )
		
		## Frequencies that the fluxes correspond to
		frq = aa.get_afreqs()
		frq.shape = (1,nFreq)
		srcs_fq.append( frq )
		
		## Source shape parameters
		shp = numpy.array(src.srcshape)
		shp.shape = (3,1)
		srcs_sh.append( shp )
		
	# Build the simulation cache
	aa.sim_cache( numpy.concatenate(srcs_eq, axis=1), 
				jys=numpy.concatenate(srcs_jy, axis=0),
				mfreqs=numpy.concatenate(srcs_fq, axis=0),
				srcshapes=numpy.concatenate(srcs_sh, axis=1) )
	aa._cache['s_top'] = numpy.concatenate(srcs_tp, axis=0)
	aa._cache['s_ha'] = numpy.concatenate(srcs_ha, axis=0)
	aa._cache['s_dec'] = numpy.concatenate(srcs_dc, axis=0)
	
	# Build the simulated data.  If no baseline list is provided, build all 
	# baselines available
	if baselines is None:
		baselines = uvUtils.getBaselines(numpy.zeros(len(aa.ants)), Indicies=True)
		
	# Define output data structure
	freq = aa.get_afreqs()*1e9
	if len(freq.shape) == 2:
		freq = freq[0,:]
	UVData = {'freq': freq, 'uvw': {}, 'vis': {}, 'wgt': {}, 'msk': {}, 'bls': {}, 'jd': {}}
	if mask is None:
		UVData['isMasked'] = False
	else:
		UVData['isMasked'] = True
		
	# Go!
	if phaseCenter is not 'z':
		phaseCenter.compute(aa)
		pcAz = phaseCenter.az*180/numpy.pi
		pcEl = phaseCenter.alt*180/numpy.pi
	else:
		pcAz = 0.0
		pcEl = 90.0
		
	for p,pol in enumerate(pols):
		## Apply the antenna gain pattern for each source
		if not flatResponse:
			if p == 0:
				for i in xrange(aa._cache['jys'].shape[0]):
					aa._cache['jys'][i,:] *= Bij_sf(aa._cache['s_top'][i,:], pol)
			else:
				for i in xrange(aa._cache['jys'].shape[0]):
					aa._cache['jys'][i,:] /= Bij_sf(aa._cache['s_top'][i,:], pols[p-1])	# Remove the old pol
					aa._cache['jys'][i,:] *=  Bij_sf(aa._cache['s_top'][i,:], pol)
					
		## Simulate
		if not flatResponse:
			uvw1, vis1 = FastVis(aa, baselines, chanMin, chanMax, pcAz, pcEl, resolve_src=resolve_src)
		else:
			currentVars = locals().keys()
			if 'uvw1' not in currentVars or 'vis1' not in currentVars:
				uvw1, vis1 = FastVis(aa, baselines, chanMin, chanMax, pcAz, pcEl, resolve_src=resolve_src)
				
		## Unpack the data and repack it into a data dictionary
		UVData['bls'][pol] = []
		UVData['uvw'][pol] = []
		UVData['vis'][pol] = []
		UVData['wgt'][pol] = []
		UVData['msk'][pol] = []
		UVData['jd'][pol] = []
		
		if mask is None:
			for i in xrange(uvw1.shape[0]):
				UVData['bls'][pol].append( baselines[i] )
				UVData['uvw'][pol].append( uvw1[i,:,:] )
				UVData['vis'][pol].append( vis1[i,:] )
				UVData['wgt'][pol].append( numpy.ones_like(vis1[i,:])*len(UVData['vis'][pol][-1]) )
				UVData['msk'][pol].append( numpy.zeros_like(UVData['vis'][pol][-1]) )
				UVData['jd'][pol].append( jd )
				
		else:
			for i in xrange(uvw1.shape[0]):
				msk = mask[i]
				
				UVData['bls'][pol].append( baselines[i] )
				UVData['uvw'][pol].append( uvw1[i,:,:].compress(numpy.logical_not(msk), axis=2) )
				UVData['vis'][pol].append( vis1[i,:].compress(numpy.logical_not(msk)) )
				UVData['wgt'][pol].append( numpy.ones_like(UVData['vis'][pol][-1])*len(UVData['vis'][pol][-1]) )
				UVData['msk'][pol].append( msk )
				UVData['jd'][pol].append( jd )
				
	# Cleanup
	if not flatResponse:
		for i in xrange(aa._cache['jys'].shape[0]):
			aa._cache['jys'][i,:] /= Bij_sf(aa._cache['s_top'][i,:], pols[-1])	# Remove the old pol
			aa._cache['jys'][i,:] /= Gij_sf								# Remove the bandpass
			
	# Return
	return UVData


def buildSimData(aa, srcs, pols=['xx', 'yy', 'xy', 'yx'], jd=None, chan=None, phaseCenter='z', baselines=None, mask=None,  flatResponse=False, resolve_src=False, verbose=False):
	"""
	Given an AIPY AntennaArray object and a dictionary of sources from 
	aipy.src.get_catalog, returned a data dictionary of simulated data taken at 
	zenith.  Optinally, the data can be masked using some referenced (observed) 
	data set or only a specific sub-set of baselines.
	
	.. versionchanged:: 1.0.1
		* Added a 'flatResponse' keyword to make it easy to toggle on and off
		  the spectral and spatial response of the array for the simulation
		* Added a 'resolve_src' keyword to turn on source resolution effects
		
	..versionchanged:: 0.4.0
		Added the 'pols' keyword to only compute certain polarization components
	"""
	
	# Update the JD if necessary
	if jd is not None:
		try:
			junk = len(jd)
		except TypeError:
			jd = [jd]
	else:
		jd = [aa.get_jultime()]
		
	# Build up output dictionary
	freq = aa.get_afreqs()*1e9
	if len(freq.shape) == 2:
		freq = freq[0,:]
	UVData = {'freq': freq, 'uvw': {}, 'vis': {}, 'wgt': {}, 'msk': {}, 'bls': {}, 'jd': {}}
	if mask is None:
		UVData['isMasked'] = False
	else:
		UVData['isMasked'] = True
	
	# Loop over polarizations to populate the simulated data set
	for pol in pols:
		UVData['bls'][pol] = []
		UVData['uvw'][pol] = []
		UVData['vis'][pol] = []
		UVData['wgt'][pol] = []
		UVData['msk'][pol] = []
		UVData['jd'][pol] = []

	# Loop over Julian days to fill in the simulated data set
	jdCounter = 1
	for juldate in jd:
		oBlk = __buildSimData(aa, srcs, pols=pols, jd=juldate, chan=chan, phaseCenter=phaseCenter, baselines=baselines, mask=mask, verbose=verbose, count=jdCounter, max=len(jd), flatResponse=flatResponse, resolve_src=resolve_src)
		jdCounter = jdCounter + 1

		for pol in oBlk['bls']:
			for b,u,v,w,m,j in zip(oBlk['bls'][pol], oBlk['uvw'][pol], oBlk['vis'][pol], oBlk['wgt'][pol], oBlk['msk'][pol], oBlk['jd'][pol]):
				UVData['bls'][pol].append( b )
				UVData['uvw'][pol].append( u )
				UVData['vis'][pol].append( v )
				UVData['wgt'][pol].append( w )
				UVData['msk'][pol].append( m )
				UVData['jd'][pol].append( j )
				
	return UVData


def scaleData(dataDict, amps, delays, phaseOffsets=None):
	"""
	Apply a set of antenna-based real gain values and phase delays in ns to a 
	data dictionary.  Returned the new scaled and delayed dictionary.
	
	..versionchanged:: 0.6.3
		Added a keyword so that phase offsets (in radians) can also be specified
	..versionchanged:: 0.4.0
		The delays are now expected to be in nanoseconds rather than radians.
	"""
	
	# Build the data dictionary to hold the scaled and delayed data
	sclUVData = {'freq': (dataDict['freq']).copy(), 'uvw': {}, 'vis': {}, 'wgt': {}, 'msk': {}, 'bls': {}, 'jd': {}}
	if dataDict['isMasked']:
		sclUVData['isMasked'] = True
	else:
		sclUVData['isMasked'] = False
	fq = dataDict['freq'] / 1e9
	
	if phaseOffsets is None:
		phaseOffsets = numpy.zeros_like(delays)
		
	cGains = []
	for i in xrange(len(amps)):
		cGains.append( amps[i]*numpy.exp(2j*numpy.pi*fq*delays[i] + 1j*phaseOffsets[i]) )
		
	# Apply the scales and delays for all polarization pairs found in the original data
	for pol in dataDict['vis'].keys():
		sclUVData['bls'][pol] = []
		sclUVData['uvw'][pol] = []
		sclUVData['vis'][pol] = []
		sclUVData['wgt'][pol] = copy.copy(dataDict['wgt'][pol])
		sclUVData['msk'][pol] = copy.copy(dataDict['msk'][pol])
		sclUVData['jd'][pol] = copy.copy(dataDict['jd'][pol])

		for (i,j),uvw,vis in zip(dataDict['bls'][pol], dataDict['uvw'][pol], dataDict['vis'][pol]):
			sclUVData['bls'][pol].append( (i,j) )
			sclUVData['uvw'][pol].append( uvw )
			sclUVData['vis'][pol].append( vis*cGains[j].conj()*cGains[i] )
			
	return sclUVData
	

def shiftData(dataDict, aa):
	"""
	Shift the uvw coordinates in one data dictionary to a new set of uvw 
	coordinates that correspond to a new AntennaArray object.  This is useful
	for looking at how positional errors in the array affect the data.
	"""
	
	# Build the data dictionary to hold the shifted data
	shftData = {'freq': dataDict['freq'].copy(), 'uvw': {}, 'vis': {}, 'wgt': {}, 'msk': {}, 'bls': {}, 'jd': {}}
	if dataDict['isMasked']:
		shftData['isMasked'] = True
	else:
		shftData['isMasked'] = False
		
	# Apply the coordinate shift all polarization pairs found in the original data
	for pol in dataDict['vis'].keys():
		shftData['bls'][pol] = copy.copy(dataDict['bls'][pol])
		shftData['uvw'][pol] = []
		shftData['vis'][pol] = copy.copy(dataDict['vis'][pol])
		shftData['wgt'][pol] = copy.copy(dataDict['wgt'][pol])
		shftData['msk'][pol] = copy.copy(dataDict['msk'][pol])
		shftData['jd'][pol] = copy.copy(dataDict['jd'][pol])

		for (i,j),m in zip(dataDict['bls'][pol], dataDict['msk'][pol]):
			crds = aa.gen_uvw(j, i, src='z')[:,0,:]
			if dataDict['isMasked']:
				crds = crds.compress(numpy.logical_not(numpy.logical_not(m), axis=2))
			shftData['uvw'][pol].append( crds )
			
	return shftData


def addBaselineNoise(dataDict, SEFD, tInt, bandwidth=None, efficiency=1.0):
	"""
	Given a data dictionary of visibilities, an SEFD or array SEFDs in Jy, 
	and an integration time in seconds, add noise to the visibilities 
	assuming that the "weak source" limit.
	
	This function implements Equation 9-15 from Chapter 9 of "Synthesis 
	Imaging in Radio Astronomy II".
	
	.. versionadded:: 1.0.2
	"""
	
	# Figure out the bandwidth from the frequency list in the 
	if bandwidth is None:
		try:
			bandwidth = dataDict['freq'][1] - dataDict['freq'][0]
		except IndexError:
			raise RuntimeError("Too few frequencies to determine the bandwidth, use the 'bandwidth' keyword")
			
	# Calculate the standard deviation of the real/imaginary noise
	visNoiseSigma = SEFD / efficiency / numpy.sqrt(2.0*bandwidth*tInt)
	
	# Make sure that we have an SEFD for each antenna
	ants = []
	for pol in dataDict['bls'].keys():
		for i,j in dataDict['bls'][pol]:
			if i not in ants:
				ants.append( i )
			if j not in ants:
				ants.append( j )
	nAnts = len(ants)
	try:
		if len(SEFD) != nAnts:
			raise RuntimeError("Mis-match between the number of SEFDs supplied and the number of antennas in the data dictionary")
			
	except TypeError:
		SEFD = numpy.ones(nAnts)*SEFD
		
	# Build the data dictionary to hold the data with noise added
	bnData = {'freq': dataDict['freq'].copy(), 'uvw': {}, 'vis': {}, 'wgt': {}, 'msk': {}, 'bls': {}, 'jd': {}}
	if dataDict['isMasked']:
		bnData['isMasked'] = True
	else:
		bnData['isMasked'] = False
		
	# Copy the original data over and add in the noise
	for pol in dataDict['vis'].keys():
		bnData['bls'][pol] = copy.copy(dataDict['bls'][pol])
		bnData['uvw'][pol] = copy.copy(dataDict['uvw'][pol])
		bnData['vis'][pol] = []
		bnData['wgt'][pol] = copy.copy(dataDict['wgt'][pol])
		bnData['msk'][pol] = copy.copy(dataDict['msk'][pol])
		bnData['jd'][pol] = copy.copy(dataDict['jd'][pol])
		
		## Apply this to the visibilities
		for bl in xrange(len(dataDict['vis'][pol])):
			vShape = dataDict['vis'][pol][bl].shape
			
			### Calculate the expected noise
			visNoise = visNoiseSigma * (numpy.random.randn(*vShape) + 1j*numpy.random.randn(*vShape))
			
			### Apply
			bnData['vis'][pol].append( dataDict['vis'][pol][bl] + visNoise )
			
	return bnData
