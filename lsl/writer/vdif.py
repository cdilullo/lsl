# -*- coding: utf-8 -*-

"""Module to write VDIF frames."""

import math
import numpy
import ephem
import struct

from lsl.common import stations as lwa_common
from lsl.common import dp as dp_common
from lsl.reader import tbw
import lsl.astro as astro

__version__ = '0.1'
__revision__ = '$ Revision: 3 $'
__all__ = ['Frame', '__version__', '__revision__', '__all__']

vdifEpoch = ephem.Date('2000/01/01 00:00:00.00')
unixEpoch = ephem.Date('1970/01/01 00:00:00.00')


def log2(numb):
	"""Calculate the log-base 2 of a number or array."""
  
	return numpy.log(numb) / numpy.log(2.0)


class Frame(object):
	"""Object to create and write a VDIF (VLBI Data Interchange Format) 
	frame (version 1, June 26, 2009)."""
	
	def __init__(self, stand=0, time=0, bits=16, data=None, sampleRate=dp_common.fS):
		self.stand = stand
		self.time = time
		self.bits = bits
		self.data = numpy.squeeze(data)
		self.sampleRate = sampleRate
		if data.dtype.kind == 'c':
			self.dataReal = False
		else:
			self.dataReal = True

		# Convert the time from UNIX to epoch and make the data ready to 
		# be written to the disk.
		self.__dataAdjust()
		self.__setEpoch()
		
	def __dataAdjust(self):
		"""Adjust the data to the number of bits required for the frame 
		and convert to unsigned 32-bit integers.  Also, interlace complex
		arrays."""
		
		if self.dataReal:
			interlaced = 1*self.data
		else:
			interlaced = numpy.zeros(2*len(self.data))
			interlaced[0::2] = self.data.real
			interlaced[1::2] = self.data.imag
		
		biased = interlaced - interlaced.min()
		
		scaled = 1.0*biased/biased.max() * (2**self.bits-1)
		scaled = scaled.astype(numpy.uint32)
		
		self.data = scaled
		
	def __setEpoch(self):
		"""Convert a UNIX timestap in seconds since 1970 to seconds since
		January 1st, 2000 using ephem.Data objects.  Also, calculate the 
		frame number since the beginning of the second."""
		
		# UNIX time to seconds since DJD  = 0
		curEpoch = float(unixEpoch)*astro.SECS_IN_DAY + self.time
		self.epoch = 0
		# Seconds since the VDIF epoch
		epochSeconds = curEpoch - float(vdifEpoch)*astro.SECS_IN_DAY
		# Integer seconds
		self.seconds = long(epochSeconds)
		
		# Compute the frames since the beginning of the second
		frame = (epochSeconds - self.seconds) * (self.sampleRate/len(self.data))
		if self.dataReal:
			self.frame = long(frame)
		else:
			self.frame = long(frame * 2)

	def createRawFrame(self):
		"""Using the data and information stored in the object, create a
		numpy array on unsigned 1-byte integers that represents the frame."""
		
		# Find out how many samples can be packed into a work and build
		# the corresponding numpy array.
		samplesPerWord = int(32 / self.bits)
		raw = numpy.zeros(32 + samplesPerWord*len(self.data), dtype=numpy.uint8)

		# Valid data, standard (not legacy) 32-bit header, and seconds since 
		# the 01/01/2000 epoch.
		raw[3] = (0 << 7) | (0 << 6) | ((self.seconds >> 24) & 63)
		raw[2] = (self.seconds >> 16) & 255
		raw[1] = (self.seconds >> 8) & 255
		raw[0] = self.seconds & 255

		# Refernce epoch (0 == 01/01/2000) and frame count
		raw[7] = self.epoch & 63
		raw[6] = (self.frame >> 16) & 255
		raw[5] = (self.frame >> 8) & 255
		raw[4] = self.frame & 255

		# VDIF version number, number of channels (just 1), and data frame 
		# length in units to 8-bytes (8 raw array elements)
		raw[11] = (1 << 6) | (0 & 31)
		raw[10] = ((len(raw) / 8) >> 16) & 255
		raw[9] = ((len(raw) / 8) >> 8) & 255
		raw[8] = (len(raw) / 8) & 255

		# Data type, bits per sample, thread ID, and station ID
		# NB:  The thread ID is fixed at `0'
		if self.dataReal:
			raw[15] = (0 << 7) | ((self.bits & 31) << 2) | ((0 >> 8) & 3)
		else:
			raw[15] = (1 << 7) | ((self.bits & 31) << 2) | ((0 >> 8) & 3)
		raw[14] = 0 & 255
		raw[13] = (self.stand >> 8) & 255
		raw[12] = self.stand & 255
		
		# User-defined words 4 through 7 (array entries 16 to 31)
		raw[16:19] = 0
		raw[20:23] = 0
		raw[24:27] = 0
		raw[28:31] = 0
		
		# Data values
		for f in range(32,len(raw),4):
			i = (f - 32) / 4
			word = 0
			for p in range(samplesPerWord):
				word |= (self.data[i*samplesPerWord + p] << self.bits*p)
			raw[f+3] = (word >> 24) & 255
			raw[f+2] = (word >> 16) & 255
			raw[f+1] = (word >> 8) & 255
			raw[f+0] = word & 255
				
		return raw

	def writeRawFrame(self, fh):
		"""Create a numpy representation of the VDIF frame and then write 
		it to the specified file handle."""
		
		rawFrame = self.createRawFrame()
		rawFrame.tofile(fh)


if __name__ == "__main__":
	fh = open('/home/jayce/TBW Data/multiTBW_Sept_19_8pm.dat', 'rb')
	
	# Make sure that the data is TBW and determine the data length
	test = tbw.readFrame(fh)
	print("TBW Data:  %s" % test.header.isTBW())
	if not test.header.isTBW():
		raise errors.notTBWError()
	print("Data Length: %i bits" % test.getDataBits())
	fh.seek(0)

	# Create the output files
	ofh = []
	for i in range(20):
		ofh.append(open('tbw-frame-%i.vdif' % (i+1), 'wb'))

	# Read in the data and add it to the VDIF files created above
	for i in range(300000):
		# Read in the next frame and anticipate any problems that could occur
		try:
			cFrame = tbw.readFrame(fh, Verbose=False)
		except errors.eofError:
			break
		except errors.syncError:
			#print "WARNING: Mark 5C sync error on frame #%i" % (int(fh.tell())/TBWFrameSize-1)
			syncCount = syncCount + 1
			continue
		except errors.numpyError:
			break

		stand = cFrame.header.parseID()
		if cFrame.header.frameCount % 10000 == 0:
			print("%2i  %14i  %6.3f  %5i  %5i" % (stand, cFrame.data.timeTag, cFrame.getTime(), cFrame.header.frameCount, cFrame.header.secondsCount))
		
		frame1 = Frame(stand=2*(stand-1)+1, time=cFrame.getTime(), bits=cFrame.getDataBits(), 
						sampleRate = dp_common.fS, data=cFrame.data.xy[0,:])
		frame2 = Frame(stand=2*(stand-1)+2, time=cFrame.getTime(), bits=cFrame.getDataBits(), 
						sampleRate = dp_common.fS, data=cFrame.data.xy[1,:])
		
		frame1.writeRawFrame(ofh[2*(stand-1)+0])
		frame2.writeRawFrame(ofh[2*(stand-1)+1])
	fh.close()
	
	# Close all of the constituent VDIF files.
	for fh in ofh:
		fh.close()
