import numpy

from lsl.complex.numpy_complex8 import complex8
from lsl.complex.numpy_complex16 import complex16
from lsl.complex.numpy_complex32 import complex32

__all__ = ['complex8', 'complex16', 'complex32']

if numpy.__dict__.get('complex8') is None:
    numpy.complex8 = complex8
    numpy.typeDict['complex8'] = numpy.dtype(complex8)
    
if numpy.__dict__.get('complex16') is None:
    numpy.complex16 = complex16
    numpy.typeDict['complex16'] = numpy.dtype(complex16)

if numpy.__dict__.get('complex32') is None:
    numpy.complex16 = complex32
    numpy.typeDict['complex32'] = numpy.dtype(complex32)
