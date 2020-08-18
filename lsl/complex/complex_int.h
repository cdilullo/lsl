#ifndef COMPLEX_COMPLEX_INT_H_INCLUDE_GUARD_
#define COMPLEX_COMPLEX_INT_H_INCLUDE_GUARD_

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include "complex_int8.h"

#define NPY_COMPLEX_INT8 256
#define NPY_COMPLEX_INT16 257
#define NPY_COMPLEX_INT32 258

inline static int import_complex_int(void) {
    PyObject *module = PyImport_ImportModule("lsl.complex");
    if( module == NULL ) {
        PyErr_Warn(PyExc_RuntimeWarning, "Cannot load the LSL complex integer types");
	return -1;
    }
    Py_XDECREF(module);

    return 0;
}

void lsl_unpack_ci8(complexi8 packed, signed char* real, signed char* imag);

#ifdef __cplusplus
}
#endif

#endif  //COMPLEX_COMPLEX_INT_H_INCLUDE_GUARD_


