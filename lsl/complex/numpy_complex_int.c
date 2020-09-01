#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include "structmember.h"
#include "numpy/npy_3kcompat.h"

#include "complex_int.h"

// Complex 4-bit + 4-bit integer
#include "numpy_complex_int8.c"

// Complex 8-bit + 8-bit integer
#include "numpy_complex_int16.c"

// Complex 16-bit + 16-bit integer
#include "numpy_complex_int32.c"

static NPY_INLINE int PyComplexInt_Check(PyObject *o) {
    return (PyComplexInt8_Check(o) \
            || PyComplexInt16_Check(o) \
            || PyComplexInt32_Check(o));
}

static PyMethodDef ComplexIntMethods[] = {
    {NULL, NULL, 0, NULL}
};

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "numpy_complex_int",
    NULL,
    -1,
    ComplexIntMethods,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

#if defined(NPY_PY3K)
PyMODINIT_FUNC PyInit_numpy_complex_int(void) {
#else
PyMODINIT_FUNC initnumpy_complex_int(void) {
#endif
    
    PyObject *m;
    PyObject* numpy = PyImport_ImportModule("numpy");
    PyObject* numpy_dict = PyModule_GetDict(numpy);
    
#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("numpy_complex_int", ComplexIntMethods);
#endif
    
    if( !m ) {
#if defined(NPY_PY3K)
        return NULL;
#endif
    }
    
    /* Make sure NumPy is initialized */
    import_array();
    import_umath();
    
    /* Register the complexi8 array scalar type */
    int complexi8Num = create_complex_int8(m, numpy_dict);
    if( complexi8Num == -2 ) {
        PyErr_Print();
        PyErr_SetString(PyExc_SystemError, "could not initialize PyComplexI8ArrType_Type");
#if defined(NPY_PY3K)
        return NULL;
#endif
    } else if( complexi8Num == -1 ) {
#if defined(NPY_PY3K)
        return NULL;
#endif
    }
    
    /* Register the complexi16 array scalar type */
    int complexi16Num = create_complex_int16(m, numpy_dict);
    if( complexi16Num == -2 ) {
        PyErr_Print();
        PyErr_SetString(PyExc_SystemError, "could not initialize PyComplexI16ArrType_Type");
#if defined(NPY_PY3K)
        return NULL;
#endif
    } else if( complexi16Num == -1 ) {
#if defined(NPY_PY3K)
        return NULL;
#endif
    }
    
    /* Register the complexi32 array scalar type */
    int complexi32Num = create_complex_int32(m, numpy_dict);
    if( complexi32Num == -2 ) {
        PyErr_Print();
        PyErr_SetString(PyExc_SystemError, "could not initialize PyComplexI32ArrType_Type");
#if defined(NPY_PY3K)
        return NULL;
#endif
    } else if( complexi32Num == -1 ) {
#if defined(NPY_PY3K)
        return NULL;
#endif
    }
    
#if defined(NPY_PY3K)
    return m;
#endif
}
