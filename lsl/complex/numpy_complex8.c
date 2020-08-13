#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include "structmember.h"
#include "numpy/npy_3kcompat.h"

#include "complex8.h"

typedef struct {
        PyObject_HEAD
        complex8 obval;
} PyComplex8ScalarObject;

PyMemberDef PyComplex8ArrType_members[] = {
    {"real_imag", T_BYTE, offsetof(PyComplex8ScalarObject, obval), READONLY,
        "The real and imaginary parts of the complex8 integer"},
    {NULL}
};

static PyObject* PyComplex8ArrType_get_real(PyObject *self, void *closure) {
    complex8 *c = &((PyComplex8ScalarObject *)self)->obval;
    const signed char* sc = fourBitLUT[c->real_imag];
    PyObject *value = PyInt_FromLong(sc[0]);
    return value;
}

static PyObject* PyComplex8ArrType_get_imag(PyObject *self, void *closure) {
    complex8 *c = &((PyComplex8ScalarObject *)self)->obval;
    const signed char* sc = fourBitLUT[c->real_imag];
    PyObject *value = PyInt_FromLong(sc[1]);
    return value;
}

PyGetSetDef PyComplex8ArrType_getset[] = {
    {"real", PyComplex8ArrType_get_real, NULL,
        "The real part of the complex8 integer", NULL},
    {"imag", PyComplex8ArrType_get_imag, NULL,
        "The imaginary part of the complex8 integer", NULL},
    {NULL}
};

PyTypeObject PyComplex8ArrType_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "complexint.complex8",                      /* tp_name*/
    sizeof(PyComplex8ScalarObject),             /* tp_basicsize*/
    0,                                          /* tp_itemsize */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    0,                                          /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    PyComplex8ArrType_members,                  /* tp_members */
    PyComplex8ArrType_getset,                   /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
#if PY_VERSION_HEX >= 0x02060000
    0,                                          /* tp_version_tag */
#endif
};

static PyArray_ArrFuncs _PyComplex8_ArrFuncs;
PyArray_Descr *complex8_descr;

static PyObject* COMPLEX8_getitem(char *ip, PyArrayObject *ap) {
    complex8 c;
    PyObject *tuple;
    
    if( (ap == NULL) || PyArray_ISBEHAVED_RO(ap) ) {
        c = *((complex8 *) ip);
    } else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_INT8);
        descr->f->copyswap(&c.real_imag, ip, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }
    
    tuple = PyTuple_New(2);
    const signed char* sc = fourBitLUT[c.real_imag];
    PyTuple_SET_ITEM(tuple, 0, PyInt_FromLong(sc[0]));
    PyTuple_SET_ITEM(tuple, 1, PyInt_FromLong(sc[1]));
    
    return tuple;
}

static int COMPLEX8_setitem(PyObject *op, char *ov, PyArrayObject *ap) {
    complex8 c;
    
    if( PyArray_IsScalar(op, Complex8) ) {
        c = ((PyComplex8ScalarObject *) op)->obval;
    } else {
        signed char real_imag;
        real_imag  =  (unsigned char) (PyInt_AsLong(PyTuple_GetItem(op, 0)) * 16);
        real_imag |= ((unsigned char) (PyInt_AsLong(PyTuple_GetItem(op, 1)) * 16)) >> 4;
    }
    
    if( PyErr_Occurred() ) {
        if( PySequence_Check(op) ) {
            PyErr_Clear();
            PyErr_SetString(PyExc_ValueError, "setting an array element with a sequence.");
        }
        return -1;
    }
    
    if( ap == NULL || PyArray_ISBEHAVED(ap) ) {
        *((complex8 *) ov) = c;
    } else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_INT8);
        descr->f->copyswap(ov, &c.real_imag, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }
    
    return 0;
}

static void COMPLEX8_copyswap(complex8 *dst, complex8 *src, int swap, void *NPY_UNUSED(arr)) {
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_INT8);
    descr->f->copyswapn(dst, sizeof(unsigned char), src, sizeof(unsigned char), 1, swap, NULL);
    Py_DECREF(descr);
}

static void COMPLEX8_copyswapn(complex8 *dst, npy_intp dstride,
                               complex8 *src, npy_intp sstride,
                               npy_intp n, int swap, void *NPY_UNUSED(arr)) {
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_INT8);
    descr->f->copyswapn(&dst->real_imag, dstride, &src->real_imag, sstride, n, swap, NULL);
    Py_DECREF(descr);    
}

static int COMPLEX8_compare(complex8 *pa, complex8 *pb, PyArrayObject *NPY_UNUSED(ap)) {
    complex8 a = *pa, b = *pb;
    npy_bool anan, bnan;
    int ret;
    
    anan = complex8_isnan(a);
    bnan = complex8_isnan(b);
    
    if( anan ) {
        ret = bnan ? 0 : -1;
    } else if( bnan ) {
        ret = 1;
    } else if( complex8_less(a, b) ) {
        ret = -1;
    } else if( complex8_less(b, a) ) {
        ret = 1;
    } else {
        ret = 0;
    }
    
    return ret;
}

static int COMPLEX8_argmax(complex8 *ip, npy_intp n, npy_intp *max_ind, PyArrayObject *NPY_UNUSED(aip)) {
    npy_intp i;
    complex8 mp = *ip;
    
    *max_ind = 0;
    
    if( complex8_isnan(mp) ) {
        /* nan encountered; it's maximal */
        return 0;
    }

    for(i = 1; i < n; i++) {
        ip++;
        /*
         * Propagate nans, similarly as max() and min()
         */
        if( !(complex8_less_equal(*ip, mp)) ) {  /* negated, for correct nan handling */
            mp = *ip;
            *max_ind = i;
            if (complex8_isnan(mp)) {
                /* nan encountered, it's maximal */
                break;
            }
        }
    }
    return 0;
}

static npy_bool COMPLEX8_nonzero(char *ip, PyArrayObject *ap) {
    complex8 c;
    if (ap == NULL || PyArray_ISBEHAVED_RO(ap)) {
        c = *(complex8 *) ip;
    }
    else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_INT8);
        descr->f->copyswap(&c.real_imag, ip, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }
    return (npy_bool) !complex8_equal(c, (complex8) {0});
}

static void COMPLEX8_fillwithscalar(complex8 *buffer, npy_intp length, complex8 *value, void *NPY_UNUSED(ignored)) {
    npy_intp i;
    complex8 val = *value;

    for (i = 0; i < length; ++i) {
        buffer[i] = val;
    }
}

#define MAKE_T_TO_COMPLEX8(TYPE, type)                                         \
static void                                                                    \
TYPE ## _to_complex8(type *ip, complex8 *op, npy_intp n,                       \
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
{                                                                              \
    while (n--) {                                                              \
        op->real_imag = (unsigned char) ((*ip++) * 16);                        \
        *op++;                                                                 \
    }                                                                          \
}

MAKE_T_TO_COMPLEX8(BOOL, npy_bool);
MAKE_T_TO_COMPLEX8(BYTE, npy_byte);

#define MAKE_COMPLEX8_TO_CT(TYPE, type)                                        \
static void                                                                    \
complex8_to_## TYPE(complex8* ip, type *op, npy_intp n,                        \
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
{                                                                              \
    const signed char* sc;                                                     \
    while (n--) {                                                              \
        sc = fourBitLUT[ip->real_imag];                                        \
        *(op++) = (type) sc[0];                                                \
        *(op++) = (type) sc[1];                                                \
        (*ip++);                                                               \
    }                                                                          \
}

MAKE_COMPLEX8_TO_CT(CFLOAT, npy_float);
MAKE_COMPLEX8_TO_CT(CDOUBLE, npy_double);
MAKE_COMPLEX8_TO_CT(CLONGDOUBLE, npy_longdouble);

static void register_cast_function(int sourceType, int destType, PyArray_VectorUnaryFunc *castfunc) {
    PyArray_Descr *descr = PyArray_DescrFromType(sourceType);
    PyArray_RegisterCastFunc(descr, destType, castfunc);
    PyArray_RegisterCanCast(descr, destType, NPY_NOSCALAR);
    Py_DECREF(descr);
}

static PyObject* complex8_arrtype_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    complex8 c;

    if( !PyArg_ParseTuple(args, "i", &c.real_imag) ) {
        return NULL;
    }
    
    return PyArray_Scalar(&c, complex8_descr, NULL);
}

static PyObject* gentype_richcompare(PyObject *self, PyObject *other, int cmp_op) {
    PyObject *arr, *ret;

    arr = PyArray_FromScalar(self, NULL);
    if (arr == NULL) {
        return NULL;
    }
    ret = Py_TYPE(arr)->tp_richcompare(arr, other, cmp_op);
    Py_DECREF(arr);
    return ret;
}

static long complex8_arrtype_hash(PyObject *o) {
    complex8 c = ((PyComplex8ScalarObject *)o)->obval;
    long value = 0x456789;
    value = (10000004 * value) ^ _Py_HashBytes(&(c.real_imag), sizeof(unsigned char));
    if( value == -1 ) {
        value = -2;
    }
    return value;
}

static PyObject* complex8_arrtype_repr(PyObject *o) {
    char str[64];
    complex8 c = ((PyComplex8ScalarObject *)o)->obval;
    sprintf(str, "complex8(%u)", c.real_imag);
    return PyUString_FromString(str);
}

static PyObject* complex8_arrtype_str(PyObject *o) {
    char str[64];
    complex8 c = ((PyComplex8ScalarObject *)o)->obval;
    const signed char* sc = fourBitLUT[c.real_imag];
    sprintf(str, "%i%+ij", sc[0], sc[1]);
    return PyUString_FromString(str);
}

static PyMethodDef Complex8Methods[] = {
    {NULL, NULL, 0, NULL}
};

#define UNARY_UFUNC(name, ret_type)\
static void \
complex8_##name##_ufunc(char** args, npy_intp* dimensions,\
    npy_intp* steps, void* data) {\
    char *ip1 = args[0], *op1 = args[1];\
    npy_intp is1 = steps[0], os1 = steps[1];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, op1 += os1){\
        const complex8 in1 = *(complex8 *)ip1;\
        *((ret_type *)op1) = complex8_##name(in1);};}

UNARY_UFUNC(isnan, npy_bool)
UNARY_UFUNC(isinf, npy_bool)
UNARY_UFUNC(isfinite, npy_bool)
UNARY_UFUNC(absolute, npy_double)
UNARY_UFUNC(negative, complex8)
UNARY_UFUNC(conjugate, complex8)

#define BINARY_GEN_UFUNC(name, func_name, arg_type, ret_type)\
static void \
complex8_##func_name##_ufunc(char** args, npy_intp* dimensions,\
    npy_intp* steps, void* data) {\
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];\
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1){\
        const complex8 in1 = *(complex8 *)ip1;\
        const arg_type in2 = *(arg_type *)ip2;\
        *((ret_type *)op1) = complex8_##func_name(in1, in2);};};

#define BINARY_UFUNC(name, ret_type)\
    BINARY_GEN_UFUNC(name, name, complex8, ret_type)
#define BINARY_SCALAR_UFUNC(name, ret_type)\
    BINARY_GEN_UFUNC(name, name##_scalar, npy_int8, ret_type)

BINARY_UFUNC(copysign, complex8)
BINARY_UFUNC(equal, npy_bool)
BINARY_UFUNC(not_equal, npy_bool)
BINARY_UFUNC(less, npy_bool)
BINARY_UFUNC(less_equal, npy_bool)

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "numpy_complex8",
    NULL,
    -1,
    Complex8Methods,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

#if defined(NPY_PY3K)
PyMODINIT_FUNC PyInit_numpy_complex8(void) {
#else
PyMODINIT_FUNC initnumpy_complex8(void) {
#endif
    
    PyObject *m;
    int complex8Num;
    PyObject* numpy = PyImport_ImportModule("numpy");
    PyObject* numpy_dict = PyModule_GetDict(numpy);
    int arg_types[3];
    
#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("numpy_complex8", Complex8Methods);
#endif
    
    if( !m ) {
#if defined(NPY_PY3K)
        return NULL;
#endif
    }
    
    /* Fill the lookup table */
    complex8_fillLUT();
    
    /* Make sure NumPy is initialized */
    import_array();
    import_umath();
    
    /* Register the complex8 array scalar type */
#if defined(NPY_PY3K)
    PyComplex8ArrType_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
#else
    PyComplex8ArrType_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES;
#endif
    PyComplex8ArrType_Type.tp_new = complex8_arrtype_new;
    PyComplex8ArrType_Type.tp_richcompare = gentype_richcompare;
    PyComplex8ArrType_Type.tp_hash = complex8_arrtype_hash;
    PyComplex8ArrType_Type.tp_repr = complex8_arrtype_repr;
    PyComplex8ArrType_Type.tp_str = complex8_arrtype_str;
    PyComplex8ArrType_Type.tp_base = &PyGenericArrType_Type;
    if( PyType_Ready(&PyComplex8ArrType_Type) < 0 ) {
        PyErr_Print();
        PyErr_SetString(PyExc_SystemError, "could not initialize PyComplex8ArrType_Type");
#if defined(NPY_PY3K)
        return NULL;
#endif
    }
    
    /* The array functions */
    PyArray_InitArrFuncs(&_PyComplex8_ArrFuncs);
    _PyComplex8_ArrFuncs.getitem = (PyArray_GetItemFunc*)COMPLEX8_getitem;
    _PyComplex8_ArrFuncs.setitem = (PyArray_SetItemFunc*)COMPLEX8_setitem;
    _PyComplex8_ArrFuncs.copyswap = (PyArray_CopySwapFunc*)COMPLEX8_copyswap;
    _PyComplex8_ArrFuncs.copyswapn = (PyArray_CopySwapNFunc*)COMPLEX8_copyswapn;
    _PyComplex8_ArrFuncs.compare = (PyArray_CompareFunc*)COMPLEX8_compare;
    _PyComplex8_ArrFuncs.argmax = (PyArray_ArgFunc*)COMPLEX8_argmax;
    _PyComplex8_ArrFuncs.nonzero = (PyArray_NonzeroFunc*)COMPLEX8_nonzero;
    _PyComplex8_ArrFuncs.fillwithscalar = (PyArray_FillWithScalarFunc*)COMPLEX8_fillwithscalar;
    
    /* The complex8 array descr */
    complex8_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
    complex8_descr->typeobj = &PyComplex8ArrType_Type;
    complex8_descr->kind = 'i';
    complex8_descr->type = 'b';
    complex8_descr->byteorder = '=';
    complex8_descr->type_num = 0; /* assigned at registration */
    complex8_descr->elsize = sizeof(unsigned char)*1;
    complex8_descr->alignment = 1;
    complex8_descr->subarray = NULL;
    complex8_descr->fields = NULL;
    complex8_descr->names = NULL;
    complex8_descr->f = &_PyComplex8_ArrFuncs;
    
    Py_INCREF(&PyComplex8ArrType_Type);
    complex8Num = PyArray_RegisterDataType(complex8_descr);
    
    if( complex8Num < 0 ) {
#if defined(NPY_PY3K)
        return NULL;
#endif
    }
    
    register_cast_function(NPY_BOOL, complex8Num, (PyArray_VectorUnaryFunc*)BOOL_to_complex8);
    register_cast_function(NPY_BYTE, complex8Num, (PyArray_VectorUnaryFunc*)BYTE_to_complex8);
    
    register_cast_function(complex8Num, NPY_CFLOAT, (PyArray_VectorUnaryFunc*)complex8_to_CFLOAT);
    register_cast_function(complex8Num, NPY_CDOUBLE, (PyArray_VectorUnaryFunc*)complex8_to_CDOUBLE);
    register_cast_function(complex8Num, NPY_CLONGDOUBLE, (PyArray_VectorUnaryFunc*)complex8_to_CLONGDOUBLE);
    
#define REGISTER_UFUNC(name)\
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name),\
            complex8_descr->type_num, complex8_##name##_ufunc, arg_types, NULL)
    
#define REGISTER_SCALAR_UFUNC(name)\
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name),\
            complex8_descr->type_num, complex8_##name##_scalar_ufunc, arg_types, NULL)
    
    /* complex8 -> bool */
    arg_types[0] = complex8_descr->type_num;
    arg_types[1] = NPY_BOOL;
    
    REGISTER_UFUNC(isnan);
    REGISTER_UFUNC(isinf);
    REGISTER_UFUNC(isfinite);
    
    /* complex8 -> double */
    arg_types[1] = NPY_DOUBLE;
    
    REGISTER_UFUNC(absolute);
    
    /* quat -> quat */
    arg_types[1] = complex8_descr->type_num;
    
    REGISTER_UFUNC(negative);
    REGISTER_UFUNC(conjugate);

    /* complex8, complex8 -> bool */

    arg_types[2] = NPY_BOOL;

    REGISTER_UFUNC(equal);
    REGISTER_UFUNC(not_equal);
    REGISTER_UFUNC(less);
    REGISTER_UFUNC(less_equal);
    
    /* complex8, complex8 -> complex8 */

    arg_types[1] = complex8_descr->type_num;

    REGISTER_UFUNC(copysign);

    PyModule_AddObject(m, "complex8", (PyObject *)&PyComplex8ArrType_Type);

#if defined(NPY_PY3K)
    return m;
#endif
}
