#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include "structmember.h"
#include "numpy/npy_3kcompat.h"

#include "complex32.h"

typedef struct {
        PyObject_HEAD
        complex32 obval;
} PyComplex32ScalarObject;

PyMemberDef PyComplex32ArrType_members[] = {
    {"real", T_BYTE, offsetof(PyComplex32ScalarObject, obval), READONLY,
        "The real part of the complex32 integer"},
    {"imag", T_BYTE, offsetof(PyComplex32ScalarObject, obval)+1, READONLY,
        "The imaginary part of the complex32 integer"},
    {NULL}
};

static PyObject* PyComplex32ArrType_get_real(PyObject *self, void *closure) {
    complex32 *c = &((PyComplex32ScalarObject *)self)->obval;
    PyObject *value = PyInt_FromLong(c->real);
    return value;
}

static PyObject* PyComplex32ArrType_get_imag(PyObject *self, void *closure) {
    complex32 *c = &((PyComplex32ScalarObject *)self)->obval;
    PyObject *value = PyInt_FromLong(c->imag);
    return value;
}

PyGetSetDef PyComplex32ArrType_getset[] = {
    {"real", PyComplex32ArrType_get_real, NULL,
        "The real part of the complex32 integer", NULL},
    {"imag", PyComplex32ArrType_get_imag, NULL,
        "The imaginary part of the complex32 integer", NULL},
    {NULL}
};

PyTypeObject PyComplex32ArrType_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "complexint.complex32",                     /* tp_name*/
    sizeof(PyComplex32ScalarObject),            /* tp_basicsize*/
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
    PyComplex32ArrType_members,                 /* tp_members */
    PyComplex32ArrType_getset,                  /* tp_getset */
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

static PyArray_ArrFuncs _PyComplex32_ArrFuncs;
PyArray_Descr *complex32_descr;

static PyObject* COMPLEX32_getitem(char *ip, PyArrayObject *ap) {
    complex32 c;
    PyObject *tuple;
    
    if( (ap == NULL) || PyArray_ISBEHAVED_RO(ap) ) {
        c = *((complex32 *) ip);
    } else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_DOUBLE);
        descr->f->copyswap(&c.real, ip,   !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(&c.imag, ip+1, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }

    tuple = PyObject_New(PyComplex32ScalarObject, &PyComplex32ArrType_Type);
    ((PyComplex32ScalarObject *)tuple)->obval = c;
    //PyTuple_SET_ITEM(tuple, 0, PyInt_FromLong(c.real));
    //PyTuple_SET_ITEM(tuple, 1, PyInt_FromLong(c.imag));
    
    return tuple;
}

static int COMPLEX32_setitem(PyObject *op, char *ov, PyArrayObject *ap) {
    complex32 c;
    
    if( PyArray_IsScalar(op, Complex32) ) {
        c = ((PyComplex32ScalarObject *) op)->obval;
    } else {
        c.real = (signed char) PyInt_AsLong(PyTuple_GetItem(op, 0));
        c.imag = (signed char) PyInt_AsLong(PyTuple_GetItem(op, 1));
    }
    
    if( PyErr_Occurred() ) {
        if( PySequence_Check(op) ) {
            PyErr_Clear();
            PyErr_SetString(PyExc_ValueError, "setting an array element with a sequence.");
        }
        return -1;
    }
    
    if( ap == NULL || PyArray_ISBEHAVED(ap) ) {
        *((complex32 *) ov) = c;
    } else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_INT8);
        descr->f->copyswap(ov,   &c.real, !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(ov+1, &c.imag, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }
    
    return 0;
}

static void COMPLEX32_copyswap(complex32 *dst, complex32 *src, int swap, void *NPY_UNUSED(arr)) {
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_INT8);
    descr->f->copyswapn(dst, sizeof(signed char), src, sizeof(signed char), 2, swap, NULL);
    Py_DECREF(descr);
}

static void COMPLEX32_copyswapn(complex32 *dst, npy_intp dstride,
                               complex32 *src, npy_intp sstride,
                               npy_intp n, int swap, void *NPY_UNUSED(arr)) {
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_INT8);
    descr->f->copyswapn(&dst->real, dstride, &src->real, sstride, n, swap, NULL);
    descr->f->copyswapn(&dst->imag, dstride, &src->imag, sstride, n, swap, NULL);
    Py_DECREF(descr);    
}

static int COMPLEX32_compare(complex32 *pa, complex32 *pb, PyArrayObject *NPY_UNUSED(ap)) {
    complex32 a = *pa, b = *pb;
    npy_bool anan, bnan;
    int ret;
    
    anan = complex32_isnan(a);
    bnan = complex32_isnan(b);
    
    if( anan ) {
        ret = bnan ? 0 : -1;
    } else if( bnan ) {
        ret = 1;
    } else if( complex32_less(a, b) ) {
        ret = -1;
    } else if( complex32_less(b, a) ) {
        ret = 1;
    } else {
        ret = 0;
    }
    
    return ret;
}

static int COMPLEX32_argmax(complex32 *ip, npy_intp n, npy_intp *max_ind, PyArrayObject *NPY_UNUSED(aip)) {
    npy_intp i;
    complex32 mp = *ip;
    
    *max_ind = 0;
    
    if( complex32_isnan(mp) ) {
        /* nan encountered; it's maximal */
        return 0;
    }

    for(i = 1; i < n; i++) {
        ip++;
        /*
         * Propagate nans, similarly as max() and min()
         */
        if( !(complex32_less_equal(*ip, mp)) ) {  /* negated, for correct nan handling */
            mp = *ip;
            *max_ind = i;
            if (complex32_isnan(mp)) {
                /* nan encountered, it's maximal */
                break;
            }
        }
    }
    return 0;
}

static npy_bool COMPLEX32_nonzero(char *ip, PyArrayObject *ap) {
    complex32 c;
    if (ap == NULL || PyArray_ISBEHAVED_RO(ap)) {
        c = *(complex32 *) ip;
    }
    else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_INT8);
        descr->f->copyswap(&c.real, ip,   !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(&c.imag, ip+1, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }
    return (npy_bool) !complex32_equal(c, (complex32) {0,0});
}

static void COMPLEX32_fillwithscalar(complex32 *buffer, npy_intp length, complex32 *value, void *NPY_UNUSED(ignored)) {
    npy_intp i;
    complex32 val = *value;

    for (i = 0; i < length; ++i) {
        buffer[i] = val;
    }
}

#define MAKE_T_TO_COMPLEX32(TYPE, type)                                        \
static void                                                                    \
TYPE ## _to_complex32(type *ip, complex32 *op, npy_intp n,                     \
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
{                                                                              \
    while (n--) {                                                              \
        op->real = (signed char) (*ip++);                                      \
        op->imag = 0;                                                          \
        *op++;                                                                 \
    }                                                                          \
}

MAKE_T_TO_COMPLEX32(BOOL, npy_bool);
MAKE_T_TO_COMPLEX32(BYTE, npy_byte);

#define MAKE_COMPLEX32_TO_CT(TYPE, type)                                       \
static void                                                                    \
complex32_to_## TYPE(complex32* ip, type *op, npy_intp n,                      \
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
{                                                                              \
    while (n--) {                                                              \
        *(op++) = (type) ip->real;                                             \
        *(op++) = (type) ip->imag;                                             \
        (*ip++);                                                               \
    }                                                                          \
}

MAKE_COMPLEX32_TO_CT(CFLOAT, npy_float);
MAKE_COMPLEX32_TO_CT(CDOUBLE, npy_double);
MAKE_COMPLEX32_TO_CT(CLONGDOUBLE, npy_longdouble);

static void register_cast_function(int sourceType, int destType, PyArray_VectorUnaryFunc *castfunc) {
    PyArray_Descr *descr = PyArray_DescrFromType(sourceType);
    PyArray_RegisterCastFunc(descr, destType, castfunc);
    PyArray_RegisterCanCast(descr, destType, NPY_NOSCALAR);
    Py_DECREF(descr);
}

static PyObject* complex32_arrtype_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    complex32 c;

    if( !PyArg_ParseTuple(args, "ii", &c.real, &c.imag) ) {
        return NULL;
    }
    
    return PyArray_Scalar(&c, complex32_descr, NULL);
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

static long complex32_arrtype_hash(PyObject *o) {
    complex32 c = ((PyComplex32ScalarObject *)o)->obval;
    long value = 0x456789;
    value = (10000004 * value) ^ _Py_HashBytes(&(c.real), sizeof(signed char));
    value = (10000004 * value) ^ _Py_HashBytes(&(c.imag), sizeof(signed char));
    if( value == -1 ) {
        value = -2;
    }
    return value;
}

static PyObject* complex32_arrtype_repr(PyObject *o) {
    char str[64];
    complex32 c = ((PyComplex32ScalarObject *)o)->obval;
    sprintf(str, "complex32(%i, %i)", c.real, c.imag);
    return PyUString_FromString(str);
}

static PyObject* complex32_arrtype_str(PyObject *o) {
    char str[64];
    complex32 c = ((PyComplex32ScalarObject *)o)->obval;
    sprintf(str, "%i%+ij", c.real, c.imag);
    return PyUString_FromString(str);
}

static PyMethodDef Complex32Methods[] = {
    {NULL, NULL, 0, NULL}
};

#define UNARY_UFUNC(name, ret_type)\
static void \
complex32_##name##_ufunc(char** args, npy_intp* dimensions,\
    npy_intp* steps, void* data) {\
    char *ip1 = args[0], *op1 = args[1];\
    npy_intp is1 = steps[0], os1 = steps[1];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, op1 += os1){\
        const complex32 in1 = *(complex32 *)ip1;\
        *((ret_type *)op1) = complex32_##name(in1);};}

UNARY_UFUNC(isnan, npy_bool)
UNARY_UFUNC(isinf, npy_bool)
UNARY_UFUNC(isfinite, npy_bool)
UNARY_UFUNC(absolute, npy_double)
UNARY_UFUNC(negative, complex32)
UNARY_UFUNC(conjugate, complex32)

#define BINARY_GEN_UFUNC(name, func_name, arg_type, ret_type)\
static void \
complex32_##func_name##_ufunc(char** args, npy_intp* dimensions,\
    npy_intp* steps, void* data) {\
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];\
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1){\
        const complex32 in1 = *(complex32 *)ip1;\
        const arg_type in2 = *(arg_type *)ip2;\
        *((ret_type *)op1) = complex32_##func_name(in1, in2);};};

#define BINARY_UFUNC(name, ret_type)\
    BINARY_GEN_UFUNC(name, name, complex32, ret_type)
#define BINARY_SCALAR_UFUNC(name, ret_type)\
    BINARY_GEN_UFUNC(name, name##_scalar, npy_int8, ret_type)

BINARY_UFUNC(copysign, complex32)
BINARY_UFUNC(equal, npy_bool)
BINARY_UFUNC(not_equal, npy_bool)
BINARY_UFUNC(less, npy_bool)
BINARY_UFUNC(less_equal, npy_bool)

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "numpy_complex32",
    NULL,
    -1,
    Complex32Methods,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

#if defined(NPY_PY3K)
PyMODINIT_FUNC PyInit_numpy_complex32(void) {
#else
PyMODINIT_FUNC initnumpy_complex32(void) {
#endif
    
    PyObject *m;
    int complex32Num;
    PyObject* numpy = PyImport_ImportModule("numpy");
    PyObject* numpy_dict = PyModule_GetDict(numpy);
    int arg_types[3];
    
#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("numpy_complex32", Complex32Methods);
#endif
    
    if( !m ) {
#if defined(NPY_PY3K)
        return NULL;
#endif
    }
    
    /* Make sure NumPy is initialized */
    import_array();
    import_umath();
    
    /* Register the complex32 array scalar type */
#if defined(NPY_PY3K)
    PyComplex32ArrType_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
#else
    PyComplex32ArrType_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES;
#endif
    PyComplex32ArrType_Type.tp_new = complex32_arrtype_new;
    PyComplex32ArrType_Type.tp_richcompare = gentype_richcompare;
    PyComplex32ArrType_Type.tp_hash = complex32_arrtype_hash;
    PyComplex32ArrType_Type.tp_repr = complex32_arrtype_repr;
    PyComplex32ArrType_Type.tp_str = complex32_arrtype_str;
    PyComplex32ArrType_Type.tp_base = &PyGenericArrType_Type;
    if( PyType_Ready(&PyComplex32ArrType_Type) < 0 ) {
        PyErr_Print();
        PyErr_SetString(PyExc_SystemError, "could not initialize PyComplex32ArrType_Type");
        #if defined(NPY_PY3K)
        return NULL;
        #endif
    }
    
    /* The array functions */
    PyArray_InitArrFuncs(&_PyComplex32_ArrFuncs);
    _PyComplex32_ArrFuncs.getitem = (PyArray_GetItemFunc*)COMPLEX32_getitem;
    _PyComplex32_ArrFuncs.setitem = (PyArray_SetItemFunc*)COMPLEX32_setitem;
    _PyComplex32_ArrFuncs.copyswap = (PyArray_CopySwapFunc*)COMPLEX32_copyswap;
    _PyComplex32_ArrFuncs.copyswapn = (PyArray_CopySwapNFunc*)COMPLEX32_copyswapn;
    _PyComplex32_ArrFuncs.compare = (PyArray_CompareFunc*)COMPLEX32_compare;
    _PyComplex32_ArrFuncs.argmax = (PyArray_ArgFunc*)COMPLEX32_argmax;
    _PyComplex32_ArrFuncs.nonzero = (PyArray_NonzeroFunc*)COMPLEX32_nonzero;
    _PyComplex32_ArrFuncs.fillwithscalar = (PyArray_FillWithScalarFunc*)COMPLEX32_fillwithscalar;
    
    /* The complex32 array descr */
    complex32_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
    complex32_descr->typeobj = &PyComplex32ArrType_Type;
    complex32_descr->kind = 'i';
    complex32_descr->type = 'b';
    complex32_descr->byteorder = '=';
    complex32_descr->type_num = 0; /* assigned at registration */
    complex32_descr->elsize = sizeof(signed char)*2;
    complex32_descr->alignment = 1;
    complex32_descr->subarray = NULL;
    complex32_descr->fields = NULL;
    complex32_descr->names = NULL;
    complex32_descr->f = &_PyComplex32_ArrFuncs;
    
    Py_INCREF(&PyComplex32ArrType_Type);
    complex32Num = PyArray_RegisterDataType(complex32_descr);
    
    if( complex32Num < 0 ) {
#if defined(NPY_PY3K)
        return NULL;
#endif
    }
    
    register_cast_function(NPY_BOOL, complex32Num, (PyArray_VectorUnaryFunc*)BOOL_to_complex32);
    register_cast_function(NPY_BYTE, complex32Num, (PyArray_VectorUnaryFunc*)BYTE_to_complex32);
    
    register_cast_function(complex32Num, NPY_CFLOAT, (PyArray_VectorUnaryFunc*)complex32_to_CFLOAT);
    register_cast_function(complex32Num, NPY_CDOUBLE, (PyArray_VectorUnaryFunc*)complex32_to_CDOUBLE);
    register_cast_function(complex32Num, NPY_CLONGDOUBLE, (PyArray_VectorUnaryFunc*)complex32_to_CLONGDOUBLE);
    
#define REGISTER_UFUNC(name)\
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name),\
            complex32_descr->type_num, complex32_##name##_ufunc, arg_types, NULL)
    
#define REGISTER_SCALAR_UFUNC(name)\
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name),\
            complex32_descr->type_num, complex32_##name##_scalar_ufunc, arg_types, NULL)
    
    /* complex32 -> bool */
    arg_types[0] = complex32_descr->type_num;
    arg_types[1] = NPY_BOOL;
    
    REGISTER_UFUNC(isnan);
    REGISTER_UFUNC(isinf);
    REGISTER_UFUNC(isfinite);
    
    /* complex32 -> double */
    arg_types[1] = NPY_DOUBLE;
    
    REGISTER_UFUNC(absolute);
    
    /* quat -> quat */
    arg_types[1] = complex32_descr->type_num;
    
    REGISTER_UFUNC(negative);
    REGISTER_UFUNC(conjugate);

    /* complex32, complex32 -> bool */

    arg_types[2] = NPY_BOOL;

    REGISTER_UFUNC(equal);
    REGISTER_UFUNC(not_equal);
    REGISTER_UFUNC(less);
    REGISTER_UFUNC(less_equal);
    
    /* complex32, complex32 -> complex32 */

    arg_types[1] = complex32_descr->type_num;

    REGISTER_UFUNC(copysign);

    PyModule_AddObject(m, "complex32", (PyObject *)&PyComplex32ArrType_Type);

#if defined(NPY_PY3K)
    return m;
#endif
}
