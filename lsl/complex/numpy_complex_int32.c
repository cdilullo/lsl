#include "complex_int32.h"

typedef struct {
        PyObject_HEAD
        complexi32 obval;
} PyComplexInt32ScalarObject;

PyMemberDef PyComplexInt32ArrType_members[] = {
    {"real", T_BYTE, offsetof(PyComplexInt32ScalarObject, obval), READONLY,
        "The real part of the complexi32 integer"},
    {"imag", T_BYTE, offsetof(PyComplexInt32ScalarObject, obval)+1, READONLY,
        "The imaginary part of the complexi32 integer"},
    {NULL}
};

static PyObject* PyComplexInt32ArrType_get_real(PyObject *self, void *closure) {
    complexi32 *c = &((PyComplexInt32ScalarObject *)self)->obval;
    PyObject *value = PyInt_FromLong(c->real);
    return value;
}

static PyObject* PyComplexInt32ArrType_get_imag(PyObject *self, void *closure) {
    complexi32 *c = &((PyComplexInt32ScalarObject *)self)->obval;
    PyObject *value = PyInt_FromLong(c->imag);
    return value;
}

PyGetSetDef PyComplexInt32ArrType_getset[] = {
    {"real", PyComplexInt32ArrType_get_real, NULL,
        "The real part of the complexi32 integer", NULL},
    {"imag", PyComplexInt32ArrType_get_imag, NULL,
        "The imaginary part of the complexi32 integer", NULL},
    {NULL}
};

PyTypeObject PyComplexInt32ArrType_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy_complex_int.complex_int32",          /* tp_name*/
    sizeof(PyComplexInt32ScalarObject),         /* tp_basicsize*/
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
    PyComplexInt32ArrType_members,                 /* tp_members */
    PyComplexInt32ArrType_getset,                  /* tp_getset */
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

static PyArray_ArrFuncs _PyComplexInt32_ArrFuncs;
PyArray_Descr *complexi32_descr;

static PyObject* CI32_getitem(char *ip, PyArrayObject *ap) {
    complexi32 c;
    PyObject *item;
    
    if( (ap == NULL) || PyArray_ISBEHAVED_RO(ap) ) {
        c = *((complexi32 *) ip);
    } else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_DOUBLE);
        descr->f->copyswap(&c.real, ip,   !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(&c.imag, ip+1, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }

    item = (PyObject*) PyObject_New(PyComplexInt32ScalarObject, &PyComplexInt32ArrType_Type);
    ((PyComplexInt32ScalarObject *)item)->obval = c;
    
    return item;
}

static int CI32_setitem(PyObject *op, char *ov, PyArrayObject *ap) {
    complexi32 c;
    
    if( PyArray_IsScalar(op, ComplexInt32) ) {
        c = ((PyComplexInt32ScalarObject *) op)->obval;
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
        *((complexi32 *) ov) = c;
    } else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_INT8);
        descr->f->copyswap(ov,   &c.real, !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(ov+1, &c.imag, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }
    
    return 0;
}

static void CI32_copyswap(complexi32 *dst, complexi32 *src, int swap, void *NPY_UNUSED(arr)) {
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_INT8);
    descr->f->copyswapn(dst, sizeof(signed char), src, sizeof(signed char), 2, swap, NULL);
    Py_DECREF(descr);
}

static void CI32_copyswapn(complexi32 *dst, npy_intp dstride,
                               complexi32 *src, npy_intp sstride,
                               npy_intp n, int swap, void *NPY_UNUSED(arr)) {
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_INT8);
    descr->f->copyswapn(&dst->real, dstride, &src->real, sstride, n, swap, NULL);
    descr->f->copyswapn(&dst->imag, dstride, &src->imag, sstride, n, swap, NULL);
    Py_DECREF(descr);    
}

static int CI32_compare(complexi32 *pa, complexi32 *pb, PyArrayObject *NPY_UNUSED(ap)) {
    complexi32 a = *pa, b = *pb;
    npy_bool anan, bnan;
    int ret;
    
    anan = complexi32_isnan(a);
    bnan = complexi32_isnan(b);
    
    if( anan ) {
        ret = bnan ? 0 : -1;
    } else if( bnan ) {
        ret = 1;
    } else if( complexi32_less(a, b) ) {
        ret = -1;
    } else if( complexi32_less(b, a) ) {
        ret = 1;
    } else {
        ret = 0;
    }
    
    return ret;
}

static int CI32_argmax(complexi32 *ip, npy_intp n, npy_intp *max_ind, PyArrayObject *NPY_UNUSED(aip)) {
    npy_intp i;
    complexi32 mp = *ip;
    
    *max_ind = 0;
    
    if( complexi32_isnan(mp) ) {
        /* nan encountered; it's maximal */
        return 0;
    }

    for(i = 1; i < n; i++) {
        ip++;
        /*
         * Propagate nans, similarly as max() and min()
         */
        if( !(complexi32_less_equal(*ip, mp)) ) {  /* negated, for correct nan handling */
            mp = *ip;
            *max_ind = i;
            if (complexi32_isnan(mp)) {
                /* nan encountered, it's maximal */
                break;
            }
        }
    }
    return 0;
}

static npy_bool CI32_nonzero(char *ip, PyArrayObject *ap) {
    complexi32 c;
    if (ap == NULL || PyArray_ISBEHAVED_RO(ap)) {
        c = *(complexi32 *) ip;
    }
    else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_INT8);
        descr->f->copyswap(&c.real, ip,   !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(&c.imag, ip+1, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }
    return (npy_bool) !complexi32_equal(c, (complexi32) {0,0});
}

static void CI32_fillwithscalar(complexi32 *buffer, npy_intp length, complexi32 *value, void *NPY_UNUSED(ignored)) {
    npy_intp i;
    complexi32 val = *value;

    for (i = 0; i < length; ++i) {
        buffer[i] = val;
    }
}

#define MAKE_T_TO_CI32(TYPE, type)                                       \
static void TYPE ## _to_complexi32(type *ip, complexi32 *op, npy_intp n, \
                                   PyArrayObject *NPY_UNUSED(aip),       \
                                   PyArrayObject *NPY_UNUSED(aop)) {     \
    while (n--) {                                                        \
        op->real = (signed char) (*ip++);                                \
        op->imag = 0;                                                    \
        *op++;                                                           \
    }                                                                    \
}

MAKE_T_TO_CI32(BOOL, npy_bool);
MAKE_T_TO_CI32(BYTE, npy_byte);

#define MAKE_CI32_TO_CT(TYPE, type)                                     \
static void complexi32_to_## TYPE(complexi32* ip, type *op, npy_intp n, \
                                  PyArrayObject *NPY_UNUSED(aip),       \
                                  PyArrayObject *NPY_UNUSED(aop)) {     \
    while (n--) {                                                       \
        *(op++) = (type) ip->real;                                      \
        *(op++) = (type) ip->imag;                                      \
        (*ip++);                                                        \
    }                                                                   \
}

MAKE_CI32_TO_CT(CFLOAT, npy_float);
MAKE_CI32_TO_CT(CDOUBLE, npy_double);
MAKE_CI32_TO_CT(CLONGDOUBLE, npy_longdouble);

static void resister_cast_function_ci32(int sourceType, int destType, PyArray_VectorUnaryFunc *castfunc) {
    PyArray_Descr *descr = PyArray_DescrFromType(sourceType);
    PyArray_RegisterCastFunc(descr, destType, castfunc);
    PyArray_RegisterCanCast(descr, destType, NPY_NOSCALAR);
    Py_DECREF(descr);
}

static PyObject* complexi32_arrtype_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    complexi32 c;
    Py_complex cmplx;

    if( !PyArg_ParseTuple(args, "D", &cmplx) ) {
        return NULL;
    }
    
    c.real = cmplx.real;
    c.imag = cmplx.imag;
    return PyArray_Scalar(&c, complexi32_descr, NULL);
}

static PyObject* gentype_richcompare_ci32(PyObject *self, PyObject *other, int cmp_op) {
    PyObject *arr, *ret;

    arr = PyArray_FromScalar(self, NULL);
    if (arr == NULL) {
        return NULL;
    }
    ret = Py_TYPE(arr)->tp_richcompare(arr, other, cmp_op);
    Py_DECREF(arr);
    return ret;
}

static long complexi32_arrtype_hash(PyObject *o) {
    complexi32 c = ((PyComplexInt32ScalarObject *)o)->obval;
    long value = 0x456789;
    value = (10000004 * value) ^ _Py_HashDouble(c.real);
    value = (10000004 * value) ^ _Py_HashDouble(c.imag);
    if( value == -1 ) {
        value = -2;
    }
    return value;
}

static PyObject* complexi32_arrtype_repr(PyObject *o) {
    char str[64];
    complexi32 c = ((PyComplexInt32ScalarObject *)o)->obval;
    sprintf(str, "complex_int32(%i, %i)", c.real, c.imag);
    return PyUString_FromString(str);
}

static PyObject* complexi32_arrtype_str(PyObject *o) {
    char str[64];
    complexi32 c = ((PyComplexInt32ScalarObject *)o)->obval;
    sprintf(str, "%i%+ij", c.real, c.imag);
    return PyUString_FromString(str);
}

#define UNARY_UFUNC_CI32(name, ret_type)                                 \
static void complexi32_##name##_ufunc(char** args, npy_intp* dimensions, \
                                      npy_intp* steps, void* data) {     \
    char *ip1 = args[0], *op1 = args[1];                                 \
    npy_intp is1 = steps[0], os1 = steps[1];                             \
    npy_intp n = dimensions[0];                                          \
    npy_intp i;                                                          \
    for(i=0; i<n; i++, ip1+=is1, op1+=os1) {                             \
        const complexi32 in1 = *(complexi32 *)ip1;                       \
        *((ret_type *)op1) = complexi32_##name(in1);                     \
    }                                                                    \
}

UNARY_UFUNC_CI32(isnan, npy_bool)
UNARY_UFUNC_CI32(isinf, npy_bool)
UNARY_UFUNC_CI32(isfinite, npy_bool)
UNARY_UFUNC_CI32(absolute, npy_double)
UNARY_UFUNC_CI32(negative, complexi32)
UNARY_UFUNC_CI32(conjugate, complexi32)

#define BINARY_GEN_UFUNC_CI32(name, func_name, arg_type, ret_type)            \
static void complexi32_##func_name##_ufunc(char** args, npy_intp* dimensions, \
                                           npy_intp* steps, void* data) {     \
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];                      \
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];                  \
    npy_intp n = dimensions[0];                                               \
    npy_intp i;                                                               \
    for(i=0; i<n; i++, ip1+=is1, ip2+=is2, op1+=os1) {                        \
        const complexi32 in1 = *(complexi32 *)ip1;                            \
        const arg_type in2 = *(arg_type *)ip2;                                \
        *((ret_type *)op1) = complexi32_##func_name(in1, in2);                \
    }                                                                         \
}

#define BINARY_UFUNC_CI32(name, ret_type)\
    BINARY_GEN_UFUNC_CI32(name, name, complexi32, ret_type)
#define BINARY_SCALAR_UFUNC_CI32(name, ret_type)\
    BINARY_GEN_UFUNC_CI32(name, name##_scalar, npy_int8, ret_type)

BINARY_UFUNC_CI32(equal, npy_bool)
BINARY_UFUNC_CI32(not_equal, npy_bool)
BINARY_UFUNC_CI32(less, npy_bool)
BINARY_UFUNC_CI32(less_equal, npy_bool)

int create_complex_int32(PyObject* m, PyObject* numpy_dict) {
    int complexi32Num;
    int arg_types[3];
    
    /* Register the complexi32 array scalar type */
#if defined(NPY_PY3K)
    PyComplexInt32ArrType_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
#else
    PyComplexInt32ArrType_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES;
#endif
    PyComplexInt32ArrType_Type.tp_new = complexi32_arrtype_new;
    PyComplexInt32ArrType_Type.tp_richcompare = gentype_richcompare_ci32;
    PyComplexInt32ArrType_Type.tp_hash = complexi32_arrtype_hash;
    PyComplexInt32ArrType_Type.tp_repr = complexi32_arrtype_repr;
    PyComplexInt32ArrType_Type.tp_str = complexi32_arrtype_str;
    PyComplexInt32ArrType_Type.tp_base = &PyGenericArrType_Type;
    if( PyType_Ready(&PyComplexInt32ArrType_Type) < 0 ) {
        return -2;
    }
    
    /* The array functions */
    PyArray_InitArrFuncs(&_PyComplexInt32_ArrFuncs);
    _PyComplexInt32_ArrFuncs.getitem = (PyArray_GetItemFunc*)CI32_getitem;
    _PyComplexInt32_ArrFuncs.setitem = (PyArray_SetItemFunc*)CI32_setitem;
    _PyComplexInt32_ArrFuncs.copyswap = (PyArray_CopySwapFunc*)CI32_copyswap;
    _PyComplexInt32_ArrFuncs.copyswapn = (PyArray_CopySwapNFunc*)CI32_copyswapn;
    _PyComplexInt32_ArrFuncs.compare = (PyArray_CompareFunc*)CI32_compare;
    _PyComplexInt32_ArrFuncs.argmax = (PyArray_ArgFunc*)CI32_argmax;
    _PyComplexInt32_ArrFuncs.nonzero = (PyArray_NonzeroFunc*)CI32_nonzero;
    _PyComplexInt32_ArrFuncs.fillwithscalar = (PyArray_FillWithScalarFunc*)CI32_fillwithscalar;
    
    /* The complexi32 array descr */
    complexi32_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
    complexi32_descr->typeobj = &PyComplexInt32ArrType_Type;
    complexi32_descr->kind = 'i';
    complexi32_descr->type = 'b';
    complexi32_descr->byteorder = '=';
    complexi32_descr->type_num = 0; /* assigned at registration */
    complexi32_descr->elsize = sizeof(short int)*2;
    complexi32_descr->alignment = sizeof(short int);
    complexi32_descr->subarray = NULL;
    complexi32_descr->fields = NULL;
    complexi32_descr->names = NULL;
    complexi32_descr->f = &_PyComplexInt32_ArrFuncs;
    
    Py_INCREF(&PyComplexInt32ArrType_Type);
    complexi32Num = PyArray_RegisterDataType(complexi32_descr);
    if( complexi32Num < 0 || complexi32Num != NPY_COMPLEX_INT32 ) {
        return -1;
    }
    
    resister_cast_function_ci32(NPY_BOOL, complexi32Num, (PyArray_VectorUnaryFunc*)BOOL_to_complexi32);
    resister_cast_function_ci32(NPY_BYTE, complexi32Num, (PyArray_VectorUnaryFunc*)BYTE_to_complexi32);
    
    resister_cast_function_ci32(complexi32Num, NPY_CFLOAT, (PyArray_VectorUnaryFunc*)complexi32_to_CFLOAT);
    resister_cast_function_ci32(complexi32Num, NPY_CDOUBLE, (PyArray_VectorUnaryFunc*)complexi32_to_CDOUBLE);
    resister_cast_function_ci32(complexi32Num, NPY_CLONGDOUBLE, (PyArray_VectorUnaryFunc*)complexi32_to_CLONGDOUBLE);
    
#define REGISTER_UFUNC_CI32(name)\
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name),\
            complexi32_descr->type_num, complexi32_##name##_ufunc, arg_types, NULL)
    
#define REGISTER_SCALAR_UFUNC_CI32(name)\
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name),\
            complexi32_descr->type_num, complexi32_##name##_scalar_ufunc, arg_types, NULL)
    
    /* complexi32 -> bool */
    arg_types[0] = complexi32_descr->type_num;
    arg_types[1] = NPY_BOOL;
    
    REGISTER_UFUNC_CI32(isnan);
    REGISTER_UFUNC_CI32(isinf);
    REGISTER_UFUNC_CI32(isfinite);
    
    /* complexi32 -> double */
    arg_types[1] = NPY_DOUBLE;
    
    REGISTER_UFUNC_CI32(absolute);
    
    /* complexi32 -> complexi32 */
    arg_types[1] = complexi32_descr->type_num;
    
    REGISTER_UFUNC_CI32(negative);
    REGISTER_UFUNC_CI32(conjugate);

    /* complexi32, complexi32 -> bool */

    arg_types[2] = NPY_BOOL;

    REGISTER_UFUNC_CI32(equal);
    REGISTER_UFUNC_CI32(not_equal);
    REGISTER_UFUNC_CI32(less);
    REGISTER_UFUNC_CI32(less_equal);

    PyModule_AddObject(m, "complex_int32", (PyObject *)&PyComplexInt32ArrType_Type);

    return complexi32Num;
}
