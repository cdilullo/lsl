#include "complex_int8.h"

typedef struct {
        PyObject_HEAD
        complexi8 obval;
} PyComplexInt8ScalarObject;

PyMemberDef PyComplexInt8ArrType_members[] = {
    {"real_imag", T_BYTE, offsetof(PyComplexInt8ScalarObject, obval), READONLY,
        "The real and imaginary parts of the complexi8 integer"},
    {NULL}
};

static PyObject* PyComplexInt8ArrType_get_real(PyObject *self, void *closure) {
    complexi8 *c = &((PyComplexInt8ScalarObject *)self)->obval;
    const signed char* sc = fourBitLUT[c->real_imag];
    PyObject *value = PyInt_FromLong(sc[0]);
    return value;
}

static PyObject* PyComplexInt8ArrType_get_imag(PyObject *self, void *closure) {
    complexi8 *c = &((PyComplexInt8ScalarObject *)self)->obval;
    const signed char* sc = fourBitLUT[c->real_imag];
    PyObject *value = PyInt_FromLong(sc[1]);
    return value;
}

PyGetSetDef PyComplexInt8ArrType_getset[] = {
    {"real", PyComplexInt8ArrType_get_real, NULL,
        "The real part of the complexi8 integer", NULL},
    {"imag", PyComplexInt8ArrType_get_imag, NULL,
        "The imaginary part of the complexi8 integer", NULL},
    {NULL}
};

PyTypeObject PyComplexInt8ArrType_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy_complex_int.complex_int8",           /* tp_name*/
    sizeof(PyComplexInt8ScalarObject),          /* tp_basicsize*/
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
    PyComplexInt8ArrType_members,                  /* tp_members */
    PyComplexInt8ArrType_getset,                   /* tp_getset */
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

static PyArray_ArrFuncs _PyComplexInt8_ArrFuncs;
PyArray_Descr *complexi8_descr;

static PyObject* CI8_getitem(char *ip, PyArrayObject *ap) {
    complexi8 c;
    PyObject *tuple;
    
    if( (ap == NULL) || PyArray_ISBEHAVED_RO(ap) ) {
        c = *((complexi8 *) ip);
    } else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_INT8);
        descr->f->copyswap(&c.real_imag, ip, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }
    
    tuple = PyObject_New(PyComplexInt8ScalarObject, &PyComplexInt8ArrType_Type);
    ((PyComplexInt8ScalarObject *)tuple)->obval = c;
    
    return tuple;
}

static int CI8_setitem(PyObject *op, char *ov, PyArrayObject *ap) {
    complexi8 c;
    
    if( PyArray_IsScalar(op, ComplexInt8) ) {
        c = ((PyComplexInt8ScalarObject *) op)->obval;
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
        *((complexi8 *) ov) = c;
    } else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_INT8);
        descr->f->copyswap(ov, &c.real_imag, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }
    
    return 0;
}

static void CI8_copyswap(complexi8 *dst, complexi8 *src, int swap, void *NPY_UNUSED(arr)) {
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_INT8);
    descr->f->copyswapn(dst, sizeof(unsigned char), src, sizeof(unsigned char), 1, swap, NULL);
    Py_DECREF(descr);
}

static void CI8_copyswapn(complexi8 *dst, npy_intp dstride,
                               complexi8 *src, npy_intp sstride,
                               npy_intp n, int swap, void *NPY_UNUSED(arr)) {
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_INT8);
    descr->f->copyswapn(&dst->real_imag, dstride, &src->real_imag, sstride, n, swap, NULL);
    Py_DECREF(descr);    
}

static int CI8_compare(complexi8 *pa, complexi8 *pb, PyArrayObject *NPY_UNUSED(ap)) {
    complexi8 a = *pa, b = *pb;
    npy_bool anan, bnan;
    int ret;
    
    anan = complexi8_isnan(a);
    bnan = complexi8_isnan(b);
    
    if( anan ) {
        ret = bnan ? 0 : -1;
    } else if( bnan ) {
        ret = 1;
    } else if( complexi8_less(a, b) ) {
        ret = -1;
    } else if( complexi8_less(b, a) ) {
        ret = 1;
    } else {
        ret = 0;
    }
    
    return ret;
}

static int CI8_argmax(complexi8 *ip, npy_intp n, npy_intp *max_ind, PyArrayObject *NPY_UNUSED(aip)) {
    npy_intp i;
    complexi8 mp = *ip;
    
    *max_ind = 0;
    
    if( complexi8_isnan(mp) ) {
        /* nan encountered; it's maximal */
        return 0;
    }

    for(i = 1; i < n; i++) {
        ip++;
        /*
         * Propagate nans, similarly as max() and min()
         */
        if( !(complexi8_less_equal(*ip, mp)) ) {  /* negated, for correct nan handling */
            mp = *ip;
            *max_ind = i;
            if (complexi8_isnan(mp)) {
                /* nan encountered, it's maximal */
                break;
            }
        }
    }
    return 0;
}

static npy_bool CI8_nonzero(char *ip, PyArrayObject *ap) {
    complexi8 c;
    if (ap == NULL || PyArray_ISBEHAVED_RO(ap)) {
        c = *(complexi8 *) ip;
    }
    else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_INT8);
        descr->f->copyswap(&c.real_imag, ip, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }
    return (npy_bool) !complexi8_equal(c, (complexi8) {0});
}

static void CI8_fillwithscalar(complexi8 *buffer, npy_intp length, complexi8 *value, void *NPY_UNUSED(ignored)) {
    npy_intp i;
    complexi8 val = *value;

    for (i = 0; i < length; ++i) {
        buffer[i] = val;
    }
}

#define MAKE_T_TO_CI8(TYPE, type)                                         \
static void                                                                    \
TYPE ## _to_complexi8(type *ip, complexi8 *op, npy_intp n,                       \
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
{                                                                              \
    while (n--) {                                                              \
        op->real_imag = (unsigned char) ((*ip++) * 16);                        \
        *op++;                                                                 \
    }                                                                          \
}

MAKE_T_TO_CI8(BOOL, npy_bool);
MAKE_T_TO_CI8(BYTE, npy_byte);

#define MAKE_CI8_TO_CT(TYPE, type)                                        \
static void                                                                    \
complexi8_to_## TYPE(complexi8* ip, type *op, npy_intp n,                        \
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

MAKE_CI8_TO_CT(CFLOAT, npy_float);
MAKE_CI8_TO_CT(CDOUBLE, npy_double);
MAKE_CI8_TO_CT(CLONGDOUBLE, npy_longdouble);

static void resister_cast_function_ci8(int sourceType, int destType, PyArray_VectorUnaryFunc *castfunc) {
    PyArray_Descr *descr = PyArray_DescrFromType(sourceType);
    PyArray_RegisterCastFunc(descr, destType, castfunc);
    PyArray_RegisterCanCast(descr, destType, NPY_NOSCALAR);
    Py_DECREF(descr);
}

static PyObject* complexi8_arrtype_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    complexi8 c;

    if( !PyArg_ParseTuple(args, "i", &c.real_imag) ) {
        return NULL;
    }
    
    return PyArray_Scalar(&c, complexi8_descr, NULL);
}

static PyObject* gentype_richcompare_ci8(PyObject *self, PyObject *other, int cmp_op) {
    PyObject *arr, *ret;

    arr = PyArray_FromScalar(self, NULL);
    if (arr == NULL) {
        return NULL;
    }
    ret = Py_TYPE(arr)->tp_richcompare(arr, other, cmp_op);
    Py_DECREF(arr);
    return ret;
}

static long complexi8_arrtype_hash(PyObject *o) {
    complexi8 c = ((PyComplexInt8ScalarObject *)o)->obval;
    long value = 0x456789;
    value = (10000004 * value) ^ _Py_HashBytes(&(c.real_imag), sizeof(unsigned char));
    if( value == -1 ) {
        value = -2;
    }
    return value;
}

static PyObject* complexi8_arrtype_repr(PyObject *o) {
    char str[64];
    complexi8 c = ((PyComplexInt8ScalarObject *)o)->obval;
    sprintf(str, "complex_int8(%u)", c.real_imag);
    return PyUString_FromString(str);
}

static PyObject* complexi8_arrtype_str(PyObject *o) {
    char str[64];
    complexi8 c = ((PyComplexInt8ScalarObject *)o)->obval;
    const signed char* sc = fourBitLUT[c.real_imag];
    sprintf(str, "%i%+ij", sc[0], sc[1]);
    return PyUString_FromString(str);
}

#define UNARY_UFUNC_CI8(name, ret_type)\
static void \
complexi8_##name##_ufunc(char** args, npy_intp* dimensions,\
    npy_intp* steps, void* data) {\
    char *ip1 = args[0], *op1 = args[1];\
    npy_intp is1 = steps[0], os1 = steps[1];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, op1 += os1){\
        const complexi8 in1 = *(complexi8 *)ip1;\
        *((ret_type *)op1) = complexi8_##name(in1);};}

UNARY_UFUNC_CI8(isnan, npy_bool)
UNARY_UFUNC_CI8(isinf, npy_bool)
UNARY_UFUNC_CI8(isfinite, npy_bool)
UNARY_UFUNC_CI8(absolute, npy_double)
UNARY_UFUNC_CI8(negative, complexi8)
UNARY_UFUNC_CI8(conjugate, complexi8)

#define BINARY_GEN_UFUNC_CI8(name, func_name, arg_type, ret_type)\
static void \
complexi8_##func_name##_ufunc(char** args, npy_intp* dimensions,\
    npy_intp* steps, void* data) {\
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];\
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1){\
        const complexi8 in1 = *(complexi8 *)ip1;\
        const arg_type in2 = *(arg_type *)ip2;\
        *((ret_type *)op1) = complexi8_##func_name(in1, in2);};};

#define BINARY_UFUNC_CI8(name, ret_type)\
    BINARY_GEN_UFUNC_CI8(name, name, complexi8, ret_type)
#define BINARY_SCALAR_UFUNC_CI8(name, ret_type)\
    BINARY_GEN_UFUNC_CI8(name, name##_scalar, npy_int8, ret_type)

BINARY_UFUNC_CI8(equal, npy_bool)
BINARY_UFUNC_CI8(not_equal, npy_bool)
BINARY_UFUNC_CI8(less, npy_bool)
BINARY_UFUNC_CI8(less_equal, npy_bool)

int create_complex_int8(PyObject* m, PyObject* numpy_dict) {
    int complexi8Num;
    int arg_types[3];
    
    /* Fill the lookup table */
    complexi8_fillLUT();
    
    /* Register the complexi8 array scalar type */
#if defined(NPY_PY3K)
    PyComplexInt8ArrType_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
#else
    PyComplexInt8ArrType_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES;
#endif
    PyComplexInt8ArrType_Type.tp_new = complexi8_arrtype_new;
    PyComplexInt8ArrType_Type.tp_richcompare = gentype_richcompare_ci8;
    PyComplexInt8ArrType_Type.tp_hash = complexi8_arrtype_hash;
    PyComplexInt8ArrType_Type.tp_repr = complexi8_arrtype_repr;
    PyComplexInt8ArrType_Type.tp_str = complexi8_arrtype_str;
    PyComplexInt8ArrType_Type.tp_base = &PyGenericArrType_Type;
    if( PyType_Ready(&PyComplexInt8ArrType_Type) < 0 ) {
        return -2;
    }
    
    /* The array functions */
    PyArray_InitArrFuncs(&_PyComplexInt8_ArrFuncs);
    _PyComplexInt8_ArrFuncs.getitem = (PyArray_GetItemFunc*)CI8_getitem;
    _PyComplexInt8_ArrFuncs.setitem = (PyArray_SetItemFunc*)CI8_setitem;
    _PyComplexInt8_ArrFuncs.copyswap = (PyArray_CopySwapFunc*)CI8_copyswap;
    _PyComplexInt8_ArrFuncs.copyswapn = (PyArray_CopySwapNFunc*)CI8_copyswapn;
    _PyComplexInt8_ArrFuncs.compare = (PyArray_CompareFunc*)CI8_compare;
    _PyComplexInt8_ArrFuncs.argmax = (PyArray_ArgFunc*)CI8_argmax;
    _PyComplexInt8_ArrFuncs.nonzero = (PyArray_NonzeroFunc*)CI8_nonzero;
    _PyComplexInt8_ArrFuncs.fillwithscalar = (PyArray_FillWithScalarFunc*)CI8_fillwithscalar;
    
    /* The complexi8 array descr */
    complexi8_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
    complexi8_descr->typeobj = &PyComplexInt8ArrType_Type;
    complexi8_descr->kind = 'i';
    complexi8_descr->type = 'b';
    complexi8_descr->byteorder = '=';
    complexi8_descr->type_num = 0; /* assigned at registration */
    complexi8_descr->elsize = sizeof(unsigned char)*1;
    complexi8_descr->alignment = 1;
    complexi8_descr->subarray = NULL;
    complexi8_descr->fields = NULL;
    complexi8_descr->names = NULL;
    complexi8_descr->f = &_PyComplexInt8_ArrFuncs;
    
    Py_INCREF(&PyComplexInt8ArrType_Type);
    complexi8Num = PyArray_RegisterDataType(complexi8_descr);
    lsl_register_complex_int(8, complexi8Num);
    
    if( complexi8Num < 0 ) {
        return -1;
    }
    
    resister_cast_function_ci8(NPY_BOOL, complexi8Num, (PyArray_VectorUnaryFunc*)BOOL_to_complexi8);
    resister_cast_function_ci8(NPY_BYTE, complexi8Num, (PyArray_VectorUnaryFunc*)BYTE_to_complexi8);
    
    resister_cast_function_ci8(complexi8Num, NPY_CFLOAT, (PyArray_VectorUnaryFunc*)complexi8_to_CFLOAT);
    resister_cast_function_ci8(complexi8Num, NPY_CDOUBLE, (PyArray_VectorUnaryFunc*)complexi8_to_CDOUBLE);
    resister_cast_function_ci8(complexi8Num, NPY_CLONGDOUBLE, (PyArray_VectorUnaryFunc*)complexi8_to_CLONGDOUBLE);
    
#define REGISTER_UFUNC_CI8(name)\
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name),\
            complexi8_descr->type_num, complexi8_##name##_ufunc, arg_types, NULL)
    
#define REGISTER_SCALAR_UFUNC_CI8(name)\
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name),\
            complexi8_descr->type_num, complexi8_##name##_scalar_ufunc, arg_types, NULL)
    
    /* complexi8 -> bool */
    arg_types[0] = complexi8_descr->type_num;
    arg_types[1] = NPY_BOOL;
    
    REGISTER_UFUNC_CI8(isnan);
    REGISTER_UFUNC_CI8(isinf);
    REGISTER_UFUNC_CI8(isfinite);
    
    /* complexi8 -> double */
    arg_types[1] = NPY_DOUBLE;
    
    REGISTER_UFUNC_CI8(absolute);
    
    /* complexi8 -> complexi8 */
    arg_types[1] = complexi8_descr->type_num;
    
    REGISTER_UFUNC_CI8(negative);
    REGISTER_UFUNC_CI8(conjugate);

    /* complexi8, complexi8 -> bool */

    arg_types[2] = NPY_BOOL;

    REGISTER_UFUNC_CI8(equal);
    REGISTER_UFUNC_CI8(not_equal);
    REGISTER_UFUNC_CI8(less);
    REGISTER_UFUNC_CI8(less_equal);

    PyModule_AddObject(m, "complex_int8", (PyObject *)&PyComplexInt8ArrType_Type);

    return complexi8Num;
}
