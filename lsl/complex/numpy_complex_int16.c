#include "complex_int16.h"

typedef struct {
        PyObject_HEAD
        complexi16 obval;
} PyComplexInt16ScalarObject;

PyMemberDef PyComplexInt16ArrType_members[] = {
    {"real", T_BYTE, offsetof(PyComplexInt16ScalarObject, obval), READONLY,
        "The real part of the complexi16 integer"},
    {"imag", T_BYTE, offsetof(PyComplexInt16ScalarObject, obval)+1, READONLY,
        "The imaginary part of the complexi16 integer"},
    {NULL}
};

static PyObject* PyComplexInt16ArrType_get_real(PyObject *self, void *closure) {
    complexi16 *c = &((PyComplexInt16ScalarObject *)self)->obval;
    PyObject *value = PyInt_FromLong(c->real);
    return value;
}

static PyObject* PyComplexInt16ArrType_get_imag(PyObject *self, void *closure) {
    complexi16 *c = &((PyComplexInt16ScalarObject *)self)->obval;
    PyObject *value = PyInt_FromLong(c->imag);
    return value;
}

PyGetSetDef PyComplexInt16ArrType_getset[] = {
    {"real", PyComplexInt16ArrType_get_real, NULL,
        "The real part of the complexi16 integer", NULL},
    {"imag", PyComplexInt16ArrType_get_imag, NULL,
        "The imaginary part of the complexi16 integer", NULL},
    {NULL}
};

PyTypeObject PyComplexInt16ArrType_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy_complex_int.complex_int16",          /* tp_name*/
    sizeof(PyComplexInt16ScalarObject),         /* tp_basicsize*/
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
    PyComplexInt16ArrType_members,                 /* tp_members */
    PyComplexInt16ArrType_getset,                  /* tp_getset */
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

static PyArray_ArrFuncs _PyComplexInt16_ArrFuncs;
PyArray_Descr *complexi16_descr;

static PyObject* CI16_getitem(char *ip, PyArrayObject *ap) {
    complexi16 c;
    PyObject *tuple;
    
    if( (ap == NULL) || PyArray_ISBEHAVED_RO(ap) ) {
        c = *((complexi16 *) ip);
    } else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_DOUBLE);
        descr->f->copyswap(&c.real, ip,   !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(&c.imag, ip+1, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }

    tuple = PyObject_New(PyComplexInt16ScalarObject, &PyComplexInt16ArrType_Type);
    ((PyComplexInt16ScalarObject *)tuple)->obval = c;
    
    return tuple;
}

static int CI16_setitem(PyObject *op, char *ov, PyArrayObject *ap) {
    complexi16 c;
    
    if( PyArray_IsScalar(op, ComplexInt16) ) {
        c = ((PyComplexInt16ScalarObject *) op)->obval;
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
        *((complexi16 *) ov) = c;
    } else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_INT8);
        descr->f->copyswap(ov,   &c.real, !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(ov+1, &c.imag, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }
    
    return 0;
}

static void CI16_copyswap(complexi16 *dst, complexi16 *src, int swap, void *NPY_UNUSED(arr)) {
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_INT8);
    descr->f->copyswapn(dst, sizeof(signed char), src, sizeof(signed char), 2, swap, NULL);
    Py_DECREF(descr);
}

static void CI16_copyswapn(complexi16 *dst, npy_intp dstride,
                               complexi16 *src, npy_intp sstride,
                               npy_intp n, int swap, void *NPY_UNUSED(arr)) {
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_INT8);
    descr->f->copyswapn(&dst->real, dstride, &src->real, sstride, n, swap, NULL);
    descr->f->copyswapn(&dst->imag, dstride, &src->imag, sstride, n, swap, NULL);
    Py_DECREF(descr);    
}

static int CI16_compare(complexi16 *pa, complexi16 *pb, PyArrayObject *NPY_UNUSED(ap)) {
    complexi16 a = *pa, b = *pb;
    npy_bool anan, bnan;
    int ret;
    
    anan = complexi16_isnan(a);
    bnan = complexi16_isnan(b);
    
    if( anan ) {
        ret = bnan ? 0 : -1;
    } else if( bnan ) {
        ret = 1;
    } else if( complexi16_less(a, b) ) {
        ret = -1;
    } else if( complexi16_less(b, a) ) {
        ret = 1;
    } else {
        ret = 0;
    }
    
    return ret;
}

static int CI16_argmax(complexi16 *ip, npy_intp n, npy_intp *max_ind, PyArrayObject *NPY_UNUSED(aip)) {
    npy_intp i;
    complexi16 mp = *ip;
    
    *max_ind = 0;
    
    if( complexi16_isnan(mp) ) {
        /* nan encountered; it's maximal */
        return 0;
    }

    for(i = 1; i < n; i++) {
        ip++;
        /*
         * Propagate nans, similarly as max() and min()
         */
        if( !(complexi16_less_equal(*ip, mp)) ) {  /* negated, for correct nan handling */
            mp = *ip;
            *max_ind = i;
            if (complexi16_isnan(mp)) {
                /* nan encountered, it's maximal */
                break;
            }
        }
    }
    return 0;
}

static npy_bool CI16_nonzero(char *ip, PyArrayObject *ap) {
    complexi16 c;
    if (ap == NULL || PyArray_ISBEHAVED_RO(ap)) {
        c = *(complexi16 *) ip;
    }
    else {
        PyArray_Descr *descr;
        descr = PyArray_DescrFromType(NPY_INT8);
        descr->f->copyswap(&c.real, ip,   !PyArray_ISNOTSWAPPED(ap), NULL);
        descr->f->copyswap(&c.imag, ip+1, !PyArray_ISNOTSWAPPED(ap), NULL);
        Py_DECREF(descr);
    }
    return (npy_bool) !complexi16_equal(c, (complexi16) {0,0});
}

static void CI16_fillwithscalar(complexi16 *buffer, npy_intp length, complexi16 *value, void *NPY_UNUSED(ignored)) {
    npy_intp i;
    complexi16 val = *value;

    for (i = 0; i < length; ++i) {
        buffer[i] = val;
    }
}

#define MAKE_T_TO_CI16(TYPE, type)                                        \
static void                                                                    \
TYPE ## _to_complexi16(type *ip, complexi16 *op, npy_intp n,                     \
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
{                                                                              \
    while (n--) {                                                              \
        op->real = (signed char) (*ip++);                                      \
        op->imag = 0;                                                          \
        *op++;                                                                 \
    }                                                                          \
}

MAKE_T_TO_CI16(BOOL, npy_bool);
MAKE_T_TO_CI16(BYTE, npy_byte);

#define MAKE_CI16_TO_CT(TYPE, type)                                       \
static void                                                                    \
complexi16_to_## TYPE(complexi16* ip, type *op, npy_intp n,                      \
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
{                                                                              \
    while (n--) {                                                              \
        *(op++) = (type) ip->real;                                             \
        *(op++) = (type) ip->imag;                                             \
        (*ip++);                                                               \
    }                                                                          \
}

MAKE_CI16_TO_CT(CFLOAT, npy_float);
MAKE_CI16_TO_CT(CDOUBLE, npy_double);
MAKE_CI16_TO_CT(CLONGDOUBLE, npy_longdouble);

static void resister_cast_function_ci16(int sourceType, int destType, PyArray_VectorUnaryFunc *castfunc) {
    PyArray_Descr *descr = PyArray_DescrFromType(sourceType);
    PyArray_RegisterCastFunc(descr, destType, castfunc);
    PyArray_RegisterCanCast(descr, destType, NPY_NOSCALAR);
    Py_DECREF(descr);
}

static PyObject* complexi16_arrtype_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    complexi16 c;

    if( !PyArg_ParseTuple(args, "ii", &c.real, &c.imag) ) {
        return NULL;
    }
    
    return PyArray_Scalar(&c, complexi16_descr, NULL);
}

static PyObject* gentype_richcompare_ci16(PyObject *self, PyObject *other, int cmp_op) {
    PyObject *arr, *ret;

    arr = PyArray_FromScalar(self, NULL);
    if (arr == NULL) {
        return NULL;
    }
    ret = Py_TYPE(arr)->tp_richcompare(arr, other, cmp_op);
    Py_DECREF(arr);
    return ret;
}

static long complexi16_arrtype_hash(PyObject *o) {
    complexi16 c = ((PyComplexInt16ScalarObject *)o)->obval;
    long value = 0x456789;
    value = (10000004 * value) ^ _Py_HashBytes(&(c.real), sizeof(signed char));
    value = (10000004 * value) ^ _Py_HashBytes(&(c.imag), sizeof(signed char));
    if( value == -1 ) {
        value = -2;
    }
    return value;
}

static PyObject* complexi16_arrtype_repr(PyObject *o) {
    char str[64];
    complexi16 c = ((PyComplexInt16ScalarObject *)o)->obval;
    sprintf(str, "complex_int16(%i, %i)", c.real, c.imag);
    return PyUString_FromString(str);
}

static PyObject* complexi16_arrtype_str(PyObject *o) {
    char str[64];
    complexi16 c = ((PyComplexInt16ScalarObject *)o)->obval;
    sprintf(str, "%i%+ij", c.real, c.imag);
    return PyUString_FromString(str);
}

#define UNARY_UFUNC_CI16(name, ret_type)\
static void \
complexi16_##name##_ufunc(char** args, npy_intp* dimensions,\
    npy_intp* steps, void* data) {\
    char *ip1 = args[0], *op1 = args[1];\
    npy_intp is1 = steps[0], os1 = steps[1];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, op1 += os1){\
        const complexi16 in1 = *(complexi16 *)ip1;\
        *((ret_type *)op1) = complexi16_##name(in1);};}

UNARY_UFUNC_CI16(isnan, npy_bool)
UNARY_UFUNC_CI16(isinf, npy_bool)
UNARY_UFUNC_CI16(isfinite, npy_bool)
UNARY_UFUNC_CI16(absolute, npy_double)
UNARY_UFUNC_CI16(negative, complexi16)
UNARY_UFUNC_CI16(conjugate, complexi16)

#define BINARY_GEN_UFUNC_CI16(name, func_name, arg_type, ret_type)\
static void \
complexi16_##func_name##_ufunc(char** args, npy_intp* dimensions,\
    npy_intp* steps, void* data) {\
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];\
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1){\
        const complexi16 in1 = *(complexi16 *)ip1;\
        const arg_type in2 = *(arg_type *)ip2;\
        *((ret_type *)op1) = complexi16_##func_name(in1, in2);};};

#define BINARY_UFUNC_CI16(name, ret_type)\
    BINARY_GEN_UFUNC_CI16(name, name, complexi16, ret_type)
#define BINARY_SCALAR_UFUNC_CI16(name, ret_type)\
    BINARY_GEN_UFUNC_CI16(name, name##_scalar, npy_int16, ret_type)

BINARY_UFUNC_CI16(equal, npy_bool)
BINARY_UFUNC_CI16(not_equal, npy_bool)
BINARY_UFUNC_CI16(less, npy_bool)
BINARY_UFUNC_CI16(less_equal, npy_bool)

int create_complex_int16(PyObject* m, PyObject* numpy_dict) {
    int complexi16Num;
    int arg_types[3];
    
    /* Register the complexi16 array scalar type */
#if defined(NPY_PY3K)
    PyComplexInt16ArrType_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
#else
    PyComplexInt16ArrType_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES;
#endif
    PyComplexInt16ArrType_Type.tp_new = complexi16_arrtype_new;
    PyComplexInt16ArrType_Type.tp_richcompare = gentype_richcompare_ci16;
    PyComplexInt16ArrType_Type.tp_hash = complexi16_arrtype_hash;
    PyComplexInt16ArrType_Type.tp_repr = complexi16_arrtype_repr;
    PyComplexInt16ArrType_Type.tp_str = complexi16_arrtype_str;
    PyComplexInt16ArrType_Type.tp_base = &PyGenericArrType_Type;
    if( PyType_Ready(&PyComplexInt16ArrType_Type) < 0 ) {
        return -2;
    }
    
    /* The array functions */
    PyArray_InitArrFuncs(&_PyComplexInt16_ArrFuncs);
    _PyComplexInt16_ArrFuncs.getitem = (PyArray_GetItemFunc*)CI16_getitem;
    _PyComplexInt16_ArrFuncs.setitem = (PyArray_SetItemFunc*)CI16_setitem;
    _PyComplexInt16_ArrFuncs.copyswap = (PyArray_CopySwapFunc*)CI16_copyswap;
    _PyComplexInt16_ArrFuncs.copyswapn = (PyArray_CopySwapNFunc*)CI16_copyswapn;
    _PyComplexInt16_ArrFuncs.compare = (PyArray_CompareFunc*)CI16_compare;
    _PyComplexInt16_ArrFuncs.argmax = (PyArray_ArgFunc*)CI16_argmax;
    _PyComplexInt16_ArrFuncs.nonzero = (PyArray_NonzeroFunc*)CI16_nonzero;
    _PyComplexInt16_ArrFuncs.fillwithscalar = (PyArray_FillWithScalarFunc*)CI16_fillwithscalar;
    
    /* The complexi16 array descr */
    complexi16_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
    complexi16_descr->typeobj = &PyComplexInt16ArrType_Type;
    complexi16_descr->kind = 'i';
    complexi16_descr->type = 'b';
    complexi16_descr->byteorder = '=';
    complexi16_descr->type_num = 0; /* assigned at registration */
    complexi16_descr->elsize = sizeof(unsigned char)*1;
    complexi16_descr->alignment = 1;
    complexi16_descr->subarray = NULL;
    complexi16_descr->fields = NULL;
    complexi16_descr->names = NULL;
    complexi16_descr->f = &_PyComplexInt16_ArrFuncs;
    
    Py_INCREF(&PyComplexInt16ArrType_Type);
    complexi16Num = PyArray_RegisterDataType(complexi16_descr);
    lsl_register_complex_int(8, complexi16Num);
    
    if( complexi16Num < 0 ) {
        return -1;
    }
    
    resister_cast_function_ci16(NPY_BOOL, complexi16Num, (PyArray_VectorUnaryFunc*)BOOL_to_complexi16);
    resister_cast_function_ci16(NPY_BYTE, complexi16Num, (PyArray_VectorUnaryFunc*)BYTE_to_complexi16);
    
    resister_cast_function_ci16(complexi16Num, NPY_CFLOAT, (PyArray_VectorUnaryFunc*)complexi16_to_CFLOAT);
    resister_cast_function_ci16(complexi16Num, NPY_CDOUBLE, (PyArray_VectorUnaryFunc*)complexi16_to_CDOUBLE);
    resister_cast_function_ci16(complexi16Num, NPY_CLONGDOUBLE, (PyArray_VectorUnaryFunc*)complexi16_to_CLONGDOUBLE);
    
#define REGISTER_UFUNC_CI16(name)\
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name),\
            complexi16_descr->type_num, complexi16_##name##_ufunc, arg_types, NULL)
    
#define REGISTER_SCALAR_UFUNC_CI16(name)\
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name),\
            complexi16_descr->type_num, complexi16_##name##_scalar_ufunc, arg_types, NULL)
    
    /* complexi16 -> bool */
    arg_types[0] = complexi16_descr->type_num;
    arg_types[1] = NPY_BOOL;
    
    REGISTER_UFUNC_CI16(isnan);
    REGISTER_UFUNC_CI16(isinf);
    REGISTER_UFUNC_CI16(isfinite);
    
    /* complexi16 -> double */
    arg_types[1] = NPY_DOUBLE;
    
    REGISTER_UFUNC_CI16(absolute);
    
    /* complexi16 -> complexi16 */
    arg_types[1] = complexi16_descr->type_num;
    
    REGISTER_UFUNC_CI16(negative);
    REGISTER_UFUNC_CI16(conjugate);

    /* complexi16, complexi16 -> bool */

    arg_types[2] = NPY_BOOL;

    REGISTER_UFUNC_CI16(equal);
    REGISTER_UFUNC_CI16(not_equal);
    REGISTER_UFUNC_CI16(less);
    REGISTER_UFUNC_CI16(less_equal);

    PyModule_AddObject(m, "complex_int16", (PyObject *)&PyComplexInt16ArrType_Type);

    return complexi16Num;
}
