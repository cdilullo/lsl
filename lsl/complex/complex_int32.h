#ifndef COMPLEX_COMPLEX_INT32_H_INCLUDE_GUARD_
#define COMPLEX_COMPLEX_INT32_H_INCLUDE_GUARD_

#ifdef __cplusplus
extern "C" {
#endif

#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
    
typedef struct {
    short int real;
    short int imag;
} complex_int32;

// Unary bool operators
static NPY_INLINE int complex_int32_nonzero(complex_int32 c) {
    return c.real != 0 || c.imag != 0;
}

static NPY_INLINE int complex_int32_isnan(complex_int32 c) {
    return 0;
}

static NPY_INLINE int complex_int32_isinf(complex_int32 c) {
    return 0;
}

static NPY_INLINE int complex_int32_isfinite(complex_int32 c) {
    return 1;
}

// Binary bool operators
static NPY_INLINE int complex_int32_equal(complex_int32 c1, complex_int32 c2) {
    return 
        !complex_int32_isnan(c1) &&
        !complex_int32_isnan(c2) &&
        c1.real == c2.real &&
        c1.imag == c2.imag;
}

static NPY_INLINE int complex_int32_not_equal(complex_int32 c1, complex_int32 c2) {
    return !complex_int32_equal(c1, c2);
}

static NPY_INLINE int complex_int32_less(complex_int32 c1, complex_int32 c2) {
    return
        (!complex_int32_isnan(c1) &&
         !complex_int32_isnan(c2)) && (
            c1.real != c2.real ? c1.real < c2.real :
            c1.imag != c2.imag ? c1.imag < c2.imag : 0);
}

static NPY_INLINE int complex_int32_greater(complex_int32 c1, complex_int32 c2) {
    return
        (!complex_int32_isnan(c1) &&
         !complex_int32_isnan(c2)) && (
            c1.real != c2.real ? c1.real > c2.real :
            c1.imag != c2.imag ? c1.imag > c2.imag : 0);
}

static NPY_INLINE int complex_int32_less_equal(complex_int32 c1, complex_int32 c2) {
    return
        (!complex_int32_isnan(c1) &&
         !complex_int32_isnan(c2)) && (
            c1.real != c2.real ? c1.real < c2.real :
            c1.imag != c2.imag ? c1.imag < c2.imag : 1);
}

static NPY_INLINE int complex_int32_greater_equal(complex_int32 c1, complex_int32 c2) {
    return
        (!complex_int32_isnan(c1) &&
         !complex_int32_isnan(c2)) && (
            c1.real != c2.real ? c1.real > c2.real :
            c1.imag != c2.imag ? c1.imag > c2.imag : 1);
}

// Unary float returners
static NPY_INLINE double complex_int32_norm(complex_int32 c) {
    return (((int) c.real)*c.real + ((int) c.imag)*c.imag)*1.0;
}

static NPY_INLINE double complex_int32_absolute(complex_int32 c) {
    return sqrt(((int) c.real)*c.real + ((int) c.imag)*c.imag);
}

static NPY_INLINE double complex_int32_angle(complex_int32 c) {
    return atan2((int) c.imag, (int) c.real);
}

// Unary complex_int32 returners
static NPY_INLINE complex_int32 complex_int32_negative(complex_int32 c) {
    short int real = -c.real;
    short int imag = -c.imag;
    return (complex_int32) {real, imag};
}

static NPY_INLINE complex_int32 complex_int32_conjugate(complex_int32 c) {
    short int real =  c.real;
    short int imag = -c.imag;
    return (complex_int32) {real, imag};
}

// complex_int32-complex_int32/complex_int32-scalar/scalar-complex_int32 binary complex_int32 returners
static NPY_INLINE complex_int32 complex_int32_add(complex_int32 c1, complex_int32 c2) {
    short int real = c1.real + c2.real;
    short int imag = c1.imag + c2.imag;
    return (complex_int32) {real, imag};
}

static NPY_INLINE void complex_int32_inplace_add(complex_int32* c1, complex_int32 c2) {
    short int real = c1->real + c2.real;
    short int imag = c1->imag + c2.imag;
    c1->real = real;
    c1->imag = imag;
}

static NPY_INLINE complex_int32 complex_int32_scalar_add(npy_cdouble s, complex_int32 c) {
    short int real = s.real + c.real;
    short int imag = s.imag + c.imag;
    return (complex_int32) {real, imag};
}

static NPY_INLINE void complex_int32_inplace_scalar_add(npy_cdouble s, complex_int32* c) {
    short int real = s.real + c->real;
    short int imag = s.real + c->imag;
    c->real = real;
    c->imag = imag;
}

static NPY_INLINE complex_int32 complex_int32_add_scalar(complex_int32 c, npy_cdouble s) {
    short int real = s.real + c.real;
    short int imag = s.imag + c.imag;
    return (complex_int32) {real, imag};
}

static NPY_INLINE void complex_int32_inplace_add_scalar(complex_int32* c, npy_cdouble s) {
    short int real = s.real + c->real;
    short int imag = s.real + c->imag;
    c->real = real;
    c->imag = imag;
}

static NPY_INLINE complex_int32 complex_int32_subtract(complex_int32 c1, complex_int32 c2) {
    short int real = c1.real - c2.real;
    short int imag = c1.imag - c2.imag;
    return (complex_int32) {real, imag};
}

static NPY_INLINE void complex_int32_inplace_subtract(complex_int32* c1, complex_int32 c2) {
    short int real = c1->real - c2.real;
    short int imag = c1->imag - c2.imag;
    c1->real = real;
    c1->imag = imag;
}

static NPY_INLINE complex_int32 complex_int32_scalar_subtract(npy_cdouble s, complex_int32 c) {
    short int real = s.real - c.real;
    short int imag = s.imag - c.imag;
    return (complex_int32) {real, imag};
}

static NPY_INLINE void complex_int32_inplace_scalar_subtract(npy_cdouble s, complex_int32* c) {
    short int real = s.real - c->real;
    short int imag = s.real - c->imag;
    c->real = real;
    c->imag = imag;
}

static NPY_INLINE complex_int32 complex_int32_subtract_scalar(complex_int32 c, npy_cdouble s) {
    short int real = -s.real + c.real;
    short int imag = -s.imag + c.imag;
    return (complex_int32) {real, imag};
}

static NPY_INLINE void complex_int32_inplace_subtract_scalar(complex_int32* c, npy_cdouble s) {
    short int real = -s.real + c->real;
    short int imag = -s.real + c->imag;
    c->real = real;
    c->imag = imag;
}

static NPY_INLINE complex_int32 complex_int32_multiply(complex_int32 c1, complex_int32 c2) {
    short int real = c1.real*c2.real - c1.imag*c2.imag;
    short int imag = c1.imag*c2.real + c1.real*c2.imag;
    return (complex_int32) {real, imag};
}

static NPY_INLINE void complex_int32_inplace_multiply(complex_int32* c1, complex_int32 c2) {
    short int real = c1->real*c2.real - c1->imag*c2.imag;
    short int imag = c1->imag*c2.real + c1->real*c2.imag;
    c1->real = real;
    c1->imag = imag;
}

static NPY_INLINE complex_int32 complex_int32_scalar_multiply(npy_cdouble s, complex_int32 c) {
    short int real = s.real*c.real - s.imag*c.imag;
    short int imag = s.imag*c.real + s.real*c.imag;
    return (complex_int32) {real, imag};
}

static NPY_INLINE void complex_int32_inplace_scalar_multiply(npy_cdouble s, complex_int32* c) {
    short int real = s.real*c->real - s.imag*c->imag;
    short int imag = s.imag*c->real + s.real*c->imag;
    c->real = real;
    c->imag = imag;
}

static NPY_INLINE complex_int32 complex_int32_multiply_scalar(complex_int32 c, npy_cdouble s) {
    short int real = s.real*c.real - s.imag*c.imag;
    short int imag = s.imag*c.real + s.real*c.imag;
    return (complex_int32) {real, imag};
}

static NPY_INLINE void complex_int32_inplace_multiply_scalar(complex_int32* c, npy_cdouble s) {
    short int real = s.real*c->real - s.imag*c->imag;
    short int imag = s.imag*c->real + s.real*c->imag;
    c->real = real;
    c->imag = imag;
}

static NPY_INLINE complex_int32 complex_int32_divide(complex_int32 c1, complex_int32 c2) {
    int mag2 = ((int) c2.real)*c2.real + ((int) c1.imag)*c1.imag;
    short int real = (c1.real*c2.real + c1.imag*c2.imag) / mag2;
    short int imag = (c1.imag*c2.real - c1.real*c2.imag) / mag2;
    return (complex_int32) {real, imag};
}

static NPY_INLINE void complex_int32_inplace_divide(complex_int32* c1, complex_int32 c2) {
    int mag2 = ((int) c2.real)*c2.real + ((int) c2.imag)*c2.imag;
    short int real = (c1->real*c2.real + c1->imag*c2.imag) / mag2;
    short int imag = (c1->imag*c2.real - c1->real*c2.imag) / mag2;
    c1->real = real;
    c1->imag = imag;
}

static NPY_INLINE complex_int32 complex_int32_scalar_divide(npy_cdouble s, complex_int32 c) {
    int mag2 = ((int) c.real)*c.real + ((int) c.imag)*c.imag;
    short int real = (s.real*c.real + s.imag*c.imag) / mag2;
    short int imag = (s.real*c.real - s.imag*c.imag) / mag2;
    return (complex_int32) {real, imag};
}

static NPY_INLINE void complex_int32_inplace_scalar_divide(npy_cdouble s, complex_int32* c) {
    int mag2 = ((int) c->real)*c->real + ((int) c->imag)*c->imag;
    short int real = (s.real*c->real + s.imag*c->imag) / mag2;
    short int imag = (s.real*c->real - s.imag*c->imag) / mag2;
    c->real = real;
    c->imag = imag;
}

static NPY_INLINE complex_int32 complex_int32_divide_scalar(complex_int32 c, npy_cdouble s) {
    double mag2 = s.real*s.real + s.imag*s.imag;
    short int real = (c.real*s.real - c.imag*s.imag) / mag2;
    short int imag = (c.imag*s.real + c.real*s.imag) / mag2;
    return (complex_int32) {real, imag};
}

static NPY_INLINE void complex_int32_inplace_divide_scalar(complex_int32* c, npy_cdouble s) {
    double mag2 = s.real*s.real + s.imag*s.imag;
    short int real = (c->real*s.real - c->imag*s.imag) / mag2;
    short int imag = (c->imag*s.real + c->real*s.imag) / mag2;
    c->real = real;
    c->imag = imag;
}

#ifdef __cplusplus
}
#endif

#endif // COMPLEX_COMPLEX_INT32_H_INCLUDE_GUARD_
