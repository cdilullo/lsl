#ifndef COMPLEX_COMPLEX_INT16_H_INCLUDE_GUARD_
#define COMPLEX_COMPLEX_INT16_H_INCLUDE_GUARD_

#ifdef __cplusplus
extern "C" {
#endif

#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

typedef struct {
    signed char real;
    signed char imag;
} complex_int16;

// Unary bool operators
static NPY_INLINE int complex_int16_nonzero(complex_int16 c) {
    return c.real != 0 || c.imag != 0;
}

static NPY_INLINE int complex_int16_isnan(complex_int16 c) {
    return 0;
}

static NPY_INLINE int complex_int16_isinf(complex_int16 c) {
    return 0;
}

static NPY_INLINE int complex_int16_isfinite(complex_int16 c) {
    return 1;
}

// Binary bool operators
static NPY_INLINE int complex_int16_equal(complex_int16 c1, complex_int16 c2) {
    return 
        !complex_int16_isnan(c1) &&
        !complex_int16_isnan(c2) &&
        c1.real == c2.real &&
        c1.imag == c2.imag; 
}

static NPY_INLINE int complex_int16_not_equal(complex_int16 c1, complex_int16 c2) {
    return !complex_int16_equal(c1, c2);
}

static NPY_INLINE int complex_int16_less(complex_int16 c1, complex_int16 c2) {
    return
        (!complex_int16_isnan(c1) &&
         !complex_int16_isnan(c2)) && (
            c1.real != c2.real ? c1.real < c2.real :
            c1.imag != c2.imag ? c1.imag < c2.imag : 0);
}

static NPY_INLINE int complex_int16_greater(complex_int16 c1, complex_int16 c2) {
    return
        (!complex_int16_isnan(c1) &&
         !complex_int16_isnan(c2)) && (
            c1.real != c2.real ? c1.real > c2.real :
            c1.imag != c2.imag ? c1.imag > c2.imag : 0);
}

static NPY_INLINE int complex_int16_less_equal(complex_int16 c1, complex_int16 c2) {
    return
        (!complex_int16_isnan(c1) &&
         !complex_int16_isnan(c2)) && (
            c1.real != c2.real ? c1.real < c2.real :
            c1.imag != c2.imag ? c1.imag < c2.imag : 1);
}

static NPY_INLINE int complex_int16_greater_equal(complex_int16 c1, complex_int16 c2) {
    return
        (!complex_int16_isnan(c1) &&
         !complex_int16_isnan(c2)) && (
            c1.real != c2.real ? c1.real > c2.real :
            c1.imag != c2.imag ? c1.imag > c2.imag : 1);
}

// Unary float returners
static NPY_INLINE double complex_int16_norm(complex_int16 c) {
    return (((int) c.real)*c.real + ((int) c.imag)*c.imag)*1.0;
}

static NPY_INLINE double complex_int16_absolute(complex_int16 c) {
    return sqrt(((int) c.real)*c.real + ((int) c.imag)*c.imag);
}

static NPY_INLINE double complex_int16_angle(complex_int16 c) {
    return atan2((int) c.imag, (int) c.real);
}

// Unary complex_int16 returners
static NPY_INLINE complex_int16 complex_int16_negative(complex_int16 c) {
    signed char real = -c.real;
    signed char imag = -c.imag;
    return (complex_int16) {real, imag};
}

static NPY_INLINE complex_int16 complex_int16_conjugate(complex_int16 c) {
    signed char real =  c.real;
    signed char imag = -c.imag;
    return (complex_int16) {real, imag};
}

// complex_int16-complex_int16/complex_int16-scalar/scalar-complex_int16 binary complex_int16 returners
static NPY_INLINE complex_int16 complex_int16_add(complex_int16 c1, complex_int16 c2) {
    signed char real = c1.real + c2.real;
    signed char imag = c1.imag + c2.imag;
    return (complex_int16) {real, imag};
}

static NPY_INLINE void complex_int16_inplace_add(complex_int16* c1, complex_int16 c2) {
    signed char real = c1->real + c2.real;
    signed char imag = c1->imag + c2.imag;
    c1->real = real;
    c1->imag = imag;
}

static NPY_INLINE complex_int16 complex_int16_scalar_add(npy_cdouble s, complex_int16 c) {
    signed char real = s.real + c.real;
    signed char imag = s.imag + c.imag;
    return (complex_int16) {real, imag};
}

static NPY_INLINE void complex_int16_inplace_scalar_add(npy_cdouble s, complex_int16* c) {
    signed char real = s.real + c->real;
    signed char imag = s.real + c->imag;
    c->real = real;
    c->imag = imag;
}

static NPY_INLINE complex_int16 complex_int16_add_scalar(complex_int16 c, npy_cdouble s) {
    signed char real = s.real + c.real;
    signed char imag = s.imag + c.imag;
    return (complex_int16) {real, imag};
}

static NPY_INLINE void complex_int16_inplace_add_scalar(complex_int16* c, npy_cdouble s) {
    signed char real = s.real + c->real;
    signed char imag = s.real + c->imag;
    c->real = real;
    c->imag = imag;
}

static NPY_INLINE complex_int16 complex_int16_subtract(complex_int16 c1, complex_int16 c2) {
    signed char real = c1.real - c2.real;
    signed char imag = c1.imag - c2.imag;
    return (complex_int16) {real, imag};
}

static NPY_INLINE void complex_int16_inplace_subtract(complex_int16* c1, complex_int16 c2) {
    signed char real = c1->real - c2.real;
    signed char imag = c1->imag - c2.imag;
    c1->real = real;
    c1->imag = imag;
}

static NPY_INLINE complex_int16 complex_int16_scalar_subtract(npy_cdouble s, complex_int16 c) {
    signed char real = s.real - c.real;
    signed char imag = s.imag - c.imag;
    return (complex_int16) {real, imag};
}

static NPY_INLINE void complex_int16_inplace_scalar_subtract(npy_cdouble s, complex_int16* c) {
    signed char real = s.real - c->real;
    signed char imag = s.real - c->imag;
    c->real = real;
    c->imag = imag;
}

static NPY_INLINE complex_int16 complex_int16_subtract_scalar(complex_int16 c, npy_cdouble s) {
    signed char real = -s.real + c.real;
    signed char imag = -s.imag + c.imag;
    return (complex_int16) {real, imag};
}

static NPY_INLINE void complex_int16_inplace_subtract_scalar(complex_int16* c, npy_cdouble s) {
    signed char real = -s.real + c->real;
    signed char imag = -s.real + c->imag;
    c->real = real;
    c->imag = imag;
}

static NPY_INLINE complex_int16 complex_int16_multiply(complex_int16 c1, complex_int16 c2) {
    signed char real = c1.real*c2.real - c1.imag*c2.imag;
    signed char imag = c1.imag*c2.real + c1.real*c2.imag;
    return (complex_int16) {real, imag};
}

static NPY_INLINE void complex_int16_inplace_multiply(complex_int16* c1, complex_int16 c2) {
    signed char real = c1->real*c2.real - c1->imag*c2.imag;
    signed char imag = c1->imag*c2.real + c1->real*c2.imag;
    c1->real = real;
    c1->imag = imag;
}

static NPY_INLINE complex_int16 complex_int16_scalar_multiply(npy_cdouble s, complex_int16 c) {
    signed char real = s.real*c.real - s.imag*c.imag;
    signed char imag = s.imag*c.real + s.real*c.imag;
    return (complex_int16) {real, imag};
}

static NPY_INLINE void complex_int16_inplace_scalar_multiply(npy_cdouble s, complex_int16* c) {
    signed char real = s.real*c->real - s.imag*c->imag;
    signed char imag = s.imag*c->real + s.real*c->imag;
    c->real = real;
    c->imag = imag;
}

static NPY_INLINE complex_int16 complex_int16_multiply_scalar(complex_int16 c, npy_cdouble s) {
    signed char real = s.real*c.real - s.imag*c.imag;
    signed char imag = s.imag*c.real + s.real*c.imag;
    return (complex_int16) {real, imag};
}

static NPY_INLINE void complex_int16_inplace_multiply_scalar(complex_int16* c, npy_cdouble s) {
    signed char real = s.real*c->real - s.imag*c->imag;
    signed char imag = s.imag*c->real + s.real*c->imag;
    c->real = real;
    c->imag = imag;
}

static NPY_INLINE complex_int16 complex_int16_divide(complex_int16 c1, complex_int16 c2) {
    int mag2 = ((int) c2.real)*c2.real + ((int) c1.imag)*c1.imag;
    signed char real = (c1.real*c2.real + c1.imag*c2.imag) / mag2;
    signed char imag = (c1.imag*c2.real - c1.real*c2.imag) / mag2;
    return (complex_int16) {real, imag};
}

static NPY_INLINE void complex_int16_inplace_divide(complex_int16* c1, complex_int16 c2) {
    int mag2 = ((int) c2.real)*c2.real + ((int) c2.imag)*c2.imag;
    signed char real = (c1->real*c2.real + c1->imag*c2.imag) / mag2;
    signed char imag = (c1->imag*c2.real - c1->real*c2.imag) / mag2;
    c1->real = real;
    c1->imag = imag;
}

static NPY_INLINE complex_int16 complex_int16_scalar_divide(npy_cdouble s, complex_int16 c) {
    int mag2 = ((int) c.real)*c.real + ((int) c.imag)*c.imag;
    signed char real = (s.real*c.real + s.imag*c.imag) / mag2;
    signed char imag = (s.real*c.real - s.imag*c.imag) / mag2;
    return (complex_int16) {real, imag};
}

static NPY_INLINE void complex_int16_inplace_scalar_divide(npy_cdouble s, complex_int16* c) {
    int mag2 = ((int) c->real)*c->real + ((int) c->imag)*c->imag;
    signed char real = (s.real*c->real + s.imag*c->imag) / mag2;
    signed char imag = (s.real*c->real - s.imag*c->imag) / mag2;
    c->real = real;
    c->imag = imag;
}

static NPY_INLINE complex_int16 complex_int16_divide_scalar(complex_int16 c, npy_cdouble s) {
    double mag2 = s.real*s.real + s.imag*s.imag;
    signed char real = (c.real*s.real - c.imag*s.imag) / mag2;
    signed char imag = (c.imag*s.real + c.real*s.imag) / mag2;
    return (complex_int16) {real, imag};
}

static NPY_INLINE void complex_int16_inplace_divide_scalar(complex_int16* c, npy_cdouble s) {
    double mag2 = s.real*s.real + s.imag*s.imag;
    signed char real = (c->real*s.real - c->imag*s.imag) / mag2;
    signed char imag = (c->imag*s.real + c->real*s.imag) / mag2;
    c->real = real;
    c->imag = imag;
}

#ifdef __cplusplus
}
#endif

#endif // COMPLEX_COMPLEX_INT16_H_INCLUDE_GUARD_
