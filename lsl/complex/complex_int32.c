#include "complex_int32.h"
#include "math.h"

int complex_int32_isnonzero(complex_int32 c) {
    return c.real != 0 || c.imag != 0;
}

int complex_int32_isnan(complex_int32 c) {
    return 0;
}

int complex_int32_isinf(complex_int32 c) {
    return 0;
}

int complex_int32_isfinite(complex_int32 c) {
    return 1;
}

double complex_int32_absolute(complex_int32 c) {
    return sqrt(((int) c.real)*c.real + ((int) c.imag)*c.imag);
}

complex_int32 complex_int32_negative(complex_int32 c) {
    return (complex_int32) {-c.real, -c.imag};
}

complex_int32 complex_int32_conjugate(complex_int32 c) {
    return (complex_int32) {c.real, -c.imag};
}

int complex_int32_equal(complex_int32 c1, complex_int32 c2) {
    return 
        !complex_int32_isnan(c1) &&
        !complex_int32_isnan(c2) &&
        c1.real == c2.real && 
        c1.imag == c2.imag; 
}

int complex_int32_not_equal(complex_int32 c1, complex_int32 c2) {
    return !complex_int32_equal(c1, c2);
}

int complex_int32_less(complex_int32 c1, complex_int32 c2) {
    return
        (!complex_int32_isnan(c1) &&
         !complex_int32_isnan(c2)) && (
            c1.real != c2.real ? c1.real < c2.real :
            c1.imag != c2.imag ? c1.imag < c2.imag : 0);
}

int complex_int32_less_equal(complex_int32 c1, complex_int32 c2) {
    return
        (!complex_int32_isnan(c1) &&
        !complex_int32_isnan(c2)) && (
            c1.real != c2.real ? c1.real < c2.real :
            c1.imag != c2.imag ? c1.imag < c2.imag : 1);
}
