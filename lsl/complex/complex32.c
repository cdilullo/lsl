#include "complex32.h"
#include "math.h"

int complex32_isnonzero(complex32 c) {
    return c.real != 0 || c.imag != 0;
}

int complex32_isnan(complex32 c) {
    return isnan(c.real) || isnan(c.imag);
}

int complex32_isinf(complex32 c) {
    return isinf(c.real) || isinf(c.imag);
}

int complex32_isfinite(complex32 c) {
    return isfinite(c.real) && isfinite(c.imag);
}

int complex32_absolute(complex32 c) {
    return sqrt(((int) c.real)*c.real + ((int) c.imag)*c.imag);
}

complex32 complex32_negative(complex32 c) {
    return (complex32) {-c.real, -c.imag};
}

complex32 complex32_conjugate(complex32 c) {
    return (complex32) {c.real, -c.imag};
}

complex32 complex32_copysign(complex32 c1, complex32 c2) {
    return (complex32) {
        copysign(c1.real, c2.real),
        copysign(c1.imag, c2.imag)
    };
}

int complex32_equal(complex32 c1, complex32 c2) {
    return 
        !complex32_isnan(c1) &&
        !complex32_isnan(c2) &&
        c1.real == c2.real && 
        c1.imag == c2.imag; 
}

int complex32_not_equal(complex32 c1, complex32 c2) {
    return !complex32_equal(c1, c2);
}

int complex32_less(complex32 c1, complex32 c2) {
    return
        (!complex32_isnan(c1) &&
         !complex32_isnan(c2)) && (
            c1.real != c2.real ? c1.real < c2.real :
            c1.imag != c2.imag ? c1.imag < c2.imag : 0);
}

int complex32_less_equal(complex32 c1, complex32 c2) {
    return
        (!complex32_isnan(c1) &&
        !complex32_isnan(c2)) && (
            c1.real != c2.real ? c1.real < c2.real :
            c1.imag != c2.imag ? c1.imag < c2.imag : 1);
}
