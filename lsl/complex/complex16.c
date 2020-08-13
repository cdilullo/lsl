#include "complex16.h"
#include "math.h"

int complex16_isnonzero(complex16 c) {
    return c.real != 0 || c.imag != 0;
}

int complex16_isnan(complex16 c) {
    return isnan(c.real) || isnan(c.imag);
}

int complex16_isinf(complex16 c) {
    return isinf(c.real) || isinf(c.imag);
}

int complex16_isfinite(complex16 c) {
    return isfinite(c.real) && isfinite(c.imag);
}

int complex16_absolute(complex16 c) {
    return sqrt(((int) c.real)*c.real + ((int) c.imag)*c.imag);
}

complex16 complex16_negative(complex16 c) {
    return (complex16) {-c.real, -c.imag};
}

complex16 complex16_conjugate(complex16 c) {
    return (complex16) {c.real, -c.imag};
}

complex16 complex16_copysign(complex16 c1, complex16 c2) {
    return (complex16) {
        copysign(c1.real, c2.real),
        copysign(c1.imag, c2.imag)
    };
}

int complex16_equal(complex16 c1, complex16 c2) {
    return 
        !complex16_isnan(c1) &&
        !complex16_isnan(c2) &&
        c1.real == c2.real && 
        c1.imag == c2.imag; 
}

int complex16_not_equal(complex16 c1, complex16 c2) {
    return !complex16_equal(c1, c2);
}

int complex16_less(complex16 c1, complex16 c2) {
    return
        (!complex16_isnan(c1) &&
         !complex16_isnan(c2)) && (
            c1.real != c2.real ? c1.real < c2.real :
            c1.imag != c2.imag ? c1.imag < c2.imag : 0);
}

int complex16_less_equal(complex16 c1, complex16 c2) {
    return
        (!complex16_isnan(c1) &&
        !complex16_isnan(c2)) && (
            c1.real != c2.real ? c1.real < c2.real :
            c1.imag != c2.imag ? c1.imag < c2.imag : 1);
}
