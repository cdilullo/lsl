#include "complex_int32.h"
#include "math.h"

int complexi32_isnonzero(complexi32 c) {
    return c.real != 0 || c.imag != 0;
}

int complexi32_isnan(complexi32 c) {
    return 0;
}

int complexi32_isinf(complexi32 c) {
    return 0;
}

int complexi32_isfinite(complexi32 c) {
    return 1;
}

double complexi32_absolute(complexi32 c) {
    return sqrt(((int) c.real)*c.real + ((int) c.imag)*c.imag);
}

complexi32 complexi32_negative(complexi32 c) {
    return (complexi32) {-c.real, -c.imag};
}

complexi32 complexi32_conjugate(complexi32 c) {
    return (complexi32) {c.real, -c.imag};
}

int complexi32_equal(complexi32 c1, complexi32 c2) {
    return 
        !complexi32_isnan(c1) &&
        !complexi32_isnan(c2) &&
        c1.real == c2.real && 
        c1.imag == c2.imag; 
}

int complexi32_not_equal(complexi32 c1, complexi32 c2) {
    return !complexi32_equal(c1, c2);
}

int complexi32_less(complexi32 c1, complexi32 c2) {
    return
        (!complexi32_isnan(c1) &&
         !complexi32_isnan(c2)) && (
            c1.real != c2.real ? c1.real < c2.real :
            c1.imag != c2.imag ? c1.imag < c2.imag : 0);
}

int complexi32_less_equal(complexi32 c1, complexi32 c2) {
    return
        (!complexi32_isnan(c1) &&
        !complexi32_isnan(c2)) && (
            c1.real != c2.real ? c1.real < c2.real :
            c1.imag != c2.imag ? c1.imag < c2.imag : 1);
}
