#include "complex_int16.h"
#include "math.h"

int complexi16_isnonzero(complexi16 c) {
    return c.real != 0 || c.imag != 0;
}

int complexi16_isnan(complexi16 c) {
    return 0;
}

int complexi16_isinf(complexi16 c) {
    return 0;
}

int complexi16_isfinite(complexi16 c) {
    return 1;
}

double complexi16_absolute(complexi16 c) {
    return sqrt(((int) c.real)*c.real + ((int) c.imag)*c.imag);
}

complexi16 complexi16_negative(complexi16 c) {
    return (complexi16) {-c.real, -c.imag};
}

complexi16 complexi16_conjugate(complexi16 c) {
    return (complexi16) {c.real, -c.imag};
}

int complexi16_equal(complexi16 c1, complexi16 c2) {
    return 
        !complexi16_isnan(c1) &&
        !complexi16_isnan(c2) &&
        c1.real == c2.real && 
        c1.imag == c2.imag; 
}

int complexi16_not_equal(complexi16 c1, complexi16 c2) {
    return !complexi16_equal(c1, c2);
}

int complexi16_less(complexi16 c1, complexi16 c2) {
    return
        (!complexi16_isnan(c1) &&
         !complexi16_isnan(c2)) && (
            c1.real != c2.real ? c1.real < c2.real :
            c1.imag != c2.imag ? c1.imag < c2.imag : 0);
}

int complexi16_less_equal(complexi16 c1, complexi16 c2) {
    return
        (!complexi16_isnan(c1) &&
        !complexi16_isnan(c2)) && (
            c1.real != c2.real ? c1.real < c2.real :
            c1.imag != c2.imag ? c1.imag < c2.imag : 1);
}
