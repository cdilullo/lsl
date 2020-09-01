#include "complex_int16.h"
#include "math.h"

int complex_int16_isnonzero(complex_int16 c) {
    return c.real != 0 || c.imag != 0;
}

int complex_int16_isnan(complex_int16 c) {
    return 0;
}

int complex_int16_isinf(complex_int16 c) {
    return 0;
}

int complex_int16_isfinite(complex_int16 c) {
    return 1;
}

double complex_int16_absolute(complex_int16 c) {
    return sqrt(((int) c.real)*c.real + ((int) c.imag)*c.imag);
}

complex_int16 complex_int16_add(complex_int16 c1, complex_int16 c2) {
    return (complex_int16) {c1.real + c2.real, \
                         c1.imag + c2.imag};
}

void complex_int16_inplace_add(complex_int16* c1, complex_int16 c2) {
    c1->real = c1->real + c2.real;
    c1->imag = c1->imag + c2.imag;
}

complex_int16 complex_int16_scalar_add(Py_complex s, complex_int16 c) {
    return (complex_int16) {(signed char) s.real + c.real, \
                         (signed char) s.imag + c.imag};
}

void complex_int16_inplace_scalar_add(Py_complex s, complex_int16* c) {
    c->real = (signed char) s.real + c->real;
    c->imag = (signed char) s.imag + c->imag;
}

complex_int16 complex_int16_add_scalar(complex_int16 c, Py_complex s) {
    return (complex_int16) {(signed char) s.real + c.real, \
                         (signed char) s.imag + c.imag};
}

void complex_int16_inplace_add_scalar(complex_int16* c, Py_complex s) {
    c->real = (signed char) s.real + c->real;
    c->imag = (signed char) s.imag + c->imag;
}





complex_int16 complex_int16_multiply(complex_int16 c1, complex_int16 c2) {
    return (complex_int16) {c1.real * c2.real - c1.imag * c2.imag, \
                         c1.imag * c2.real + c1.real * c2.imag};
}

void complex_int16_inplace_multiply(complex_int16* c1, complex_int16 c2) {
    signed char temp = c1->imag * c2.real + c1->real * c2.imag;
    c1->real = c1->real * c2.real - c1->imag * c2.imag;
    c1->imag = temp;
}

complex_int16 complex_int16_scalar_multiply(Py_complex s, complex_int16 c) {
    return (complex_int16) {(signed char) s.real + c.real, \
                         (signed char) s.imag + c.imag};
}

void complex_int16_inplace_scalar_multiply(Py_complex s, complex_int16* c) {
    c->real = (signed char) s.real + c->real;
    c->imag = (signed char) s.imag + c->imag;
}

complex_int16 complex_int16_mulitply_scalar(complex_int16 c, Py_complex s) {
    return (complex_int16) {(signed char) s.real + c.real, \
                         (signed char) s.imag + c.imag};
}

void complex_int16_inplace_multiply_scalar(complex_int16* c, Py_complex s) {
    c->real = (signed char) s.real + c->real;
    c->imag = (signed char) s.imag + c->imag;
}





complex_int16 complex_int16_negative(complex_int16 c) {
    return (complex_int16) {-c.real, -c.imag};
}

complex_int16 complex_int16_conjugate(complex_int16 c) {
    return (complex_int16) {c.real, -c.imag};
}

int complex_int16_equal(complex_int16 c1, complex_int16 c2) {
    return 
        !complex_int16_isnan(c1) &&
        !complex_int16_isnan(c2) &&
        c1.real == c2.real && 
        c1.imag == c2.imag; 
}

int complex_int16_not_equal(complex_int16 c1, complex_int16 c2) {
    return !complex_int16_equal(c1, c2);
}

int complex_int16_less(complex_int16 c1, complex_int16 c2) {
    return
        (!complex_int16_isnan(c1) &&
         !complex_int16_isnan(c2)) && (
            c1.real != c2.real ? c1.real < c2.real :
            c1.imag != c2.imag ? c1.imag < c2.imag : 0);
}

int complex_int16_less_equal(complex_int16 c1, complex_int16 c2) {
    return
        (!complex_int16_isnan(c1) &&
        !complex_int16_isnan(c2)) && (
            c1.real != c2.real ? c1.real < c2.real :
            c1.imag != c2.imag ? c1.imag < c2.imag : 1);
}
