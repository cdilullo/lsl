#include "complex8.h"
#include "math.h"

signed char fourBitLUT[256][2];
void complex8_fillLUT() {
    int i, j;
    signed char t;
    
    for(i=0; i<256; i++) {
        for(j=0; j<2; j++) {
            t = (i >> 4*(1-j)) & 15;
            fourBitLUT[(unsigned char) i][j] = t;
            fourBitLUT[(unsigned char) i][j] -= ((t&8)<<1);
        }
    }
}

int complex8_isnonzero(complex8 c) {
    const signed char* sc = fourBitLUT[c.real_imag];
    return sc[0] != 0 || sc[1] != 0;
}

int complex8_isnan(complex8 c) {
    const signed char* sc = fourBitLUT[c.real_imag];
    return isnan(sc[0]) || isnan(sc[1]);
}

int complex8_isinf(complex8 c) {
    const signed char* sc = fourBitLUT[c.real_imag];
    return isinf(sc[0]) || isinf(sc[1]);
}

int complex8_isfinite(complex8 c) {
    const signed char* sc = fourBitLUT[c.real_imag];
    return isfinite(sc[0]) && isfinite(sc[1]);
}

int complex8_absolute(complex8 c) {
    const signed char* sc = fourBitLUT[c.real_imag];
    return sqrt(((int) sc[0])*sc[0] + ((int) sc[1])*sc[1]);
}

complex8 complex8_negative(complex8 c) {
    signed char real = -(c.real_imag & 0xF0);
    signed char imag = -(c.real_imag << 4);
    return (complex8) {real | (imag >> 4)};
}

complex8 complex8_conjugate(complex8 c) {
    signed char real = (c.real_imag & 0xF0);
    signed char imag = -(c.real_imag << 4);
    return (complex8) {real | (imag >> 4)};
}

complex8 complex8_copysign(complex8 c1, complex8 c2) {
    signed char real1 = (c1.real_imag & 0xF0);
    signed char imag1 = (c1.real_imag << 4);
    signed char real2 = (c2.real_imag & 0xF0);
    signed char imag2 = (c2.real_imag << 4);
    copysign(real1, real2);
    copysign(imag1, imag2);
    return (complex8) {real1 | (imag1 >> 4)};
}

int complex8_equal(complex8 c1, complex8 c2) {
    return 
        !complex8_isnan(c1) &&
        !complex8_isnan(c2) &&
        c1.real_imag == c2.real_imag; 
}

int complex8_not_equal(complex8 c1, complex8 c2) {
    return !complex8_equal(c1, c2);
}

int complex8_less(complex8 c1, complex8 c2) {
    const signed char* sc1 = fourBitLUT[c1.real_imag];
    const signed char* sc2 = fourBitLUT[c2.real_imag];
    return
        (!complex8_isnan(c1) &&
         !complex8_isnan(c2)) && (
            sc1[0] != sc2[0] ? sc1[0] < sc2[0] :
            sc1[1] != sc2[1] ? sc1[1] < sc2[1] : 0);
}

int complex8_less_equal(complex8 c1, complex8 c2) {
    const signed char* sc1 = fourBitLUT[c1.real_imag];
    const signed char* sc2 = fourBitLUT[c2.real_imag];
    return
        (!complex8_isnan(c1) &&
         !complex8_isnan(c2)) && (
            sc1[0] != sc2[0] ? sc1[0] < sc2[0] :
            sc1[1] != sc2[1] ? sc1[1] < sc2[1] : 1);
}
