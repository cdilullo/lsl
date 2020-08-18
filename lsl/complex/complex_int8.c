#include "complex_int8.h"
#include "math.h"

signed char fourBitLUT[256][2];
void complexi8_fillLUT() {
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

int complexi8_isnonzero(complexi8 c) {
    const signed char* sc = fourBitLUT[c.real_imag];
    return sc[0] != 0 || sc[1] != 0;
}

int complexi8_isnan(complexi8 c) {
    const signed char* sc = fourBitLUT[c.real_imag];
    return 0;
}

int complexi8_isinf(complexi8 c) {
    const signed char* sc = fourBitLUT[c.real_imag];
    return 0;
}

int complexi8_isfinite(complexi8 c) {
    const signed char* sc = fourBitLUT[c.real_imag];
    return 1;
}

double complexi8_absolute(complexi8 c) {
    const signed char* sc = fourBitLUT[c.real_imag];
    return sqrt(((int) sc[0])*sc[0] + ((int) sc[1])*sc[1]);
}

complexi8 complexi8_negative(complexi8 c) {
    const signed char* sc = fourBitLUT[c.real_imag];
    signed char real = -sc[0];
    signed char imag = -sc[1];
    return (complexi8) {((unsigned char) real << 4) | ((unsigned char) imag)};
}

complexi8 complexi8_conjugate(complexi8 c) {
    const signed char* sc = fourBitLUT[c.real_imag];
    signed char real = sc[0];
    signed char imag = -sc[1];
    return (complexi8) {((unsigned char) real << 4) | ((unsigned char) imag)};
}

int complexi8_equal(complexi8 c1, complexi8 c2) {
    return 
        !complexi8_isnan(c1) &&
        !complexi8_isnan(c2) &&
        c1.real_imag == c2.real_imag; 
}

int complexi8_not_equal(complexi8 c1, complexi8 c2) {
    return !complexi8_equal(c1, c2);
}

int complexi8_less(complexi8 c1, complexi8 c2) {
    const signed char* sc1 = fourBitLUT[c1.real_imag];
    const signed char* sc2 = fourBitLUT[c2.real_imag];
    return
        (!complexi8_isnan(c1) &&
         !complexi8_isnan(c2)) && (
            sc1[0] != sc2[0] ? sc1[0] < sc2[0] :
            sc1[1] != sc2[1] ? sc1[1] < sc2[1] : 0);
}

int complexi8_less_equal(complexi8 c1, complexi8 c2) {
    const signed char* sc1 = fourBitLUT[c1.real_imag];
    const signed char* sc2 = fourBitLUT[c2.real_imag];
    return
        (!complexi8_isnan(c1) &&
         !complexi8_isnan(c2)) && (
            sc1[0] != sc2[0] ? sc1[0] < sc2[0] :
            sc1[1] != sc2[1] ? sc1[1] < sc2[1] : 1);
}
