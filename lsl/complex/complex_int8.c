#include "complex_int8.h"

signed char fourBitLUT[256][2];
void complex_int8_fillLUT() {
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

