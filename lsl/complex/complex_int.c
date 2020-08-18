
#include "complex_int.h"
#include "complex_int8.h"

int lsl_complex_dtypes[MAX_COMPLEX_DTYPES] = {-1};

void lsl_register_complex_int(int bit_depth, int type_num) {
    // TODO:  Make this more robust (or not)
    lsl_complex_dtypes[bit_depth / 16] = type_num;
}

int lsl_get_complex_int(int bit_depth) {
    // TODO:  Make this more robust (or not)
    return lsl_complex_dtypes[bit_depth / 16];
}

void lsl_unpack_ci8(complexi8, signed char* real, signed char* imag) {
    const signed char* sc = fourBitLUT[c.real_imag];
    *real = sc[0];
    *imag = sc[1];
}
