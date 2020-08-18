
#include "complex_int.h"
#include "complex_int8.h"

void lsl_unpack_ci8(complexi8 packed, signed char* real, signed char* imag) {
    const signed char* sc = fourBitLUT[packed.real_imag];
    *real = sc[0];
    *imag = sc[1];
}
