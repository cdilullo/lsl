
#include "complex_int.h"
#include "complex_int8.h"
#include "complex_int16.h"
#include "complex_int32.h"

void lsl_unpack_ci8(complex_int8 packed, signed char* real, signed char* imag) {
    const signed char* sc = fourBitLUT[packed.real_imag];
    *real = sc[0];
    *imag = sc[1];
}

void lsl_unpack_ci16(complex_int16 packed, signed char* real, signed char* imag) {
    *real = packed.real;
    *imag = packed.imag;
}

void lsl_unpack_ci32(complex_int32 packed, short int* real, short int* imag) {
    *real = packed.real;
    *imag = packed.imag;
}
