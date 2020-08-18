#ifndef COMPLEX_COMPLEX_INT_H_INCLUDE_GUARD_
#define COMPLEX_COMPLEX_INT_H_INCLUDE_GUARD_

#ifdef __cplusplus
extern "C" {
#endif

#include "complex_int8.h"

#define LSL_MAX_COMPLEX_DTYPES 3

#define LSL_COMPLEX_INT8 55
#define LSL_COMPLEX_INT16 56
#define LSL_COMPLEX_INT32 57

extern int lsl_complex_dtypes[LSL_MAX_COMPLEX_DTYPES];

void lsl_register_complex_int(int bit_depth, int type_num);
int lsl_get_complex_int(int bit_depth);
void lsl_unpack_ci8(complexi8 packed, signed char* real, signed char* imag);

#ifdef __cplusplus
}
#endif

#endif  //COMPLEX_COMPLEX_INT_H_INCLUDE_GUARD_


