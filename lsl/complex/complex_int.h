#ifndef __COMPLEX_INT_H__
#define __COMPLEX_INT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "complex_int8.h"

#define MAX_COMPLEX_DTYPES 3

extern int lsl_complex_dtypes[MAX_COMPLEX_DTYPES];

void lsl_register_complex_int(int bit_depth, int type_num);
int lsl_get_complex_int(int bit_depth);
void lsl_unpack_ci8(complexi8, signed char* real, signed char* imag);

#ifdef __cplusplus
}
#endif

#endif


