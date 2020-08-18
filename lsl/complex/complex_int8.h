#ifndef COMPLEX_COMPLEX_INT8_H_INCLUDE_GUARD_
#define COMPLEX_COMPLEX_INT8_H_INCLUDE_GUARD_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	unsigned char real_imag;
} complexi8;


extern signed char fourBitLUT[256][2];
void complexi8_fillLUT();

int complexi8_isnonzero(complexi8 c);
int complexi8_isnan(complexi8 c);
int complexi8_isinf(complexi8 c);
int complexi8_isfinite(complexi8 c);
double complexi8_absolute(complexi8 c);
complexi8 complexi8_negative(complexi8 c);
complexi8 complexi8_conjugate(complexi8 c);
int complexi8_equal(complexi8 c1, complexi8 c2);
int complexi8_not_equal(complexi8 c1, complexi8 c2);
int complexi8_less(complexi8 c1, complexi8 c2);
int complexi8_less_equal(complexi8 c1, complexi8 c2);

#ifdef __cplusplus
}
#endif

#endif // COMPLEX_COMPLEX_INT8_H_INCLUDE_GUARD_
