#ifndef __COMPLEX8_H__
#define __COMPLEX8_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	unsigned char real_imag;
} complex8;


extern signed char fourBitLUT[256][2];
void complex8_fillLUT();

int complex8_isnonzero(complex8 c);
int complex8_isnan(complex8 c);
int complex8_isinf(complex8 c);
int complex8_isfinite(complex8 c);
int complex8_absolute(complex8 c);
complex8 complex8_negative(complex8 c);
complex8 complex8_conjugate(complex8 c);
complex8 complex8_copysign(complex8 c1, complex8 c2);
int complex8_equal(complex8 c1, complex8 c2);
int complex8_not_equal(complex8 c1, complex8 c2);
int complex8_less(complex8 c1, complex8 c2);
int complex8_less_equal(complex8 c1, complex8 c2);

#ifdef __cplusplus
}
#endif

#endif
