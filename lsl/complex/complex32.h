#ifndef __complex32_H__
#define __complex32_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	short int real;
	short int imag;
} complex32;

int complex32_isnonzero(complex32 c);
int complex32_isnan(complex32 c);
int complex32_isinf(complex32 c);
int complex32_isfinite(complex32 c);
int complex32_absolute(complex32 c);
complex32 complex32_negative(complex32 c);
complex32 complex32_conjugate(complex32 c);
complex32 complex32_copysign(complex32 c1, complex32 c2);
int complex32_equal(complex32 c1, complex32 c2);
int complex32_not_equal(complex32 c1, complex32 c2);
int complex32_less(complex32 c1, complex32 c2);
int complex32_less_equal(complex32 c1, complex32 c2);

#ifdef __cplusplus
}
#endif

#endif
