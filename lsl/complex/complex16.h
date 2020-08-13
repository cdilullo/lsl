#ifndef __COMPLEX16_H__
#define __COMPLEX16_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	signed char real;
	signed char imag;
} complex16;

int complex16_isnonzero(complex16 c);
int complex16_isnan(complex16 c);
int complex16_isinf(complex16 c);
int complex16_isfinite(complex16 c);
int complex16_absolute(complex16 c);
complex16 complex16_negative(complex16 c);
complex16 complex16_conjugate(complex16 c);
complex16 complex16_copysign(complex16 c1, complex16 c2);
int complex16_equal(complex16 c1, complex16 c2);
int complex16_not_equal(complex16 c1, complex16 c2);
int complex16_less(complex16 c1, complex16 c2);
int complex16_less_equal(complex16 c1, complex16 c2);

#ifdef __cplusplus
}
#endif

#endif
