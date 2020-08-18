#ifndef __COMPLEX_INT32_H__
#define __COMPLEX_INT32_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	short int real;
	short int imag;
} complexi32;

int complexi32_isnonzero(complexi32 c);
int complexi32_isnan(complexi32 c);
int complexi32_isinf(complexi32 c);
int complexi32_isfinite(complexi32 c);
int complexi32_absolute(complexi32 c);
complexi32 complexi32_negative(complexi32 c);
complexi32 complexi32_conjugate(complexi32 c);
int complexi32_equal(complexi32 c1, complexi32 c2);
int complexi32_not_equal(complexi32 c1, complexi32 c2);
int complexi32_less(complexi32 c1, complexi32 c2);
int complexi32_less_equal(complexi32 c1, complexi32 c2);

#ifdef __cplusplus
}
#endif

#endif
