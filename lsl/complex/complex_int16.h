#ifndef __COMPLEX_INT16_H__
#define __COMPLEX_INT16_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	signed char real;
	signed char imag;
} complexi16;

int complexi16_isnonzero(complexi16 c);
int complexi16_isnan(complexi16 c);
int complexi16_isinf(complexi16 c);
int complexi16_isfinite(complexi16 c);
int complexi16_absolute(complexi16 c);
complexi16 complexi16_negative(complexi16 c);
complexi16 complexi16_conjugate(complexi16 c);
int complexi16_equal(complexi16 c1, complexi16 c2);
int complexi16_not_equal(complexi16 c1, complexi16 c2);
int complexi16_less(complexi16 c1, complexi16 c2);
int complexi16_less_equal(complexi16 c1, complexi16 c2);

#ifdef __cplusplus
}
#endif

#endif
