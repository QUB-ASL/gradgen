/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) casadi_functions_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_f1 CASADI_PREFIX(f1)
#define casadi_f2 CASADI_PREFIX(f2)
#define casadi_f3 CASADI_PREFIX(f3)
#define casadi_f4 CASADI_PREFIX(f4)
#define casadi_f5 CASADI_PREFIX(f5)
#define casadi_f6 CASADI_PREFIX(f6)
#define casadi_f7 CASADI_PREFIX(f7)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[14] = {10, 1, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
static const casadi_int casadi_s1[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s3[7] = {10, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s4[4] = {3, 1, 0, 0};

/* casadi_quadcopter_test_f:(x[10],u[3])->(f[10]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=4.0000000000000001e-03;
  a1=arg[0]? arg[0][4] : 0;
  a2=casadi_sq(a1);
  a3=arg[0]? arg[0][5] : 0;
  a4=casadi_sq(a3);
  a2=(a2+a4);
  a4=arg[0]? arg[0][6] : 0;
  a5=casadi_sq(a4);
  a2=(a2+a5);
  a2=sqrt(a2);
  a5=cosh(a2);
  a5=(a0*a5);
  a6=arg[0]? arg[0][0] : 0;
  a5=(a5*a6);
  a7=(a1*a2);
  a7=(a0*a7);
  a8=arg[0]? arg[0][1] : 0;
  a7=(a7*a8);
  a5=(a5-a7);
  a7=(a3*a2);
  a7=(a0*a7);
  a9=arg[0]? arg[0][2] : 0;
  a7=(a7*a9);
  a5=(a5-a7);
  a7=(a4*a2);
  a7=(a0*a7);
  a10=arg[0]? arg[0][3] : 0;
  a7=(a7*a10);
  a5=(a5-a7);
  if (res[0]!=0) res[0][0]=a5;
  a5=(a1*a2);
  a5=(a0*a5);
  a5=(a5*a6);
  a7=cosh(a2);
  a7=(a0*a7);
  a7=(a7*a8);
  a5=(a5+a7);
  a7=(a4*a2);
  a7=(a0*a7);
  a7=(a7*a9);
  a5=(a5+a7);
  a7=(a3*a2);
  a7=(a0*a7);
  a7=(a7*a10);
  a5=(a5+a7);
  if (res[0]!=0) res[0][1]=a5;
  a5=(a3*a2);
  a5=(a0*a5);
  a5=(a5*a6);
  a7=(a4*a2);
  a7=(a0*a7);
  a7=(a7*a8);
  a5=(a5-a7);
  a7=cosh(a2);
  a7=(a0*a7);
  a7=(a7*a9);
  a5=(a5+a7);
  a7=(a1*a2);
  a7=(a0*a7);
  a7=(a7*a10);
  a5=(a5+a7);
  if (res[0]!=0) res[0][2]=a5;
  a5=(a4*a2);
  a5=(a0*a5);
  a5=(a5*a6);
  a6=(a3*a2);
  a6=(a0*a6);
  a6=(a6*a8);
  a5=(a5+a6);
  a6=(a1*a2);
  a6=(a0*a6);
  a6=(a6*a9);
  a5=(a5-a6);
  a2=cosh(a2);
  a0=(a0*a2);
  a0=(a0*a10);
  a5=(a5+a0);
  if (res[0]!=0) res[0][3]=a5;
  a5=8.0000000000000002e-03;
  a0=2.2874690000000002e+00;
  a10=arg[0]? arg[0][7] : 0;
  a0=(a0*a10);
  a2=5.5928411633109619e+01;
  a6=4.6140000000000000e-02;
  a6=(a6*a4);
  a9=(a3*a6);
  a8=3.0140000000000000e-02;
  a8=(a8*a3);
  a7=(a4*a8);
  a9=(a9-a7);
  a2=(a2*a9);
  a0=(a0-a2);
  a0=(a5*a0);
  a0=(a1+a0);
  if (res[0]!=0) res[0][4]=a0;
  a0=1.3569988600000000e+00;
  a2=arg[0]? arg[0][8] : 0;
  a0=(a0*a2);
  a9=3.3178500331785003e+01;
  a7=1.7880000000000000e-02;
  a7=(a7*a1);
  a11=(a4*a7);
  a6=(a1*a6);
  a11=(a11-a6);
  a9=(a9*a11);
  a0=(a0-a9);
  a0=(a5*a0);
  a0=(a3+a0);
  if (res[0]!=0) res[0][5]=a0;
  a0=-4.2388942000000002e-01;
  a9=arg[0]? arg[0][9] : 0;
  a0=(a0*a9);
  a11=1.2465900000000001e+02;
  a6=arg[1]? arg[1][2] : 0;
  a11=(a11*a6);
  a0=(a0+a11);
  a11=2.1673168617251843e+01;
  a1=(a1*a8);
  a3=(a3*a7);
  a1=(a1-a3);
  a11=(a11*a1);
  a0=(a0-a11);
  a0=(a5*a0);
  a4=(a4+a0);
  if (res[0]!=0) res[0][6]=a4;
  a4=5040.;
  a0=arg[1]? arg[1][0] : 0;
  a0=(a4*a0);
  a11=20.;
  a1=(a11*a10);
  a0=(a0-a1);
  a0=(a5*a0);
  a10=(a10+a0);
  if (res[0]!=0) res[0][7]=a10;
  a10=arg[1]? arg[1][1] : 0;
  a10=(a4*a10);
  a0=(a11*a2);
  a10=(a10-a0);
  a10=(a5*a10);
  a2=(a2+a10);
  if (res[0]!=0) res[0][8]=a2;
  a4=(a4*a6);
  a11=(a11*a9);
  a4=(a4-a11);
  a5=(a5*a4);
  a9=(a9+a5);
  if (res[0]!=0) res[0][9]=a9;
  return 0;
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_f(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_f_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_f_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_f_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_f_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_f_release(int mem) {
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_f_incref(void) {
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_f_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int casadi_quadcopter_test_f_n_in(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_int casadi_quadcopter_test_f_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real casadi_quadcopter_test_f_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* casadi_quadcopter_test_f_name_in(casadi_int i){
  switch (i) {
    case 0: return "x";
    case 1: return "u";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* casadi_quadcopter_test_f_name_out(casadi_int i){
  switch (i) {
    case 0: return "f";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_quadcopter_test_f_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_quadcopter_test_f_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_f_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* casadi_quadcopter_test_jfx:(x[10],u[3],d[10])->(jfx[10]) */
static int casadi_f1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a4, a5, a6, a7, a8, a9;
  a0=4.0000000000000001e-03;
  a1=arg[0]? arg[0][4] : 0;
  a2=casadi_sq(a1);
  a3=arg[0]? arg[0][5] : 0;
  a4=casadi_sq(a3);
  a2=(a2+a4);
  a4=arg[0]? arg[0][6] : 0;
  a5=casadi_sq(a4);
  a2=(a2+a5);
  a2=sqrt(a2);
  a5=cosh(a2);
  a5=(a0*a5);
  a6=arg[2]? arg[2][0] : 0;
  a5=(a5*a6);
  a7=(a1*a2);
  a7=(a0*a7);
  a8=arg[2]? arg[2][1] : 0;
  a7=(a7*a8);
  a5=(a5+a7);
  a7=(a3*a2);
  a7=(a0*a7);
  a9=arg[2]? arg[2][2] : 0;
  a7=(a7*a9);
  a5=(a5+a7);
  a7=(a4*a2);
  a7=(a0*a7);
  a10=arg[2]? arg[2][3] : 0;
  a7=(a7*a10);
  a5=(a5+a7);
  if (res[0]!=0) res[0][0]=a5;
  a5=cosh(a2);
  a5=(a0*a5);
  a5=(a5*a8);
  a7=(a1*a2);
  a7=(a0*a7);
  a7=(a7*a6);
  a5=(a5-a7);
  a7=(a4*a2);
  a7=(a0*a7);
  a7=(a7*a9);
  a5=(a5-a7);
  a7=(a3*a2);
  a7=(a0*a7);
  a7=(a7*a10);
  a5=(a5+a7);
  if (res[0]!=0) res[0][1]=a5;
  a5=(a4*a2);
  a5=(a0*a5);
  a5=(a5*a8);
  a7=(a3*a2);
  a7=(a0*a7);
  a7=(a7*a6);
  a5=(a5-a7);
  a7=cosh(a2);
  a7=(a0*a7);
  a7=(a7*a9);
  a5=(a5+a7);
  a7=(a1*a2);
  a7=(a0*a7);
  a7=(a7*a10);
  a5=(a5-a7);
  if (res[0]!=0) res[0][2]=a5;
  a5=(a3*a2);
  a5=(a0*a5);
  a5=(a5*a8);
  a7=(a4*a2);
  a7=(a0*a7);
  a7=(a7*a6);
  a5=(a5-a7);
  a7=(a1*a2);
  a7=(a0*a7);
  a7=(a7*a9);
  a5=(a5+a7);
  a7=cosh(a2);
  a7=(a0*a7);
  a7=(a7*a10);
  a5=(a5+a7);
  if (res[0]!=0) res[0][3]=a5;
  a5=arg[0]? arg[0][0] : 0;
  a7=sinh(a2);
  a11=(a1/a2);
  a12=(a7*a11);
  a12=(a0*a12);
  a12=(a5*a12);
  a13=arg[0]? arg[0][1] : 0;
  a14=(a1*a11);
  a14=(a2+a14);
  a14=(a0*a14);
  a14=(a13*a14);
  a12=(a12-a14);
  a14=arg[0]? arg[0][2] : 0;
  a15=(a3*a11);
  a15=(a0*a15);
  a15=(a14*a15);
  a12=(a12-a15);
  a15=arg[0]? arg[0][3] : 0;
  a16=(a4*a11);
  a16=(a0*a16);
  a16=(a15*a16);
  a12=(a12-a16);
  a12=(a12*a6);
  a16=(a1*a11);
  a16=(a2+a16);
  a16=(a0*a16);
  a16=(a5*a16);
  a17=sinh(a2);
  a18=(a17*a11);
  a18=(a0*a18);
  a18=(a13*a18);
  a16=(a16+a18);
  a18=(a4*a11);
  a18=(a0*a18);
  a18=(a14*a18);
  a16=(a16+a18);
  a18=(a3*a11);
  a18=(a0*a18);
  a18=(a15*a18);
  a16=(a16+a18);
  a16=(a16*a8);
  a12=(a12+a16);
  a16=(a3*a11);
  a16=(a0*a16);
  a16=(a5*a16);
  a18=(a4*a11);
  a18=(a0*a18);
  a18=(a13*a18);
  a16=(a16-a18);
  a18=sinh(a2);
  a19=(a18*a11);
  a19=(a0*a19);
  a19=(a14*a19);
  a16=(a16+a19);
  a19=(a1*a11);
  a19=(a2+a19);
  a19=(a0*a19);
  a19=(a15*a19);
  a16=(a16+a19);
  a16=(a16*a9);
  a12=(a12+a16);
  a16=(a4*a11);
  a16=(a0*a16);
  a16=(a5*a16);
  a19=(a3*a11);
  a19=(a0*a19);
  a19=(a13*a19);
  a16=(a16+a19);
  a19=(a1*a11);
  a19=(a2+a19);
  a19=(a0*a19);
  a19=(a14*a19);
  a16=(a16-a19);
  a19=sinh(a2);
  a11=(a19*a11);
  a11=(a0*a11);
  a11=(a15*a11);
  a16=(a16+a11);
  a16=(a16*a10);
  a12=(a12+a16);
  a16=arg[2]? arg[2][4] : 0;
  a12=(a12+a16);
  a11=8.0000000000000002e-03;
  a20=3.3178500331785003e+01;
  a21=1.7880000000000000e-02;
  a22=(a21*a4);
  a23=4.6140000000000000e-02;
  a24=(a23*a4);
  a22=(a22-a24);
  a22=(a20*a22);
  a22=(a11*a22);
  a25=arg[2]? arg[2][5] : 0;
  a22=(a22*a25);
  a12=(a12-a22);
  a22=2.1673168617251843e+01;
  a26=3.0140000000000000e-02;
  a27=(a26*a3);
  a28=(a21*a3);
  a28=(a27-a28);
  a28=(a22*a28);
  a28=(a11*a28);
  a29=arg[2]? arg[2][6] : 0;
  a28=(a28*a29);
  a12=(a12-a28);
  if (res[0]!=0) res[0][4]=a12;
  a12=(a3/a2);
  a28=(a7*a12);
  a28=(a0*a28);
  a28=(a5*a28);
  a30=(a1*a12);
  a30=(a0*a30);
  a30=(a13*a30);
  a28=(a28-a30);
  a30=(a3*a12);
  a30=(a2+a30);
  a30=(a0*a30);
  a30=(a14*a30);
  a28=(a28-a30);
  a30=(a4*a12);
  a30=(a0*a30);
  a30=(a15*a30);
  a28=(a28-a30);
  a28=(a28*a6);
  a30=(a1*a12);
  a30=(a0*a30);
  a30=(a5*a30);
  a31=(a17*a12);
  a31=(a0*a31);
  a31=(a13*a31);
  a30=(a30+a31);
  a31=(a4*a12);
  a31=(a0*a31);
  a31=(a14*a31);
  a30=(a30+a31);
  a31=(a3*a12);
  a31=(a2+a31);
  a31=(a0*a31);
  a31=(a15*a31);
  a30=(a30+a31);
  a30=(a30*a8);
  a28=(a28+a30);
  a30=(a3*a12);
  a30=(a2+a30);
  a30=(a0*a30);
  a30=(a5*a30);
  a31=(a4*a12);
  a31=(a0*a31);
  a31=(a13*a31);
  a30=(a30-a31);
  a31=(a18*a12);
  a31=(a0*a31);
  a31=(a14*a31);
  a30=(a30+a31);
  a31=(a1*a12);
  a31=(a0*a31);
  a31=(a15*a31);
  a30=(a30+a31);
  a30=(a30*a9);
  a28=(a28+a30);
  a30=(a4*a12);
  a30=(a0*a30);
  a30=(a5*a30);
  a31=(a3*a12);
  a31=(a2+a31);
  a31=(a0*a31);
  a31=(a13*a31);
  a30=(a30+a31);
  a31=(a1*a12);
  a31=(a0*a31);
  a31=(a14*a31);
  a30=(a30-a31);
  a12=(a19*a12);
  a12=(a0*a12);
  a12=(a15*a12);
  a30=(a30+a12);
  a30=(a30*a10);
  a28=(a28+a30);
  a30=5.5928411633109619e+01;
  a12=(a26*a4);
  a24=(a24-a12);
  a24=(a30*a24);
  a24=(a11*a24);
  a24=(a24*a16);
  a28=(a28-a24);
  a28=(a28+a25);
  a26=(a26*a1);
  a21=(a21*a1);
  a26=(a26-a21);
  a22=(a22*a26);
  a22=(a11*a22);
  a22=(a22*a29);
  a28=(a28-a22);
  if (res[0]!=0) res[0][5]=a28;
  a28=(a4/a2);
  a7=(a7*a28);
  a7=(a0*a7);
  a7=(a5*a7);
  a22=(a1*a28);
  a22=(a0*a22);
  a22=(a13*a22);
  a7=(a7-a22);
  a22=(a3*a28);
  a22=(a0*a22);
  a22=(a14*a22);
  a7=(a7-a22);
  a22=(a4*a28);
  a22=(a2+a22);
  a22=(a0*a22);
  a22=(a15*a22);
  a7=(a7-a22);
  a7=(a7*a6);
  a6=(a1*a28);
  a6=(a0*a6);
  a6=(a5*a6);
  a17=(a17*a28);
  a17=(a0*a17);
  a17=(a13*a17);
  a6=(a6+a17);
  a17=(a4*a28);
  a17=(a2+a17);
  a17=(a0*a17);
  a17=(a14*a17);
  a6=(a6+a17);
  a17=(a3*a28);
  a17=(a0*a17);
  a17=(a15*a17);
  a6=(a6+a17);
  a6=(a6*a8);
  a7=(a7+a6);
  a6=(a3*a28);
  a6=(a0*a6);
  a6=(a5*a6);
  a8=(a4*a28);
  a8=(a2+a8);
  a8=(a0*a8);
  a8=(a13*a8);
  a6=(a6-a8);
  a18=(a18*a28);
  a18=(a0*a18);
  a18=(a14*a18);
  a6=(a6+a18);
  a18=(a1*a28);
  a18=(a0*a18);
  a18=(a15*a18);
  a6=(a6+a18);
  a6=(a6*a9);
  a7=(a7+a6);
  a4=(a4*a28);
  a2=(a2+a4);
  a2=(a0*a2);
  a5=(a5*a2);
  a2=(a3*a28);
  a2=(a0*a2);
  a13=(a13*a2);
  a5=(a5+a13);
  a13=(a1*a28);
  a13=(a0*a13);
  a14=(a14*a13);
  a5=(a5-a14);
  a19=(a19*a28);
  a0=(a0*a19);
  a15=(a15*a0);
  a5=(a5+a15);
  a5=(a5*a10);
  a7=(a7+a5);
  a3=(a23*a3);
  a3=(a3-a27);
  a30=(a30*a3);
  a30=(a11*a30);
  a30=(a30*a16);
  a7=(a7-a30);
  a23=(a23*a1);
  a21=(a21-a23);
  a20=(a20*a21);
  a11=(a11*a20);
  a11=(a11*a25);
  a7=(a7-a11);
  a7=(a7+a29);
  if (res[0]!=0) res[0][6]=a7;
  a7=1.8299752000000002e-02;
  a7=(a7*a16);
  a16=8.3999999999999997e-01;
  a11=arg[2]? arg[2][7] : 0;
  a11=(a16*a11);
  a7=(a7+a11);
  if (res[0]!=0) res[0][7]=a7;
  a7=1.0855990880000001e-02;
  a7=(a7*a25);
  a25=arg[2]? arg[2][8] : 0;
  a25=(a16*a25);
  a7=(a7+a25);
  if (res[0]!=0) res[0][8]=a7;
  a7=-3.3911153600000004e-03;
  a7=(a7*a29);
  a29=arg[2]? arg[2][9] : 0;
  a16=(a16*a29);
  a7=(a7+a16);
  if (res[0]!=0) res[0][9]=a7;
  return 0;
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_jfx(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f1(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_jfx_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_jfx_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_jfx_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_jfx_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_jfx_release(int mem) {
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_jfx_incref(void) {
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_jfx_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int casadi_quadcopter_test_jfx_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int casadi_quadcopter_test_jfx_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real casadi_quadcopter_test_jfx_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* casadi_quadcopter_test_jfx_name_in(casadi_int i){
  switch (i) {
    case 0: return "x";
    case 1: return "u";
    case 2: return "d";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* casadi_quadcopter_test_jfx_name_out(casadi_int i){
  switch (i) {
    case 0: return "jfx";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_quadcopter_test_jfx_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_quadcopter_test_jfx_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_jfx_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* casadi_quadcopter_test_jfu:(x[10],u[3],d[10])->(jfu[3]) */
static int casadi_f2(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2;
  a0=4.0320000000000000e+01;
  a1=arg[2]? arg[2][7] : 0;
  a1=(a0*a1);
  if (res[0]!=0) res[0][0]=a1;
  a1=arg[2]? arg[2][8] : 0;
  a1=(a0*a1);
  if (res[0]!=0) res[0][1]=a1;
  a1=9.9727200000000005e-01;
  a2=arg[2]? arg[2][6] : 0;
  a1=(a1*a2);
  a2=arg[2]? arg[2][9] : 0;
  a0=(a0*a2);
  a1=(a1+a0);
  if (res[0]!=0) res[0][2]=a1;
  return 0;
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_jfu(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f2(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_jfu_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_jfu_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_jfu_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_jfu_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_jfu_release(int mem) {
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_jfu_incref(void) {
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_jfu_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int casadi_quadcopter_test_jfu_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int casadi_quadcopter_test_jfu_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real casadi_quadcopter_test_jfu_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* casadi_quadcopter_test_jfu_name_in(casadi_int i){
  switch (i) {
    case 0: return "x";
    case 1: return "u";
    case 2: return "d";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* casadi_quadcopter_test_jfu_name_out(casadi_int i){
  switch (i) {
    case 0: return "jfu";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_quadcopter_test_jfu_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_quadcopter_test_jfu_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_jfu_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* casadi_quadcopter_test_ell:(x[10],u[3])->(ell) */
static int casadi_f3(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2;
  a0=5.;
  a1=arg[0]? arg[0][0] : 0;
  a1=casadi_sq(a1);
  a0=(a0*a1);
  a1=1.0000000000000000e-02;
  a2=arg[0]? arg[0][1] : 0;
  a2=casadi_sq(a2);
  a2=(a1*a2);
  a0=(a0+a2);
  a2=arg[0]? arg[0][2] : 0;
  a2=casadi_sq(a2);
  a1=(a1*a2);
  a0=(a0+a1);
  if (res[0]!=0) res[0][0]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_ell(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f3(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_ell_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_ell_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_ell_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_ell_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_ell_release(int mem) {
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_ell_incref(void) {
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_ell_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int casadi_quadcopter_test_ell_n_in(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_int casadi_quadcopter_test_ell_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real casadi_quadcopter_test_ell_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* casadi_quadcopter_test_ell_name_in(casadi_int i){
  switch (i) {
    case 0: return "x";
    case 1: return "u";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* casadi_quadcopter_test_ell_name_out(casadi_int i){
  switch (i) {
    case 0: return "ell";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_quadcopter_test_ell_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_quadcopter_test_ell_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_ell_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* casadi_quadcopter_test_ellx:(x[10],u[3])->(ellx[10x1,3nz]) */
static int casadi_f4(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1;
  a0=5.;
  a1=arg[0]? arg[0][0] : 0;
  a1=(a1+a1);
  a0=(a0*a1);
  if (res[0]!=0) res[0][0]=a0;
  a0=1.0000000000000000e-02;
  a1=arg[0]? arg[0][1] : 0;
  a1=(a1+a1);
  a1=(a0*a1);
  if (res[0]!=0) res[0][1]=a1;
  a1=arg[0]? arg[0][2] : 0;
  a1=(a1+a1);
  a0=(a0*a1);
  if (res[0]!=0) res[0][2]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_ellx(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f4(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_ellx_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_ellx_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_ellx_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_ellx_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_ellx_release(int mem) {
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_ellx_incref(void) {
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_ellx_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int casadi_quadcopter_test_ellx_n_in(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_int casadi_quadcopter_test_ellx_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real casadi_quadcopter_test_ellx_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* casadi_quadcopter_test_ellx_name_in(casadi_int i){
  switch (i) {
    case 0: return "x";
    case 1: return "u";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* casadi_quadcopter_test_ellx_name_out(casadi_int i){
  switch (i) {
    case 0: return "ellx";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_quadcopter_test_ellx_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_quadcopter_test_ellx_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_ellx_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* casadi_quadcopter_test_ellu:(x[10],u[3])->(ellu[3x1,0nz]) */
static int casadi_f5(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_ellu(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f5(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_ellu_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_ellu_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_ellu_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_ellu_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_ellu_release(int mem) {
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_ellu_incref(void) {
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_ellu_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int casadi_quadcopter_test_ellu_n_in(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_int casadi_quadcopter_test_ellu_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real casadi_quadcopter_test_ellu_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* casadi_quadcopter_test_ellu_name_in(casadi_int i){
  switch (i) {
    case 0: return "x";
    case 1: return "u";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* casadi_quadcopter_test_ellu_name_out(casadi_int i){
  switch (i) {
    case 0: return "ellu";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_quadcopter_test_ellu_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_quadcopter_test_ellu_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_ellu_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* casadi_quadcopter_test_vf:(x[10])->(vf) */
static int casadi_f6(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3;
  a0=5.0000000000000000e-01;
  a1=arg[0]? arg[0][0] : 0;
  a1=casadi_sq(a1);
  a2=50.;
  a3=arg[0]? arg[0][1] : 0;
  a3=casadi_sq(a3);
  a2=(a2*a3);
  a1=(a1+a2);
  a2=100.;
  a3=arg[0]? arg[0][2] : 0;
  a3=casadi_sq(a3);
  a2=(a2*a3);
  a1=(a1+a2);
  a0=(a0*a1);
  if (res[0]!=0) res[0][0]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_vf(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f6(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_vf_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_vf_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_vf_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_vf_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_vf_release(int mem) {
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_vf_incref(void) {
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_vf_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int casadi_quadcopter_test_vf_n_in(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_int casadi_quadcopter_test_vf_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real casadi_quadcopter_test_vf_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* casadi_quadcopter_test_vf_name_in(casadi_int i){
  switch (i) {
    case 0: return "x";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* casadi_quadcopter_test_vf_name_out(casadi_int i){
  switch (i) {
    case 0: return "vf";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_quadcopter_test_vf_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_quadcopter_test_vf_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_vf_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 1;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* casadi_quadcopter_test_vfx:(x[10])->(vfx[10x1,3nz]) */
static int casadi_f7(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1;
  a0=5.0000000000000000e-01;
  a1=arg[0]? arg[0][0] : 0;
  a1=(a1+a1);
  a0=(a0*a1);
  if (res[0]!=0) res[0][0]=a0;
  a0=25.;
  a1=arg[0]? arg[0][1] : 0;
  a1=(a1+a1);
  a0=(a0*a1);
  if (res[0]!=0) res[0][1]=a0;
  a0=50.;
  a1=arg[0]? arg[0][2] : 0;
  a1=(a1+a1);
  a0=(a0*a1);
  if (res[0]!=0) res[0][2]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_vfx(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f7(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_vfx_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_vfx_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_vfx_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_vfx_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_vfx_release(int mem) {
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_vfx_incref(void) {
}

CASADI_SYMBOL_EXPORT void casadi_quadcopter_test_vfx_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int casadi_quadcopter_test_vfx_n_in(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_int casadi_quadcopter_test_vfx_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real casadi_quadcopter_test_vfx_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* casadi_quadcopter_test_vfx_name_in(casadi_int i){
  switch (i) {
    case 0: return "x";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* casadi_quadcopter_test_vfx_name_out(casadi_int i){
  switch (i) {
    case 0: return "vfx";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_quadcopter_test_vfx_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_quadcopter_test_vfx_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int casadi_quadcopter_test_vfx_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 1;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif