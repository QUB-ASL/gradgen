/*
clang -fPIC -shared casadi_functions.c -o casadi_functions.so
clang -DTEST_INTERFACE -o test interface.c casadi_functions.so -lm
*/

#include <stdlib.h>

/*
 * This is to be used ONLY for DEBUG purposes
 * Compile with -DTEST_INTERFACE
 */
#ifdef TEST_INTERFACE
#include <stdio.h>
#endif

#include "glob_header.h"

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

extern int casadi_ball_test_f(
    const casadi_real **arg,
    casadi_real **res,
    casadi_int *iw,
    casadi_real *w,
    int mem);

extern int casadi_ball_test_jfx(
    const casadi_real **arg,
    casadi_real **res,
    casadi_int *iw,
    casadi_real *w,
    int mem);

extern int casadi_ball_test_jfu(
    const casadi_real **arg,
    casadi_real **res,
    casadi_int *iw,
    casadi_real *w,
    int mem);

extern int casadi_ball_test_ell(
    const casadi_real **arg,
    casadi_real **res,
    casadi_int *iw,
    casadi_real *w,
    int mem);

extern int casadi_ball_test_ellx(
    const casadi_real **arg,
    casadi_real **res,
    casadi_int *iw,
    casadi_real *w,
    int mem);

extern int casadi_ball_test_ellu(
    const casadi_real **arg,
    casadi_real **res,
    casadi_int *iw,
    casadi_real *w,
    int mem);

extern int casadi_ball_test_vf(
    const casadi_real **arg,
    casadi_real **res,
    casadi_int *iw,
    casadi_real *w,
    int mem);

extern int casadi_ball_test_vfx(
    const casadi_real **arg,
    casadi_real **res,
    casadi_int *iw,
    casadi_real *w,
    int mem);


/* -------------------------- */
/*     Integer Workspaces     */
/* -------------------------- */

#if F_SZ_IW_BALL_TEST > 0
static casadi_int iws_f[F_SZ_IW_BALL_TEST];  
#else
static casadi_int *iws_f = NULL;
#endif

#if JFX_SZ_IW_BALL_TEST > 0
static casadi_int iws_jfx[JFX_SZ_IW_BALL_TEST];  
#else
static casadi_int *iws_jfx = NULL;
#endif


#if JFU_SZ_IW_BALL_TEST > 0
static casadi_int iws_jfu[JFU_SZ_IW_BALL_TEST];  
#else
static casadi_int *iws_jfu = NULL;
#endif

#if ELL_SZ_IW_BALL_TEST > 0
static casadi_int iws_ell[ELL_SZ_IW_BALL_TEST];  
#else
static casadi_int *iws_ell = NULL;
#endif

#if ELLX_SZ_IW_BALL_TEST > 0
static casadi_int iws_ellx[ELLX_SZ_IW_BALL_TEST];  
#else
static casadi_int *iws_ellx = NULL;
#endif

#if ELLU_SZ_IW_BALL_TEST > 0
static casadi_int iws_ellu[ELLU_SZ_IW_BALL_TEST];  
#else
static casadi_int *iws_ellu = NULL;
#endif

#if VF_SZ_IW_BALL_TEST > 0
static casadi_int iws_vf[VF_SZ_IW_BALL_TEST];  
#else
static casadi_int *iws_vf = NULL;
#endif

#if VFX_SZ_IW_BALL_TEST > 0
static casadi_int iws_vfx[VFX_SZ_IW_BALL_TEST];  
#else
static casadi_int *iws_vfx = NULL;
#endif


/* -------------------------- */
/*       Real Workspaces      */
/* -------------------------- */

#if F_SZ_W_BALL_TEST > 0
static casadi_real ws_f[F_SZ_W_BALL_TEST];
#else
static casadi_real *ws_f = NULL;
#endif

#if JFX_SZ_W_BALL_TEST > 0
static casadi_real ws_jfx[JFX_SZ_W_BALL_TEST];
#else
static casadi_real *ws_jfx = NULL;
#endif

#if JFU_SZ_W_BALL_TEST > 0
static casadi_real ws_jfu[JFU_SZ_W_BALL_TEST];
#else
static casadi_real *ws_jfu = NULL;
#endif

#if ELL_SZ_W_BALL_TEST > 0
static casadi_real ws_ell[ELL_SZ_W_BALL_TEST];
#else
static casadi_real *ws_ell = NULL;
#endif

#if ELLX_SZ_W_BALL_TEST > 0
static casadi_real ws_ellx[ELLX_SZ_W_BALL_TEST];
#else
static casadi_real *ws_ellx = NULL;
#endif

#if ELLU_SZ_W_BALL_TEST > 0
static casadi_real ws_ellu[ELLU_SZ_W_BALL_TEST];
#else
static casadi_real *ws_ellu = NULL;
#endif

#if VF_SZ_W_BALL_TEST > 0
static casadi_real ws_vf[VF_SZ_W_BALL_TEST];
#else
static casadi_real *ws_vf = NULL;
#endif

#if VFX_SZ_W_BALL_TEST > 0
static casadi_real ws_vfx[VFX_SZ_W_BALL_TEST];
#else
static casadi_real *ws_vfx = NULL;
#endif


/* -------------------------- */
/*        Result Spaces       */
/* -------------------------- */
static casadi_real *f_res[F_SZ_RES_BALL_TEST];
static casadi_real *jfx_res[JFX_SZ_RES_BALL_TEST];
static casadi_real *jfu_res[JFU_SZ_RES_BALL_TEST];
static casadi_real *ell_res[ELL_SZ_RES_BALL_TEST];
static casadi_real *ellx_res[ELLX_SZ_RES_BALL_TEST];
static casadi_real *ellu_res[ELLU_SZ_RES_BALL_TEST];
static casadi_real *vf_res[VF_SZ_RES_BALL_TEST];
static casadi_real *vfx_res[VFX_SZ_RES_BALL_TEST];


/* ------------------------------------ */
/*      Persistent (x, u, d) memory     */
/* ------------------------------------ */
#define N_PERSISTENT_MEM_BALL_TEST (2 * NX_BALL_TEST + NU_BALL_TEST)
static casadi_real persistent_memory[N_PERSISTENT_MEM_BALL_TEST];  /* (x, u, d) */
#define IDX_X_BALL_TEST 0
#define IDX_U_BALL_TEST IDX_X_BALL_TEST + NX_BALL_TEST
#define IDX_D_BALL_TEST IDX_U_BALL_TEST + NU_BALL_TEST


void init_interface_ball_test(void) {
    /* nothing */
}

/* Copy (x) into persistent_memory; arg = {x}  */
static void copy_x_to_persistent(const casadi_real** arg) {
    unsigned int i;
    for (i=0; i<NX_BALL_TEST; i++)  
        persistent_memory[IDX_X_BALL_TEST + i] = arg[0][i];  /* copy x */    
}

/* Copy (x, u) into persistent_memory; arg = {x, u}  */
static void copy_xu_to_persistent(const casadi_real** arg) {
    unsigned int i;
    copy_x_to_persistent(arg);  /* copy x */
    for (i=0; i<NU_BALL_TEST; i++)  
        persistent_memory[IDX_U_BALL_TEST + i] = arg[1][i];  /* copy u */    
}

/* Copy (x, u, d) into persistent_memory; arg = {x, u, d}  */
static void copy_xud_to_persistent(const casadi_real** arg) {
    unsigned int i;
    copy_xu_to_persistent(arg);
    for (i=0; i<NX_BALL_TEST; i++)  
        persistent_memory[IDX_D_BALL_TEST + i] = arg[2][i];  /* copy d */    
}

int ball_test_f(const casadi_real** arg, casadi_real** res) {
    copy_xu_to_persistent(arg);  /* arg = {x, u} */
    const casadi_real* args__[F_SZ_ARG_BALL_TEST] = {
        persistent_memory + IDX_X_BALL_TEST,
        persistent_memory + IDX_U_BALL_TEST};
    f_res[0] = res[0];
    return casadi_ball_test_f(args__, f_res, iws_f, ws_f, 0);
}

int ball_test_jfx(const casadi_real** arg, casadi_real** res) {
    copy_xud_to_persistent(arg);  /* arg = {x, u, d} */
    const casadi_real* args__[JFX_SZ_ARG_BALL_TEST] = {
        persistent_memory + IDX_X_BALL_TEST,
        persistent_memory + IDX_U_BALL_TEST,
        persistent_memory + IDX_D_BALL_TEST};
    jfx_res[0] = res[0];
    return casadi_ball_test_jfx(args__, jfx_res, iws_jfx, ws_jfx, 0);
}

int ball_test_jfu(const casadi_real** arg, casadi_real** res) {
    copy_xud_to_persistent(arg);  /* arg = {x, u, d} */
    const casadi_real* args__[JFU_SZ_ARG_BALL_TEST] = {
        persistent_memory + IDX_X_BALL_TEST,
        persistent_memory + IDX_U_BALL_TEST,
        persistent_memory + IDX_D_BALL_TEST};
    jfu_res[0] = res[0];
    return casadi_ball_test_jfu(args__, jfu_res, iws_jfu, ws_jfu, 0);
}

int ball_test_ell(const casadi_real** arg, casadi_real** res) {
    copy_xu_to_persistent(arg);  /* arg = {x, u} */
    const casadi_real* args__[ELL_SZ_ARG_BALL_TEST] = {
        persistent_memory + IDX_X_BALL_TEST,
        persistent_memory + IDX_U_BALL_TEST};
    ell_res[0] = res[0];
    return casadi_ball_test_ell(args__, ell_res, iws_ell, ws_ell, 0);
}

int ball_test_ellx(const casadi_real** arg, casadi_real** res) {
    copy_xu_to_persistent(arg);  /* arg = {x, u} */
    const casadi_real* args__[ELLX_SZ_ARG_BALL_TEST] = {
        persistent_memory + IDX_X_BALL_TEST,
        persistent_memory + IDX_U_BALL_TEST};
    ellx_res[0] = res[0];
    return casadi_ball_test_ellx(args__, ellx_res, iws_ellx, ws_ellx, 0);
}

int ball_test_ellu(const casadi_real** arg, casadi_real** res) {
    copy_xu_to_persistent(arg);  /* arg = {x, u} */
    const casadi_real* args__[ELLU_SZ_ARG_BALL_TEST] = {
        persistent_memory + IDX_X_BALL_TEST,
        persistent_memory + IDX_U_BALL_TEST};
    ellu_res[0] = res[0];
    return casadi_ball_test_ellu(args__, ellu_res, iws_ellu, ws_ellu, 0);
}

int ball_test_vf(const casadi_real** arg, casadi_real** res) {
    copy_x_to_persistent(arg);  /* arg = {x} */
    const casadi_real* args__[VF_SZ_ARG_BALL_TEST] = {persistent_memory + IDX_X_BALL_TEST};
    vf_res[0] = res[0];
    return casadi_ball_test_vf(args__, vf_res, iws_vf, ws_vf, 0);
}

int ball_test_vfx(const casadi_real** arg, casadi_real** res) {
    copy_x_to_persistent(arg);  /* arg = {x} */
    const casadi_real* args__[VFX_SZ_ARG_BALL_TEST] = {persistent_memory + IDX_X_BALL_TEST};
    vfx_res[0] = res[0];
    return casadi_ball_test_vfx(args__, vfx_res, iws_vfx, ws_vfx, 0);
}

#ifdef TEST_INTERFACE
/* -------------------------- */
/*           Testing          */
/* -------------------------- */

void call_f_test(){
    double x[NX_BALL_TEST] = {1.570796326794897, 10., 2.};
    double u[NU_BALL_TEST] = {1.5, 2.1};
    double x_next[NX_BALL_TEST];
    const double *arg[2] = {x, u};
    double *res[1] = {x_next};
    ball_test_f(arg, res);

    int i;
    for (i = 0; i < NX_BALL_TEST; i++)
    {
        printf("x+ >> %g\n", x_next[i]);
    }
}

void call_jfx_test()
{
    double x[NX_BALL_TEST] = {1.570796326794897, 10., 2.};
    double u[NU_BALL_TEST] = {1.5, 2.1};
    double d[NX_BALL_TEST] = {0, 1, 0};

    double jfx[NX_BALL_TEST];
    const double *arg[3] = {x, u, d};
    double *res[1] = {jfx};
    ball_test_jfx(arg, res);

    for (unsigned int i = 0; i < NX_BALL_TEST; i++)
    {
        printf("Jxf >> %g\n", jfx_res[0][i]);
    }
}

void call_jfu_test()
{
    double x[NX_BALL_TEST] = {1.570796326794897, 10., 2.};
    double u[NU_BALL_TEST] = {1.5, 2.1};
    double d[NX_BALL_TEST] = {1, 1, 0};

    double jfu[NX_BALL_TEST];
    const double *arg[3] = {x, u, d};
    double *res[1] = {jfu};
    ball_test_jfu(arg, res);

    for (unsigned int i = 0; i < NX_BALL_TEST; i++)
    {
        printf("Juf >> %g\n", jfu_res[0][i]);
    }
}

void call_ellx_test()
{
    double x[NX_BALL_TEST] = {1.570796326794897, 10., 2.};
    double u[NU_BALL_TEST] = {1.5, 2.1};

    double ellx[NX_BALL_TEST];
    const double *arg[2] = {x, u};
    double *res[1] = {ellx};
    ball_test_ellx(arg, res);

    for (unsigned int i = 0; i < NX_BALL_TEST; i++)
    {
        printf("ell_x >> %g\n", ellx_res[0][i]);
    }
}

void call_vfx_test()
{
    double x[NX_BALL_TEST] = {1.570796326794897, 10., 2.};

    double vfx[NX_BALL_TEST];
    const double *arg[1] = {x};
    double *res[1] = {vfx};
    ball_test_vfx(arg, res);

    for (unsigned int i = 0; i < NX_BALL_TEST; i++)
    {
        printf("vfx >> %g\n", vfx_res[0][i]);
    }
}

/* for testing purposes only! */
int main()
{
    call_f_test();
    call_jfx_test();
    call_jfu_test();
    call_ellx_test();
    call_vfx_test();
    return 0;
}   
#endif
