/*
 * This is to be used ONLY for DEBUG purposes
 * Compile with -DTEST_INTERFACE
 */
#ifdef TEST_INTERFACE
#include <stdio.h>
#include <math.h>
#include <sys/time.h>  // for linux and mac
#endif

// gap

/*
 * This is to be used ONLY for DEBUG purposes
 * Compile with -DTEST_INTERFACE
 */
#if defined(TEST_INTERFACE) // && defined(PRECONDITIONING_BM_ALGO_2_INV_PEND_OPEN)

static casadi_real u_test[NU_BM_ALGO_2_INV_PEND_OPEN];
static casadi_real xi_test[NXI_BM_ALGO_2_INV_PEND_OPEN];
static casadi_real p_test[NP_BM_ALGO_2_INV_PEND_OPEN];
static casadi_real g_test[NU_BM_ALGO_2_INV_PEND_OPEN];

static void init_up_test(void) {
    unsigned int i;
    for (i=0; i<NU_BM_ALGO_2_INV_PEND_OPEN; i++){
        u_test[i] = 20 + i;
    }
    for (i=0; i<NP_BM_ALGO_2_INV_PEND_OPEN; i++){
        p_test[i] = 1.5 + 15 * i;
    }
}

static void init_casadi_grad_test(void) {
    unsigned int i;
    for (i=0; i<NU_BM_ALGO_2_INV_PEND_OPEN; i++){
        u_test[i] = 1.0;
    }
    for (i=0; i<NXI_BM_ALGO_2_INV_PEND_OPEN; i++){
        xi_test[i] = 0.0;
    }
    for (i=0; i<NP_BM_ALGO_2_INV_PEND_OPEN; i++){
        p_test[i] = 0.0;
    }
    for (i=0; i<NU_BM_ALGO_2_INV_PEND_OPEN; i++){
        g_test[i] = 0.0;  // initialise gradient
    }
}

static void print_static_array(void){
    unsigned int i;
    for (i=0; i<NU_BM_ALGO_2_INV_PEND_OPEN; i++){
        printf("u[%2d] = %4.2f\n", i, uxip_space[i]);
    }
    for (i=0; i<NXI_BM_ALGO_2_INV_PEND_OPEN; i++){
        printf("xi[%2d] = %4.2f\n", i, uxip_space[IDX_XI_BM_ALGO_2_INV_PEND_OPEN+i]);
    }
    for (i=0; i<NP_BM_ALGO_2_INV_PEND_OPEN; i++){
        printf("p[%2d] = %4.2f\n", i, uxip_space[IDX_P_BM_ALGO_2_INV_PEND_OPEN+i]);
    }
    printf("w_cost = %g\n", uxip_space[IDX_WC_BM_ALGO_2_INV_PEND_OPEN]);
#if N1_BM_ALGO_2_INV_PEND_OPEN > 0
     for (i=0; i<N1_BM_ALGO_2_INV_PEND_OPEN; i++){
        printf("w1[%2d] = %g\n", i, uxip_space[IDX_W1_BM_ALGO_2_INV_PEND_OPEN+i]);
    }
#endif /* IF N1 > 0 */
#if N2_BM_ALGO_2_INV_PEND_OPEN > 0
     for (i=0; i<N2_BM_ALGO_2_INV_PEND_OPEN; i++){
        printf("w2[%2d] = %g\n", i, uxip_space[IDX_W2_BM_ALGO_2_INV_PEND_OPEN+i]);
    }
#endif /* IF N2 > 0 */
}

static casadi_real test_initial_penalty(void) {
    const casadi_real *args[2] = {u_test, p_test};
    casadi_real initial_penalty = -1.;
    casadi_real *res[1] = { &initial_penalty };
    init_penalty_function_bm_algo_2_inv_pend_open(args, res);
    return initial_penalty;
}

static void test_casadi_gradient_calculation(void) {
    const casadi_real *args[3] = {u_test, xi_test, p_test};
    casadi_real *res[1] = { g_test };
    grad_cost_function_bm_algo_2_inv_pend_open(args, res);
}

static void run_test_initial_penalty(void) {
    init_interface_bm_algo_2_inv_pend_open();
    init_up_test();
    const casadi_real *argz[2] = {u_test, p_test};
    preconditioning_www_bm_algo_2_inv_pend_open(argz);

    /*
     * Since this is invoked after `test_w_cost`, `test_w1` and `test_w2`, the ws have been computed previously
     * and are available in `uxipw_space`. The caller does need to provide them
     */
    casadi_real rho1 = test_initial_penalty();
    print_static_array();
    printf("rho1 = %g\n", rho1);
}

long timeNow(void);
long timeNow(void)
{
  // Special struct defined by sys/time.h
  struct timeval tv;
  // Long int to store the elapsed time
  long fullTime;
  // This only works under GNU C I think
  gettimeofday(&tv, NULL);
  // Do some math to convert struct -> long
  fullTime = tv.tv_sec*1000000 + tv.tv_usec;
  return fullTime;
}

static void run_test_casadi_gradient_calculation(void) {
    init_interface_bm_algo_2_inv_pend_open();
    init_casadi_grad_test();
    const casadi_real *argz[2] = {u_test, p_test};
    preconditioning_www_bm_algo_2_inv_pend_open(argz);

    /*
    * Since this is invoked after `test_w_cost`, `test_w1` and `test_w2`, the ws have been computed previously
    * and are available in `uxipw_space`. The caller does need to provide them
    */
    int n_runs = 1e6;
    float total = n_runs;
    long before, after, duration;
    before = timeNow();
    for (int i=0; i<n_runs; i++) {
        test_casadi_gradient_calculation();
    }
    after = timeNow();
    duration = after - before;
    float data_mean = duration / total;
    // print the average runtime (in microsecond) for calculating the total gradient once
    printf("mean:%.2f", data_mean);
}

int main(void) {
//    run_test_initial_penalty();
    run_test_casadi_gradient_calculation();
    return 0;
}

#endif /* END of TEST_INTERFACE and PRECONDITIONING_BM_ALGO_2_INV_PEND_OPEN */