use libc::{c_double, c_int};

/// number of states 
pub const NX: usize = 4;
/// number of inputs
pub const NU: usize = 1;
/// prediction horizon
pub const NPRED: usize = 15;


extern "C" {
    fn init_interface_ball_test();

    fn ball_test_f(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;        
    fn ball_test_jfx(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;
    fn ball_test_jfu(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;
    fn ball_test_ell(arg: *const *const c_double, casadi_results:  *mut *mut c_double) -> c_int;
    fn ball_test_ellx(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;
    fn ball_test_ellu(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;
    fn ball_test_vf(arg: *const *const  c_double, casadi_results:  *mut *mut c_double) -> c_int;
    fn ball_test_vfx(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;
}


pub fn init_ball_test() {
    unsafe {
        init_interface_ball_test();
    }
}

pub fn f(x: &[f64], u: &[f64], fxu: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr()];
    let result = &mut [fxu.as_mut_ptr()];
    unsafe { ball_test_f(arguments.as_ptr(), result.as_mut_ptr()) as i32}
}

pub fn jfx(x: &[f64], u: &[f64], d: &[f64], jfxd: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr(), d.as_ptr()];
    let result = &mut [jfxd.as_mut_ptr()];
    unsafe { ball_test_jfx(arguments.as_ptr(), result.as_mut_ptr()) as i32 }
}

pub fn jfu(x: &[f64], u: &[f64], d: &[f64], jfud: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr(), d.as_ptr()];
    let result = &mut [jfud.as_mut_ptr()];
    unsafe { ball_test_jfu(arguments.as_ptr(), result.as_mut_ptr()) as i32 }
}


pub fn ell(x: &[f64], u: &[f64], ell_out: &mut f64) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr()];
    let mut result =  ell_out as *mut f64 ;
    unsafe {  ball_test_ell(arguments.as_ptr(),   &mut result) as i32 }
}

pub fn ellx(x: &[f64], u: &[f64], ellx_out: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr()];
    let result = &mut [ellx_out.as_mut_ptr()];
    unsafe { ball_test_ellx(arguments.as_ptr(), result.as_mut_ptr()) as i32 }
}

pub fn ellu(x: &[f64], u: &[f64], ellu_out: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr()];
    let result = &mut [ellu_out.as_mut_ptr()];
    unsafe { ball_test_ellu(arguments.as_ptr(), result.as_mut_ptr()) as i32 }
}

pub fn vf(x: &[f64], vf_out: &mut f64) -> i32 {
    let arguments = &[x.as_ptr()];
    let mut result =  vf_out as *mut f64 ;
    unsafe { ball_test_vf(arguments.as_ptr(), &mut result) as i32 }
}


pub fn vfx(x: &[f64], vfx_out: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr()];
    let result = &mut [vfx_out.as_mut_ptr()];
    unsafe { ball_test_vfx(arguments.as_ptr(), result.as_mut_ptr()) as i32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tst_init() {
        init_ball_test();
    }

    #[test]
    fn tst_f(){
        let x = [0.1; NX];
        let u = [0.1; NU];
        let mut fxu = [0.0; NX];
        assert_eq!(0, f(&x, &u, &mut fxu));
    }  
    
    #[test]
    fn tst_jfx() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let d = [-0.8, 6.2, 0.0];
        let mut jfx_d = [0.0; NX];
        assert_eq!(0, jfx(&x, &u, &d, &mut jfx_d));
    }

    #[test]
    fn tst_jfu() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let d = [-0.8, 6.2, 0.0];
        let mut jfu_d = [0.0; NX];
        assert_eq!(0, jfu(&x, &u, &d, &mut jfu_d));
    }

    #[test]
    fn tst_ellx() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let mut ellx_res = [0.0; NX];
        assert_eq!(0, ellx(&x, &u, &mut ellx_res));
    }

    #[test]
    fn tst_ellu() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let mut ellu_res = [0.0; NU];
        assert_eq!(0, ellu(&x, &u, &mut ellu_res));
    }

    #[test]
    fn tst_vfx() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let mut vfx_res = [0.0; NX];
        assert_eq!(0, vfx(&x, &mut vfx_res));
    }
}