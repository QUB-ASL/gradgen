use libc::{c_double, c_int};

/// number of states 
pub const NX: usize = {{ nx }};
/// number of inputs
pub const NU: usize = {{ nu }};
/// prediction horizon
pub const NPRED: usize = {{ N }};


extern "C" {
    fn init_interface_{{name}}();

    fn {{name}}_f(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;
    fn {{name}}_fx(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;
    fn {{name}}_fu(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;
    fn {{name}}_ellx(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;
    fn {{name}}_ellu(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;
    fn {{name}}_vfx(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;
}


pub fn init_{{name}}() {
    unsafe {
        init_interface_{{name}}();
    }
}

pub fn f(x: &[f64], u: &[f64], fxu: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr()];
    let result = &mut [fxu.as_mut_ptr()];
    unsafe { {{name}}_f(arguments.as_ptr(), result.as_mut_ptr()) as i32}
}

pub fn fx(x: &[f64], u: &[f64], d: &[f64], fxd: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr(), d.as_ptr()];
    let result = &mut [fxd.as_mut_ptr()];
    unsafe { {{name}}_fx(arguments.as_ptr(), result.as_mut_ptr()) as i32 }
}

pub fn fu(x: &[f64], u: &[f64], d: &[f64], fud: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr(), d.as_ptr()];
    let result = &mut [fud.as_mut_ptr()];
    unsafe { {{name}}_fu(arguments.as_ptr(), result.as_mut_ptr()) as i32 }
}

pub fn ellx(x: &[f64], u: &[f64], ellx_out: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr()];
    let result = &mut [ellx_out.as_mut_ptr()];
    unsafe { {{name}}_ellx(arguments.as_ptr(), result.as_mut_ptr()) as i32 }
}

pub fn ellu(x: &[f64], u: &[f64], ellu_out: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr()];
    let result = &mut [ellu_out.as_mut_ptr()];
    unsafe { {{name}}_ellu(arguments.as_ptr(), result.as_mut_ptr()) as i32 }
}

pub fn vfx(x: &[f64], vfx_out: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr()];
    let result = &mut [vfx_out.as_mut_ptr()];
    unsafe { {{name}}_vfx(arguments.as_ptr(), result.as_mut_ptr()) as i32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tst_init() {
        init_{{name}}();
    }

    #[test]
    fn tst_f(){
        let x = [0.1; NX];
        let u = [0.1; NU];
        let mut fxu = [0.0; NX];
        assert_eq!(0, f(&x, &u, &mut fxu));
    }

    #[test]
    fn tst_fx() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let d = [-0.8, 6.2, 0.0];
        let mut fx_d = [0.0; NX];
        assert_eq!(0, fx(&x, &u, &d, &mut fx_d));
    }

    #[test]
    fn tst_fu() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let d = [-0.8, 6.2, 0.0];
        let mut fu_d = [0.0; NX];
        assert_eq!(0, fu(&x, &u, &d, &mut fu_d));
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