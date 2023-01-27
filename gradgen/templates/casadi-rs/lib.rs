use libc::{c_double, c_int};

pub const NX_{{ name | upper }}: usize = {{ nx }};
pub const NU_{{ name | upper }}: usize = {{ nu }};
pub const NPRED_{{ name | upper }}: usize = {{ N }};


extern "C" {
    fn init_interface_{{name}}();

    fn {{name}}_f(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;        
    fn {{name}}_jfx(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;
    fn {{name}}_jfu(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;
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

pub fn jfx(x: &[f64], u: &[f64], d: &[f64], jfxd: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr(), d.as_ptr()];
    let result = &mut [jfxd.as_mut_ptr()];
    unsafe { {{name}}_jfx(arguments.as_ptr(), result.as_mut_ptr()) as i32 }
}

pub fn jfu(x: &[f64], u: &[f64], d: &[f64], jfud: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr(), d.as_ptr()];
    let result = &mut [jfud.as_mut_ptr()];
    unsafe { {{name}}_jfu(arguments.as_ptr(), result.as_mut_ptr()) as i32 }
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

pub fn vfx(x: &[f64], u: &[f64], vfx_out: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr()];
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
        let x = [0.1; NX_{{ name | upper }}];
        let u = [0.1; NU_{{ name | upper }}];
        let mut fxu = [0.0; NX_{{ name | upper }}];
        assert_eq!(0, super::f(&x, &u, &mut fxu));
        println!("{:?}", fxu);
    }

    #[test]
    fn tst_simulate(){
        let mut x = vec![vec![0.0; NX_{{ name | upper }}]; NPRED_{{ name | upper }} + 1];
        x[0].copy_from_slice(&[1., 2., 3.]);
        let u = [0.1; 2]; // test with constant input
        let mut x_next = vec![0.0; NX_{{ name | upper }}];
        for i in 0..=NPRED_{{ name | upper }} - 1 {
            assert_eq!(0, super::f(&x[i], &u, &mut x_next));
            x[i + 1].copy_from_slice(&x_next);
        }
        println!("{:?}", x);
    }
    
    #[test]
    fn tst_jfx() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let d = [-0.8, 6.2, 0.0];
        let mut jfx_d = [0.0; NX_{{ name | upper }}];
        assert_eq!(0, super::jfx(&x, &u, &d, &mut jfx_d));
        println!("{:?}", jfx_d);
    }

    #[test]
    fn tst_jfu() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let d = [-0.8, 6.2, 0.0];
        let mut jfu_d = [0.0; NX_{{ name | upper }}];
        assert_eq!(0, super::jfu(&x, &u, &d, &mut jfu_d));
        println!("{:?}", jfu_d);
    }

    #[test]
    fn tst_ellx() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let mut ellx_res = [0.0; NX_{{ name | upper }}];
        assert_eq!(0, super::ellx(&x, &u, &mut ellx_res));
        println!("{:?}", ellx_res);
    }

    #[test]
    fn tst_ellu() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let mut ellu_res = [0.0; NU_{{ name | upper }}];
        assert_eq!(0, super::ellu(&x, &u, &mut ellu_res));
        println!("{:?}", ellu_res);
    }

    #[test]
    fn tst_vfx() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let mut vfx_res = [0.0; NX_{{ name | upper }}];
        assert_eq!(0, super::vfx(&x, &u, &mut vfx_res));
        println!("{:?}", vfx_res);
    }
}
