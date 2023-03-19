use libc::{c_double, c_int};

/// number of states 
pub const NX: usize = {{ nx }};
/// number of inputs
pub const NU: usize = {{ nu }};
/// prediction horizon
pub const NPRED: usize = {{ N }};


extern "C" {
    fn init_interface_{{name}}();

    {% for index_w in w %}
    fn {{name}}_f{{index_w}}(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;
    fn {{name}}_jfx{{index_w}}(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;
    fn {{name}}_jfu{{index_w}}(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;
    fn {{name}}_ellx{{index_w}}(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;
    fn {{name}}_ellu{{index_w}}(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;
    {% endfor %}
    fn {{name}}_vfx(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;
}


pub fn init_{{name}}() {
    unsafe {
        init_interface_{{name}}();
    }
}

{% for index_w in w %}
pub fn f{{index_w}}(x: &[f64], u: &[f64], fxu: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr()];
    let result = &mut [fxu.as_mut_ptr()];
    unsafe { {{name}}_f{{index_w}}(arguments.as_ptr(), result.as_mut_ptr()) as i32}
}

pub fn jfx{{index_w}}(x: &[f64], u: &[f64], d: &[f64], jfxd: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr(), d.as_ptr()];
    let result = &mut [jfxd.as_mut_ptr()];
    unsafe { {{name}}_jfx{{index_w}}(arguments.as_ptr(), result.as_mut_ptr()) as i32 }
}

pub fn jfu{{index_w}}(x: &[f64], u: &[f64], d: &[f64], jfud: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr(), d.as_ptr()];
    let result = &mut [jfud.as_mut_ptr()];
    unsafe { {{name}}_jfu{{index_w}}(arguments.as_ptr(), result.as_mut_ptr()) as i32 }
}

pub fn ellx{{index_w}}(x: &[f64], u: &[f64], ellx_out: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr()];
    let result = &mut [ellx_out.as_mut_ptr()];
    unsafe { {{name}}_ellx{{index_w}}(arguments.as_ptr(), result.as_mut_ptr()) as i32 }
}

pub fn ellu{{index_w}}(x: &[f64], u: &[f64], ellu_out: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr()];
    let result = &mut [ellu_out.as_mut_ptr()];
    unsafe { {{name}}_ellu{{index_w}}(arguments.as_ptr(), result.as_mut_ptr()) as i32 }
}
{% endfor %}

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
    fn tst_f0(){
        let x = [0.1; NX];
        let u = [0.1; NU];
        let mut fxu = [0.0; NX];
        assert_eq!(0, f0(&x, &u, &mut fxu));
    }  
    
    #[test]
    fn tst_jfx0() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let d = [-0.8, 6.2, 0.0];
        let mut jfx_d = [0.0; NX];
        assert_eq!(0, jfx0(&x, &u, &d, &mut jfx_d));
    }

    #[test]
    fn tst_jfu0() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let d = [-0.8, 6.2, 0.0];
        let mut jfu_d = [0.0; NX];
        assert_eq!(0, jfu0(&x, &u, &d, &mut jfu_d));
    }

    #[test]
    fn tst_ellx0() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let mut ellx_res = [0.0; NX];
        assert_eq!(0, ellx0(&x, &u, &mut ellx_res));
    }

    #[test]
    fn tst_ellu0() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let mut ellu_res = [0.0; NU];
        assert_eq!(0, ellu0(&x, &u, &mut ellu_res));
    }

    #[test]
    fn tst_vfx() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let mut vfx_res = [0.0; NX];
        assert_eq!(0, vfx(&x, &mut vfx_res));
    }
}
