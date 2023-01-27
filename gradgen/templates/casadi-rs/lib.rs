use libc::{c_double, c_int};

extern "C" {
    fn init_interface_{{name}}();

    fn {{name}}_f(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;        
    fn {{name}}_jfx(arg: *const *const c_double, casadi_results: *mut *mut c_double) -> c_int;

}


pub fn init_{{name}}() {
    unsafe {
        init_interface_{{name}}();
    }
}

pub fn f(x: &[f64], u: &[f64], fxu: &mut [f64]) -> i32{
    let arguments = &[x.as_ptr(), u.as_ptr()];
    let result = &mut [fxu.as_mut_ptr()];

    unsafe { {{name}}_f(arguments.as_ptr(), result.as_mut_ptr()) as i32}
}

pub fn jfx(x: &[f64], u: &[f64], d: &[f64], jfxd: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr(), d.as_ptr()];
    let result = &mut [jfxd.as_mut_ptr()];
    unsafe { {{name}}_jfx(arguments.as_ptr(), result.as_mut_ptr()) as i32 }
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
        let x = [0.1; 3];
        let u = [0.1; 2];
        let mut fxu = [0.0; 3];
        assert_eq!(0, super::f(&x, &u, &mut fxu));
        println!("{:?}", fxu);
    }
    
    #[test]
    fn tst_jfx() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let d = [-0.8, 6.2, 0.0];
        let mut jfx_d = [0.0; 3];
        assert_eq!(0, super::jfx(&x, &u, &d, &mut jfx_d));
        println!("{:?}", jfx_d);
    }
}