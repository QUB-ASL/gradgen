use libc::{c_double, c_int};

/// number of states 
pub const NX: usize = {{ nx }};
/// number of inputs
pub const NU: usize = {{ nu }};
/// prediction horizon
pub const NPRED: usize = {{ N }};

/* SCENARIO TREE */ 

/// Number of stages (i.e., prediction horizon)
pub const NUM_STAGES: usize = {{ tree.num_stages }};

/// Total number of nodes
pub const NUM_NODES: usize = {{ tree.num_nodes }};

/// Number of non-leaf nodes
pub const NUM_NONLEAF_NODES: usize = {{ children_to | length }};

/// List of stage indices of each node
pub const STAGE_OF_NODE : &[usize]= &[
    {{ tree._ScenarioTree__stages | join(', ') }}
];

/// List of ancestors of each node
pub const ANCESTOR_OF_NODE : &[usize]= &[
    0, {{ tree._ScenarioTree__ancestors[1:] | join(', ') }}
];

/// Event, w, leading up to a given node
pub const EVENT_AT_NODE : &[usize]= &[
    0, {{ tree._ScenarioTree__w_idx[1:] | join(', ') }}
];


/// Children (from) of a node
pub const CHILDREN_OF_NODE_FROM : &[usize]= &[
    {{ children_from | join(', ') }}
];

/// Children (to) of a node
pub const CHILDREN_OF_NODE_TO : &[usize]= &[
    {{ children_to | join(', ') }}
];


pub const NODES_AT_STAGE_FROM : &[usize]= &[
    {{ nodes_at_stage_from | join(', ') }}
];

pub const NODES_AT_STAGE_TO : &[usize]= &[
    {{ nodes_at_stage_to | join(', ') }}
];


/// Probability at given node
pub const PROBABILITY_AT_NODE : &[f64]= &[
    {{ tree._ScenarioTree__probability[:tree.num_nodes] | join(', ') }}
];



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

pub fn f(x: &[f64], u: &[f64], w: &[f64], fxu: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr(), w.as_ptr()];
    let result = &mut [fxu.as_mut_ptr()];
    unsafe { {{name}}_f(arguments.as_ptr(), result.as_mut_ptr()) as i32}
}

pub fn jfx(x: &[f64], u: &[f64], d: &[f64], w: &[f64], jfx_d: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr(), d.as_ptr(), w.as_ptr()];
    let result = &mut [jfx_d.as_mut_ptr()];
    unsafe { {{name}}_jfx(arguments.as_ptr(), result.as_mut_ptr()) as i32 }
}

pub fn jfu(x: &[f64], u: &[f64], d: &[f64], w: &[f64], jfu_d: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr(), d.as_ptr(), w.as_ptr()];
    let result = &mut [jfu_d.as_mut_ptr()];
    unsafe { {{name}}_jfu(arguments.as_ptr(), result.as_mut_ptr()) as i32 }
}

pub fn ellx(x: &[f64], u: &[f64], w: &[f64], ellx_out: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr(), w.as_ptr()];
    let result = &mut [ellx_out.as_mut_ptr()];
    unsafe { {{name}}_ellx(arguments.as_ptr(), result.as_mut_ptr()) as i32 }
}

pub fn ellu(x: &[f64], u: &[f64], w: &[f64], ellu_out: &mut [f64]) -> i32 {
    let arguments = &[x.as_ptr(), u.as_ptr(), w.as_ptr()];
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
        let w = [0.0];
        let mut fxu = [0.0; NX];
        assert_eq!(0, f(&x, &u, &w, &mut fxu));
    }

    #[test]
    fn tst_jfx() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let d = [-0.8, 6.2, 0.0];
        let w = [0.0];
        let mut jfx_d = [0.0; NX];
        assert_eq!(0, jfx(&x, &u, &d, &w, &mut jfx_d));
    }

    #[test]
    fn tst_jfu() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let d = [-0.8, 6.2, 0.0];
        let w = [0.0];
        let mut jfu_d = [0.0; NX];
        assert_eq!(0, jfu(&x, &u, &d, &w, &mut jfu_d));
    }

    #[test]
    fn tst_ellx() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let w = [0.0];
        let mut ellx_res = [0.0; NX];
        assert_eq!(0, ellx(&x, &u, &w, &mut ellx_res));
    }

    #[test]
    fn tst_ellu() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let w = [0.0];
        let mut ellu_res = [0.0; NU];
        assert_eq!(0, ellu(&x, &u, &w, &mut ellu_res));
    }

    #[test]
    fn tst_vfx() {
        let x = [0.1, 0.2, 0.3];
        let u = [1.1, 2.2];
        let mut vfx_res = [0.0; NX];
        assert_eq!(0, vfx(&x, &mut vfx_res));
    }
}