use composed_kernel::{
    composed_kernel_composed_demo_f, composed_kernel_composed_demo_f_meta,
    composed_kernel_composed_demo_grad_x, composed_kernel_composed_demo_grad_x_meta,
    FunctionMetadata,
};

fn print_metadata(label: &str, metadata: FunctionMetadata) {
    println!("{label}: {metadata:#?}");
}

fn main() {
    print_metadata(
        "composed_demo_f metadata",
        composed_kernel_composed_demo_f_meta(),
    );
    print_metadata(
        "composed_demo_grad_x metadata",
        composed_kernel_composed_demo_grad_x_meta(),
    );

    let x = [1.0_f64, 2.0_f64];
    let parameters = [
        3.0_f64, 4.0_f64, 5.0_f64, 6.0_f64, 7.0_f64, 8.0_f64, 9.0_f64, 10.0_f64, 11.0_f64,
        12.0_f64, 13.0_f64,
    ];

    let mut y = [0.0_f64; 1];
    let mut work = vec![0.0_f64; composed_kernel_composed_demo_f_meta().workspace_size];
    composed_kernel_composed_demo_f(&x, &parameters, &mut y, &mut work);
    println!("f(x, parameters) = {:?}", y);

    let mut grad = [0.0_f64; 2];
    let mut grad_work = vec![0.0_f64; composed_kernel_composed_demo_grad_x_meta().workspace_size];
    composed_kernel_composed_demo_grad_x(&x, &parameters, &mut grad, &mut grad_work);
    println!("grad f(x, parameters) = {:?}", grad);
}
