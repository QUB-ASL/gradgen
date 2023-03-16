use std::path::Path;

fn main() {
    assert!(
        Path::new("extern/casadi_functions.c").exists(),
        "extern/casadi_functions.c is missing"
    );
    assert!(
        Path::new("extern/interface.c").exists(),
        "extern/interface.c is missing"
    );

    cc::Build::new()
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wno-unused-variable")
        .flag_if_supported("-Wno-long-long")
        .flag_if_supported("-Wno-unused-parameter")
        .pic(true)
        .include("src")
        .file("extern/casadi_functions.c")
        .file("extern/interface.c")
        .compile("icasadi");

    // Rerun if these autogenerated files change
    println!("cargo:rerun-if-changed=extern/casadi_functions.c");
    println!("cargo:rerun-if-changed=extern/interface.c");
}