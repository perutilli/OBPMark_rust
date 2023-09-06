extern crate cc;

#[cfg(feature = "float")]
const C_TYPE: &str = "FLOAT";
#[cfg(feature = "double")]
const C_TYPE: &str = "DOUBLE";
#[cfg(feature = "int")]
const C_TYPE: &str = "INT";
#[cfg(not(any(
    feature = "float",
    feature = "double",
    feature = "int",
)))]
const C_TYPE: &str = "FLOAT";

// #[cfg(all(feature = "std", feature = "riscv_hard_float"))]
// compile_error!("Specifying floating point ABI is not supported with standard library enabled");

fn main() {
    let mut base_config = cc::Build::new();
    let compiler_config = if cfg!(feature = "std") {
        base_config
            .file("src/cpu_functions.c")
            .define(C_TYPE, None)
            .opt_level(3)
            .warnings(false)
            .define("STD", None)
    } else if cfg!(feature = "riscv_hard_float") {
        base_config
            .file("src/cpu_functions.c")
            .define(C_TYPE, None)
            .opt_level(3)
            .warnings(false)
            .flag("-mabi=lp64d")
    } else {
        unreachable!("No valid configuration found")
    };
    compiler_config.compile("reference_algorithms");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cpu_functions.c");
}
