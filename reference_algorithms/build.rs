extern crate cc;

#[cfg(feature = "float")]
const C_TYPE: &str = "FLOAT";
#[cfg(feature = "double")]
const C_TYPE: &str = "DOUBLE";
#[cfg(feature = "int")]
const C_TYPE: &str = "INT";
#[cfg(feature = "half")]
compile_error!("Half precision is not supported yet");
#[cfg(not(any(
    feature = "float",
    feature = "double",
    feature = "int",
    feature = "half"
)))]
const C_TYPE: &str = "FLOAT";

fn main() {
    cc::Build::new()
        .file("src/cpu_functions.c")
        .define(C_TYPE, None)
        .flag("-O3")
        .warnings(false)
        .compile("reference_algorithms");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cpu_functions.c");
}
