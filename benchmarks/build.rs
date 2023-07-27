extern crate cc;

#[cfg(feature = "float")]
const C_TYPE: &str = "FLOAT";
#[cfg(feature = "double")]
const C_TYPE: &str = "DOUBLE";
#[cfg(feature = "int")]
const C_TYPE: &str = "INT";
#[cfg(not(any(feature = "float", feature = "double", feature = "int")))]
const C_TYPE: &str = "FLOAT";

fn main() {
    cc::Build::new()
        .file("reference_versions/cpu_functions.c")
        .define(C_TYPE, None)
        .warnings(false)
        .compile("reference_functions");

    println!("cargo:rerun-if-changed=build.rs");
}
