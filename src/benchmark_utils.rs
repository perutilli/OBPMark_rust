use clap::Parser;

use crate::BaseMatrix;

#[cfg(feature = "1d")]
pub type Matrix = crate::matrix_1d::Matrix;
#[cfg(feature = "2d")]
pub type Matrix = crate::matrix_2d::Matrix;
#[cfg(not(any(feature = "1d", feature = "2d", feature = "ndarray")))]
pub type Matrix = crate::matrix_2d::Matrix;
#[cfg(feature = "ndarray")]
pub type Matrix = crate::matrix_ndarray::Matrix; // once again for linting reasons

#[derive(Parser, Debug)]
pub struct CommonArgs {
    /// Size of the matrix (or matrices)
    #[arg(short, long)]
    pub size: usize,

    /// Export the result in hex format to file <export>
    #[arg(short, long)]
    pub export: Option<String>,

    /// Verifies the result against 2d reference implementation
    #[arg(short, long)]
    pub verify: Option<Option<String>>,

    /// Prints the output in human readable format to stdout
    #[arg(short, long, default_value_t = false)]
    pub output: bool,

    /// Prints the kernel execution time to stdout
    #[arg(short, long, default_value_t = false)]
    pub timing: bool,

    // TODO: maybe this should be benchmark specific to avoid confusion
    /// Uses "mat_A.in" ["mat_B.in"] for input data, can take 1 or 2 files
    #[arg(short, long, num_args = 1..=2)]
    pub input: Option<Vec<String>>,
}

pub fn verify(mat: &Matrix, mat_ref: &Matrix) {
    // TODO: potentially use epsilon for comparison
    if mat.get_data() != mat_ref.get_data() {
        println!("Verification failed");
    } else {
        println!("Verification passed");
    }
}
