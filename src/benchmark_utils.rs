use clap::Parser;

#[cfg(feature = "float")]
pub type Number = f32;
#[cfg(feature = "double")]
pub type Number = f64;
#[cfg(feature = "int")]
pub type Number = i32;
#[cfg(feature = "half")]
pub type Number = half::f16;
#[cfg(not(any(
    feature = "float",
    feature = "double",
    feature = "int",
    feature = "half"
)))]
pub type Number = f32;

#[cfg(feature = "1d")]
pub type Matrix = crate::matrix_1d::Matrix1d<Number>;
#[cfg(feature = "2d")]
pub type Matrix = crate::matrix_2d::Matrix2d<Number>;
#[cfg(not(any(feature = "1d", feature = "2d")))]
pub type Matrix = crate::matrix_2d::Matrix2d<Number>;

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

    /// Uses "mat_A.in" ["mat_B.in"] for input data, can take 1 or 2 files
    #[arg(short, long, num_args = 1..=2)]
    pub input: Option<Vec<String>>,

    /// Print the input matrix (or matrices) to stdout
    #[arg(long, default_value_t = false)]
    pub print_input: bool,

    /// Number of threads to use (default: 1)
    #[arg(short, long, default_value_t = 1)]
    pub parallel: usize,

    /// Random seed to use (default: 3894283)
    #[arg(long, default_value_t = 3894283)]
    pub seed: u64,
}

#[macro_export]
macro_rules! verify {
    ($res: expr, $ref_res: expr) => {
        if $res != $ref_res {
            println!("Verification failed");
        } else {
            println!("Verification passed");
        }
    };
}

#[macro_export]
macro_rules! number {
    ($e:expr) => {
        $e.parse::<Number>().unwrap()
    };
}
