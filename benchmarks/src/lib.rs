pub mod benchmark_utils {
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
    pub type Matrix = obpmark_library::matrix_1d::Matrix1d<Number>;
    #[cfg(feature = "2d")]
    pub type Matrix = obpmark_library::matrix_2d::Matrix2d<Number>;
    #[cfg(not(any(feature = "1d", feature = "2d")))]
    pub type Matrix = obpmark_library::matrix_2d::Matrix2d<Number>;

    #[derive(clap::ValueEnum, Clone, Debug)]
    pub enum Implementation {
        Sequential,
        StdParallel,
        Rayon,
    }

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

        /// Number of threads to use
        #[arg(short, long)]
        pub nthreads: Option<usize>,

        /// Random seed to use (default: 3894283)
        #[arg(long, default_value_t = 3894283)]
        pub seed: u64,

        /// Parallel implementation to use (default: Naive)
        #[arg(value_enum, long, default_value_t = Implementation::Sequential)]
        pub implementation: Implementation,
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
}

pub mod reference_implementations {
    #[allow(unused_imports)]
    use std::ffi::{c_double, c_float, c_int};

    #[cfg(feature = "float")]
    type CType = c_float;
    #[cfg(feature = "double")]
    type CType = c_double;
    #[cfg(feature = "int")]
    type CType = c_int;
    #[cfg(feature = "half")]
    compile_error!("Half precision validation not supported yet");
    #[cfg(not(any(
        feature = "float",
        feature = "double",
        feature = "int",
        feature = "half"
    )))]
    type CType = f32;

    extern "C" {
        // void matrix_multiplication(const bench_t *A, const bench_t *B, bench_t *C, const unsigned int n, const unsigned int m, const unsigned int w)
        pub fn matrix_multiplication(
            a: *const CType,
            b: *const CType,
            c: *mut CType,
            n: usize,
            m: usize,
            k: usize,
        );

        // void relu(const bench_t *A, bench_t *B, const unsigned int size)void relu(const bench_t *A, bench_t *B, const unsigned int size)
        pub fn relu(a: *const CType, b: *mut CType, size: usize);

        // void softmax(const bench_t *A, bench_t *B, const unsigned int size)
        pub fn softmax(a: *const CType, b: *mut CType, size: usize);

        // void matrix_convolution(const bench_t *A, bench_t *kernel, bench_t *B, const int size, const int kernel_size)
        pub fn matrix_convolution(
            a: *const CType,
            kernel: *const CType,
            b: *mut CType,
            size: usize,
            kernel_size: usize,
        );

        // need to fix my code before I can use this
        // void correlation_2D(const bench_t *A, const bench_t *B, result_bench_t *R, const int size)
        // pub fn correlation_2d(a: *const CType, b: *const CType, r: *mut CType, size: usize);
    }
}
