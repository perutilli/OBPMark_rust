/***
 * This is a special case of the convolution benchmark, where we have 1d matrix and kernel.
 */

#![allow(non_snake_case)]
use clap::Parser;
use core::panic;
use obpmark_library::{rayon_traits::RayonConvolution, BaseMatrix, Convolution, Padding};
use std::path::Path;
use std::time::Instant;

use benchmarks::benchmark_utils::{CommonArgs, Implementation, Matrix, Number};
use obpmark_library::matrix_2d::Matrix2d as RefMatrix;

use benchmarks::{number, verify};

#[derive(Parser, Debug)]
#[command(about = "Finite impulse response filter benchmark")]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,

    /// Kernel size
    #[clap(short, long)]
    kernel_size: usize,
}

fn main() {
    let args = Args::parse();

    let A;
    let kernel;
    let mut B;

    match args.common.input {
        Some(v) => {
            if v.len() != 2 {
                panic!("Expected 2 input files, got {}", v.len());
            }
            A = Matrix::from_file(Path::new(&v[0]), 1, args.common.size).unwrap();
            kernel = Matrix::from_file(Path::new(&v[1]), 1, args.kernel_size).unwrap();
        }
        None => {
            A = Matrix::from_random_seed(
                args.common.seed,
                1,
                args.common.size,
                number!("-10"),
                number!("10"),
            );
            kernel = Matrix::from_random_seed(
                args.common.seed + 10,
                1,
                args.kernel_size,
                number!("-10"),
                number!("10"),
            );
        }
    }

    if args.common.print_input {
        println!("A:");
        println!("{}", A);
        println!("Kernel:");
        println!("{}", kernel);
    }

    B = Matrix::zeroes(1, args.common.size);

    let t0 = Instant::now();

    match (args.common.nthreads, args.common.implementation) {
        (None, Implementation::Rayon) => {
            A.rayon_convolute(&kernel, Padding::Zeroes, &mut B).unwrap();
        }
        (Some(_), Implementation::Rayon) => {
            panic!("Cannot specify number of threads for Rayon implementation")
        }
        (Some(n), Implementation::Sequential) if n != 1 => {
            panic!("Invalid parameter combination: sequential with nthreads != 1")
        }
        (_, Implementation::Sequential) => A.convolute(&kernel, Padding::Zeroes, &mut B).unwrap(),
        (_n, Implementation::StdParallel) => unimplemented!("Naive parallel not yet implemented"),
    }

    let t1 = Instant::now();

    if args.common.timing {
        println!("Elapsed: {:.2?}", t1 - t0);
    }

    if args.common.output {
        println!("Output:");
        println!("{}", B);
    }

    match args.common.export {
        Some(filename) => {
            // export output
            B.to_file(Path::new(&filename)).unwrap();
        }
        None => (),
    }

    match args.common.verify {
        Some(Some(filename)) => {
            // verify against file
            let B_ref = Matrix::from_file(Path::new(&filename), 1, args.common.size).unwrap();
            verify!(B.get_data(), B_ref.get_data());
        }
        Some(None) => {
            // verify against cpu implementation
            let B_ref = get_ref_result(&A, args.common.size, &kernel, args.kernel_size);
            verify!(B.get_data(), B_ref.get_data());
        }
        None => (),
    }
}

fn get_ref_result(
    A: &Matrix,
    size: usize,
    kernel: &Matrix,
    kernel_size: usize,
) -> RefMatrix<Number> {
    let A_ref = RefMatrix::new(A.get_data(), 1, size);
    let kernel_ref = RefMatrix::new(kernel.get_data(), 1, kernel_size);

    let mut B_ref = RefMatrix::zeroes(1, size);

    A_ref
        .convolute(&kernel_ref, Padding::Zeroes, &mut B_ref)
        .unwrap();

    B_ref
}
