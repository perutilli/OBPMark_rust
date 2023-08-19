#![allow(non_snake_case)]
use obpmark_library::{
    parallel_traits::ParallelMaxPooling, rayon_traits::RayonMaxPooling, BaseMatrix, MaxPooling,
};
use reference_algorithms::max_pooling;
use std::time::Instant;

use clap::Parser;

use benchmarks::benchmark_utils::{CommonArgs, Implementation, Matrix, Number};
use benchmarks::{number, verify};
use obpmark_library::matrix_2d::Matrix2d as RefMatrix;

#[derive(Parser, Debug)]
#[command(about = "Max pooling benchmark")]
struct Args {
    /// Common arguments
    #[clap(flatten)]
    common: CommonArgs,

    /// Stride
    #[clap(long)]
    stride: usize,
}

fn main() {
    let args = Args::parse();

    if args.common.size % args.stride != 0 {
        panic!("Size must be a multiple of stride");
    }

    let B_size = args.common.size / args.stride;

    let A;
    let mut B = Matrix::zeroes(B_size, B_size);

    match args.common.input {
        Some(v) => {
            // read input from file
            if v.len() != 1 {
                panic!("Expected 1 input files, got {}", v.len());
            }
            // read the matrix/matrices
            unimplemented!("Reading input from file not yet implemented")
        }
        None => {
            // generate input
            A = Matrix::from_random_seed(
                args.common.seed,
                args.common.size,
                args.common.size,
                number!("-10"),
                number!("10"),
            );
        }
    }

    if args.common.print_input {
        println!("A:");
        println!("{}", A);
    }

    let t0 = Instant::now();

    match (args.common.nthreads, args.common.implementation) {
        (None, Implementation::Rayon) => {
            A.rayon_max_pooling(&mut B, args.stride, args.stride)
                .unwrap();
        }
        (Some(_), Implementation::Rayon) => {
            panic!("Cannot specify number of threads for Rayon implementation")
        }
        (Some(n), Implementation::Sequential) if n != 1 => {
            panic!("Invalid parameter combination: sequential with nthreads != 1")
        }
        (_, Implementation::Sequential) => A.max_pooling(&mut B, args.stride, args.stride).unwrap(),
        (Some(n), Implementation::StdParallel) => A
            .parallel_max_pooling(&mut B, args.stride, args.stride, n)
            .unwrap(),
        (None, Implementation::StdParallel) => {
            // TODO: use number of cores
            A.parallel_max_pooling(&mut B, args.stride, args.stride, 8)
                .unwrap();
        }
    }

    let t1 = Instant::now();

    if args.common.timing {
        // print timing
        println!("Elapsed: {:.2?}", t1 - t0);
    }

    if args.common.output {
        // print output
        println!("Output:");
        println!("{}", B);
    }

    match args.common.export {
        Some(filename) => {
            // export output
            unimplemented!("Export not yet implemented, filename: {}", filename);
        }
        None => (),
    }

    match args.common.verify {
        Some(Some(filename)) => {
            // verify against file
            unimplemented!("Verify not yet implemented, filename: {}", filename);
        }
        Some(None) => {
            // verify against cpu implementation
            let B_ref = get_ref_result(A, args.common.size, args.stride, B_size);
            verify!(B.get_data(), B_ref.get_data());
        }
        None => (),
    }
}

fn get_ref_result(A: Matrix, size: usize, stride: usize, B_size: usize) -> RefMatrix<Number> {
    let A_ref = A.to_c_format();
    let mut B_ref = vec![number!("0"); B_size * B_size];

    let t = Instant::now();
    unsafe {
        max_pooling(
            A_ref.as_ptr(),
            B_ref.as_mut_ptr(),
            size,
            stride,
            size / stride,
        )
    }
    println!("C code: {:.2?}", t.elapsed());
    let B_ref = B_ref.chunks(B_size).map(|c| c.to_vec()).collect();

    RefMatrix::new(B_ref, B_size, B_size)
}
