#![allow(non_snake_case)]
use std::time::Instant;

use obpmark_rust::benchmark_utils::CommonArgs;
use obpmark_rust::{BaseMatrix, MaxPooling};

use clap::Parser;

#[cfg(feature = "1d")]
use obpmark_rust::matrix_1d::Matrix;
#[cfg(feature = "2d")]
use obpmark_rust::matrix_2d::Matrix;
#[cfg(not(any(feature = "1d", feature = "2d")))]
use obpmark_rust::matrix_2d::Matrix;

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

    let seed = 38945;

    if args.common.size % args.stride != 0 {
        // TODO: check if this is generally true in common max pooling implementations
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
            A = Matrix::from_random_seed(seed, args.common.size, args.common.size);
        }
    }

    let t0 = Instant::now();

    A.max_pooling(&mut B, args.stride, args.stride).unwrap();

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
            if verify(&A, &B, args.common.size, args.stride, B_size) {
                println!("Verification successful");
            } else {
                println!("Verification failed");
            }
        }
        None => (),
    }
}

fn verify(A: &Matrix, B: &Matrix, size: usize, stride: usize, B_size: usize) -> bool {
    let A_ref = obpmark_rust::matrix_2d::Matrix::new(A.get_data(), size, size);
    let mut B_ref = obpmark_rust::matrix_2d::Matrix::zeroes(B_size, B_size);

    A_ref.max_pooling(&mut B_ref, stride, stride).unwrap();

    return B_ref.get_data() == B.get_data();
}
