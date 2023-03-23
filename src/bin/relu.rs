#![allow(non_snake_case)] // TODO: decide if we want to keep this or not
use clap::Parser;
use core::panic;
use obpmark_rust::{BaseMatrix, Relu};
use std::time::Instant;

use obpmark_rust::benchmark_utils::CommonArgs;

#[cfg(feature = "1d")]
use obpmark_rust::matrix_1d::Matrix;
#[cfg(feature = "2d")]
use obpmark_rust::matrix_2d::Matrix;
#[cfg(not(any(feature = "1d", feature = "2d", feature = "ndarray")))]
use obpmark_rust::matrix_2d::Matrix;
#[cfg(feature = "ndarray")] // TODO: ndarray not supported yet
use obpmark_rust::matrix_ndarray::Matrix; // once again for linting reasons

fn main() {
    let args = CommonArgs::parse();

    let seed = 38945;

    let A;

    match args.input {
        Some(v) => {
            // read input from file
            if v.len() != 1 {
                panic!("Expected 1 input file, got {}", v.len());
            }
            // read the matrix/matrices
            unimplemented!("Reading input from file not yet implemented");
        }
        None => {
            // generate input
            A = Matrix::from_random_seed(seed, args.size, args.size);
        }
    }

    let mut B = Matrix::zeroes(args.size, args.size);

    let t0 = Instant::now();

    A.relu(&mut B).unwrap();

    let t1 = Instant::now();

    if args.timing {
        println!("Elapsed: {:.2?}", t1 - t0);
    }

    if args.output {
        // print output
        println!("Output:");
        println!("{}", B);
    }

    match args.export {
        Some(filename) => {
            // export output
            unimplemented!(
                "Exporting output not yet implemented, filename: {}",
                filename
            );
        }
        None => (),
    }

    match args.verify {
        Some(Some(filename)) => {
            // verify against file
            unimplemented!(
                "Verifying against file not yet implemented, filename: {}",
                filename
            );
        }
        Some(None) => {
            // verify against cpu implementation
            if verify(&A, &B, args.size) {
                println!("Verification passed");
            } else {
                println!("Verification failed");
            }
        }
        None => (),
    }
}

fn verify(A: &Matrix, B: &Matrix, size: usize) -> bool {
    let A_ref = obpmark_rust::matrix_2d::Matrix::new(A.get_data(), size, size);
    let mut B_ref = obpmark_rust::matrix_2d::Matrix::zeroes(size, size);

    A_ref.relu(&mut B_ref).unwrap();

    return B_ref.get_data() == B.get_data();
}
