#![allow(non_snake_case)] // TODO: decide if we want to keep this or not
use clap::Parser;
use core::panic;
use obpmark_rust::{BaseMatrix, Softmax};
use std::time::Instant;

use obpmark_rust::benchmark_utils::CommonArgs;

#[cfg(feature = "1d")]
use obpmark_rust::matrix_1d::Matrix;
#[cfg(feature = "2d")]
use obpmark_rust::matrix_2d::Matrix;
#[cfg(not(any(feature = "1d", feature = "2d", feature = "ndarray")))]
use obpmark_rust::matrix_2d::Matrix;
#[cfg(feature = "ndarray")]
use obpmark_rust::matrix_ndarray::Matrix; // once again for linting reasons

fn main() {
    let args = CommonArgs::parse();

    let seed: u64 = 34523459;

    let A;
    let mut B;

    match args.input {
        Some(v) => {
            if v.len() != 1 {
                panic!("Expected 1 input files, got {}", v.len());
            }
            unimplemented!("Reading input from file not yet implemented");
        }
        None => {
            A = Matrix::from_random_seed(seed, args.size, args.size);
        }
    }

    B = Matrix::zeroes(args.size, args.size);

    let t0 = Instant::now();

    A.softmax(&mut B).unwrap();

    let t1 = Instant::now();

    if args.timing {
        println!("Elapsed: {:.2?}", t1 - t0);
    }

    if args.output {
        println!("Output:");
        println!("{}", B);
    }

    match args.export {
        Some(filename) => {
            // export output
            unimplemented!("Export not yet implemented, filename: {}", filename);
        }
        None => (),
    }

    match args.verify {
        Some(Some(filename)) => {
            // verify against file
            unimplemented!(
                "Verification with file not yet implemented, filename: {}",
                filename
            );
        }
        Some(None) => {
            // verify against cpu implementation
            if verify(&A, &B, args.size) {
                println!("Verification successful");
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

    A_ref.softmax(&mut B_ref).unwrap();

    B.get_data() == B_ref.get_data()
}
