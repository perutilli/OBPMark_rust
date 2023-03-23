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
    let size = args.size;

    let A;
    let mut B;

    let seed: u64 = 9453458;

    match args.input {
        Some(_) => {
            panic!("Input files not supported yet");
        }
        None => {
            println!("No input files specified, generating random matrices");
            A = Matrix::from_random_seed(seed, size, size);
            B = Matrix::zeroes(size, size);
        }
    }

    let now = Instant::now();

    A.relu(&mut B).unwrap();

    println!("Elapsed: {:.2?}", now.elapsed());

    if args.output {
        println!("{}", B);
    }
}
