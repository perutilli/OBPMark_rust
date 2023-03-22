/***
 * Matrix multiplication benchmark
 * It multiplies two square matrices with side length `size`
 */

#![allow(non_snake_case)] // TODO: decide if we want to keep this or not
use clap::Parser;
use core::panic;
use obpmark_rust::{FromRandomSeed, MatMul};
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
    let size = args.size;

    let A;
    let B;
    let mut C;

    let seed: u64 = 34523459;

    match args.input {
        Some(_) => {
            panic!("Input files not supported yet");
        }
        None => {
            println!("No input files specified, generating random matrices");
            A = Matrix::from_random_seed(seed, size, size);
            // TODO: decide if this offset to the seed is ok
            B = Matrix::from_random_seed(seed + 10, size, size);
            // TODO: while not strictly necessary, we would want C to be initialized to 0
            C = Matrix::from_random_seed(seed, size, size);
        }
    }

    if size <= 10 {
        println!("{}", A);
        println!("{}", B);
    }

    let now = Instant::now();

    A.multiply(&B, &mut C).unwrap();

    println!("Elapsed: {:.2?}", now.elapsed());

    if args.output {
        println!("{}", C);
    }

    if args.verification {
        let A_ref = obpmark_rust::matrix_2d::Matrix::new(A.get_data(), size, size);
        let B_ref = obpmark_rust::matrix_2d::Matrix::new(B.get_data(), size, size);

        let mut C_ref = obpmark_rust::matrix_2d::Matrix::from_random_seed(seed, size, size);

        A_ref.multiply(&B_ref, &mut C_ref).unwrap();

        if C.get_data() == C_ref.get_data() {
            println!("Verification passed");
        } else {
            let C_data = C.get_data();
            let C_ref_data = C_ref.get_data();
            for i in 0..size {
                for j in 0..size {
                    if (C_data[i][j] - C_ref_data[i][j]) as f64 > 1e-4 {
                        println!("Verification failed");
                        println!("C: \n{:?}", C_data[i][j]);
                        println!("C_ref: \n{:?}", C_ref_data[i][j]);
                        return;
                    }
                }
            }
            println!("Verification passed using epsilon of 1e-4");
        }
    }
}
