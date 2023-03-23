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

    let seed: u64 = 34523459;

    let A;
    let B;
    let mut C;

    match args.input {
        Some(v) => {
            if v.len() != 2 {
                panic!("Expected 2 input files, got {}", v.len());
            }
            A = Matrix::from_random_seed(seed, args.size, args.size);
            B = Matrix::from_random_seed(seed + 10, args.size, args.size);
            // TODO: placeholder for the compiler, not yet implemented
        }
        None => {
            A = Matrix::from_random_seed(seed, args.size, args.size);
            // TODO: decide if this offset to the seed is ok
            B = Matrix::from_random_seed(seed + 10, args.size, args.size);
        }
    }

    // TODO: implement a zero method that returns a matrix of zeros
    C = Matrix::from_random_seed(seed, args.size, args.size);

    let t0 = Instant::now();

    A.multiply(&B, &mut C).unwrap();

    let t1 = Instant::now();

    if args.timing {
        println!("Elapsed: {:.2?}", t1 - t0);
    }

    if args.output {
        println!("Output:");
        println!("{}", C);
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
            verify(&A, &B, &C, args.size);
        }
        None => (),
    }
}

fn verify(A: &Matrix, B: &Matrix, C: &Matrix, size: usize) {
    let A_ref = obpmark_rust::matrix_2d::Matrix::new(A.get_data(), size, size);
    let B_ref = obpmark_rust::matrix_2d::Matrix::new(B.get_data(), size, size);

    let mut C_ref = obpmark_rust::matrix_2d::Matrix::from_random_seed(0, size, size);

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
