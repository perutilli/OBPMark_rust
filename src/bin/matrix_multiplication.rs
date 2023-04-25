/***
 * Matrix multiplication benchmark
 * It multiplies two square matrices with side length `size`
 */

#![allow(non_snake_case)]
use clap::Parser;
use core::panic;
use obpmark_rust::{BaseMatrix, MatMul, ParallelMatMul};
use std::{path::Path, time::Instant};

use obpmark_rust::benchmark_utils::{CommonArgs, Matrix, Number};
use obpmark_rust::matrix_2d::Matrix2d as RefMatrix;

use obpmark_rust::verify;

#[derive(Parser, Debug)]
#[command(about = "Matrix multiplication benchmark")]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,
}

fn main() {
    let args = Args::parse();

    let seed: u64 = 34523459;

    let A;
    let B;
    let mut C;

    match args.common.input {
        Some(v) => {
            if v.len() != 2 {
                panic!("Expected 2 input files, got {}", v.len());
            }
            A = Matrix::from_file(Path::new(&v[0]), args.common.size, args.common.size).unwrap();
            B = Matrix::from_file(Path::new(&v[1]), args.common.size, args.common.size).unwrap();
        }
        None => {
            A = Matrix::from_random_seed(
                seed,
                args.common.size,
                args.common.size,
                "-10".parse::<Number>().unwrap(),
                "10".parse::<Number>().unwrap(),
            );
            B = Matrix::from_random_seed(
                seed + 10,
                args.common.size,
                args.common.size,
                "-10".parse::<Number>().unwrap(),
                "10".parse::<Number>().unwrap(),
            );
        }
    }

    if args.common.print_input {
        println!("A:");
        println!("{}", A);
        println!("B:");
        println!("{}", B);
    }

    C = Matrix::zeroes(args.common.size, args.common.size);

    let t0 = Instant::now();

    match args.common.parallel {
        1 => A.multiply(&B, &mut C).unwrap(),
        n => A.parallel_multiply(&B, &mut C, n).unwrap(),
    }

    let t1 = Instant::now();

    if args.common.timing {
        println!("Elapsed: {:.2?}", t1 - t0);
    }

    if args.common.output {
        println!("Output:");
        println!("{}", C);
    }

    match args.common.export {
        Some(filename) => {
            // export output
            C.to_file(Path::new(&filename)).unwrap();
        }
        None => (),
    }

    match args.common.verify {
        Some(Some(filename)) => {
            // verify against file
            let C_ref = Matrix::from_file(Path::new(&filename), args.common.size, args.common.size)
                .unwrap();
            if C.get_data() == C_ref.get_data() {
                println!("Verification passed");
            } else {
                println!("Verification failed");
            }
        }
        Some(None) => {
            // verify against cpu implementation
            let C_ref = get_ref_result(&A, &B, args.common.size);
            verify!(C.get_data(), C_ref.get_data());
        }
        None => (),
    }
}

fn get_ref_result(A: &Matrix, B: &Matrix, size: usize) -> RefMatrix<Number> {
    let A_ref = RefMatrix::new(A.get_data(), size, size);
    let B_ref = RefMatrix::new(B.get_data(), size, size);

    let mut C_ref = RefMatrix::zeroes(size, size);

    A_ref.multiply(&B_ref, &mut C_ref).unwrap();

    C_ref
}
