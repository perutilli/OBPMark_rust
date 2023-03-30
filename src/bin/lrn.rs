#![allow(non_snake_case)]
use clap::Parser;
use core::panic;
use obpmark_rust::{BaseMatrix, LRN};
use std::path::Path;
use std::time::Instant;

use obpmark_rust::benchmark_utils::{CommonArgs, Matrix, Number};
use obpmark_rust::matrix_2d::Matrix2d as RefMatrix;

use obpmark_rust::{number, verify};

const ALPHA: Number = 10e-4;
const BETA: Number = 0.75;
const K: Number = 2.0;

#[derive(Parser, Debug)]
#[command(about = "LRN benchmark")]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,
}

fn main() {
    let args = Args::parse();

    let seed: u64 = 34523459;

    let A;
    let mut B;

    match args.common.input {
        Some(v) => {
            if v.len() != 1 {
                panic!("Expected 1 input files, got {}", v.len());
            }
            A = Matrix::from_file(Path::new(&v[0]), args.common.size, args.common.size).unwrap();
        }
        None => {
            A = Matrix::from_random_seed(
                seed,
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

    B = Matrix::zeroes(args.common.size, args.common.size);

    let t0 = Instant::now();

    A.lrn(&mut B, ALPHA, BETA, K).unwrap();

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
            let B_ref = Matrix::from_file(Path::new(&filename), args.common.size, args.common.size)
                .unwrap();
            verify!(B.get_data(), B_ref.get_data());
        }
        Some(None) => {
            // verify against cpu implementation
            let B_ref = get_ref_result(&A, args.common.size);
            verify!(B.get_data(), B_ref.get_data());
        }
        None => (),
    }
}

fn get_ref_result(A: &Matrix, size: usize) -> RefMatrix<Number> {
    let A_ref = RefMatrix::new(A.get_data(), size, size);
    let mut B_ref = RefMatrix::zeroes(size, size);

    A_ref.lrn(&mut B_ref, ALPHA, BETA, K).unwrap();

    B_ref
}
