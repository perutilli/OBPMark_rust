#![allow(non_snake_case)] // TODO: decide if we want to keep this or not
use clap::Parser;
use core::panic;
use obpmark_rust::{BaseMatrix, Softmax};
use std::time::Instant;

use obpmark_rust::benchmark_utils::{verify, CommonArgs, Matrix};
use obpmark_rust::matrix_2d::Matrix as RefMatrix;

#[derive(Parser, Debug)]
#[command(about = "Softmax function benchmark")]
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
            unimplemented!("Reading input from file not yet implemented");
        }
        None => {
            A = Matrix::from_random_seed(seed, args.common.size, args.common.size);
        }
    }

    B = Matrix::zeroes(args.common.size, args.common.size);

    let t0 = Instant::now();

    A.softmax(&mut B).unwrap();

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
            unimplemented!("Export not yet implemented, filename: {}", filename);
        }
        None => (),
    }

    match args.common.verify {
        Some(Some(filename)) => {
            // verify against file
            unimplemented!(
                "Verification with file not yet implemented, filename: {}",
                filename
            );
        }
        Some(None) => {
            // verify against cpu implementation
            let B_ref = get_ref_result(&A, args.common.size);
            verify(&B, &B_ref);
        }
        None => (),
    }
}

fn get_ref_result(A: &Matrix, size: usize) -> RefMatrix {
    let A_ref = RefMatrix::new(A.get_data(), size, size);

    let mut B_ref = RefMatrix::zeroes(size, size);

    A_ref.softmax(&mut B_ref).unwrap();

    B_ref
}
