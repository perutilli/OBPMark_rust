#![allow(non_snake_case)]
use clap::Parser;
use core::panic;
use obpmark_rust::{BaseMatrix, Relu};
use std::time::Instant;

use obpmark_rust::benchmark_utils::{CommonArgs, Matrix, Number};
use obpmark_rust::matrix_2d::Matrix2d as RefMatrix;

use obpmark_rust::verify;

#[derive(Parser, Debug)]
#[command(about = "Rectified Linear Unit benchmark")]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,
}

fn main() {
    let args = Args::parse();

    let A;

    match args.common.input {
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
            A = Matrix::from_random_seed(
                args.common.seed,
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
    }

    let mut B = Matrix::zeroes(args.common.size, args.common.size);

    let t0 = Instant::now();

    A.relu(&mut B).unwrap();

    let t1 = Instant::now();

    if args.common.timing {
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
            unimplemented!(
                "Exporting output not yet implemented, filename: {}",
                filename
            );
        }
        None => (),
    }

    match args.common.verify {
        Some(Some(filename)) => {
            // verify against file
            unimplemented!(
                "Verifying against file not yet implemented, filename: {}",
                filename
            );
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

    A_ref.relu(&mut B_ref).unwrap();

    B_ref
}
