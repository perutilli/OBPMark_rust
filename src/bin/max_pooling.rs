#![allow(non_snake_case)]
use obpmark_rust::{BaseMatrix, MaxPooling};
use std::time::Instant;

use clap::Parser;

use obpmark_rust::benchmark_utils::{verify, CommonArgs, Matrix, Number};
use obpmark_rust::matrix_2d::Matrix2d as RefMatrix;

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
            A = Matrix::from_random_seed(
                seed,
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
            let B_ref = get_ref_result(&A, args.common.size, args.stride, B_size);
            verify(&B.get_data(), &B_ref.get_data());
        }
        None => (),
    }
}

fn get_ref_result(A: &Matrix, size: usize, stride: usize, B_size: usize) -> RefMatrix<Number> {
    let A_ref = RefMatrix::new(A.get_data(), size, size);
    let mut B_ref = RefMatrix::zeroes(B_size, B_size);

    A_ref.max_pooling(&mut B_ref, stride, stride).unwrap();

    B_ref
}
