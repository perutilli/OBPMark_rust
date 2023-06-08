#![allow(non_snake_case)]
use clap::Parser;
use core::panic;
use obpmark_library::{BaseMatrix, Correlation};
use std::{path::Path, time::Instant};

use benchmarks::benchmark_utils::{CommonArgs, Matrix, Number};
use obpmark_library::matrix_2d::Matrix2d as RefMatrix;

use benchmarks::{number, verify};

#[derive(Parser, Debug)]
#[command(about = "2D correlation benchmark")]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,
}

fn main() {
    let args = Args::parse();

    let A;
    let B;

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
                args.common.seed,
                args.common.size,
                args.common.size,
                number!("-10"),
                number!("10"),
            );
            B = Matrix::from_random_seed(
                args.common.seed + 10,
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
        println!("B:");
        println!("{}", B);
    }

    let t0 = Instant::now();

    let res = A.correlation(&B).unwrap();

    let t1 = Instant::now();

    if args.common.timing {
        println!("Elapsed: {:.2?}", t1 - t0);
    }

    if args.common.output {
        println!("Output:");
        println!("Correlation = {}", res);
    }

    match args.common.export {
        Some(filename) => {
            // export output
            // TODO: this is a very hacky way to do this, make it better
            obpmark_library::matrix_1d::Matrix1d::<f64>::new(vec![vec![res; 1]; 1], 1, 1)
                .to_file(Path::new(&filename))
                .unwrap();
        }
        None => (),
    }

    match args.common.verify {
        Some(Some(_filename)) => {
            /* verify against file TODO: how do we deal with this?
            let C_ref = Matrix::from_file(Path::new(&filename), args.common.size, args.common.size)
                .unwrap();
            if C.get_data() == C_ref.get_data() {
                println!("Verification passed");
            } else {
                println!("Verification failed");
            }
             */
        }
        Some(None) => {
            // verify against cpu implementation
            let res_ref = get_ref_result(&A, &B, args.common.size);
            // verify(&C.get_data(), &C_ref.get_data());
            verify!(&res, &res_ref);
        }
        None => (),
    }
}

fn get_ref_result(A: &Matrix, B: &Matrix, size: usize) -> f64 {
    let A_ref = RefMatrix::new(A.get_data(), size, size);
    let B_ref = RefMatrix::new(B.get_data(), size, size);

    let res = A_ref.correlation(&B_ref).unwrap();

    res
}
