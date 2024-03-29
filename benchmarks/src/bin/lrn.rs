#![allow(non_snake_case)]
use clap::Parser;
use core::panic;
use obpmark_library::{parallel_traits::ParallelLRN, rayon_traits::RayonLRN, BaseMatrix, LRN};
use std::path::Path;
use std::time::Instant;

use benchmarks::benchmark_utils::{verify_toll, CommonArgs, Implementation, Matrix, Number};
use obpmark_library::matrix_2d::Matrix2d as RefMatrix;

use benchmarks::{number, verify};

use reference_algorithms::lrn;

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
                args.common.seed,
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

    match (args.common.nthreads, args.common.implementation) {
        (None, Implementation::Rayon) => {
            A.rayon_lrn(&mut B, ALPHA, BETA, K).unwrap();
        }
        (Some(_), Implementation::Rayon) => {
            panic!("Cannot specify number of threads for Rayon implementation")
        }
        (Some(n), Implementation::Sequential) if n != 1 => {
            panic!("Invalid parameter combination: sequential with nthreads != 1")
        }
        (_, Implementation::Sequential) => A.lrn(&mut B, ALPHA, BETA, K).unwrap(),
        (Some(n), Implementation::StdParallel) => {
            A.parallel_lrn(&mut B, ALPHA, BETA, K, n).unwrap()
        }
        (None, Implementation::StdParallel) => {
            // TODO: use number of cores
            A.parallel_lrn(&mut B, ALPHA, BETA, K, 8).unwrap()
        }
    }

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
            let B_ref = get_ref_result(A, args.common.size);
            verify_toll(&B.get_data(), &B_ref.get_data(), 1e-6);
        }
        None => (),
    }
}

fn get_ref_result(A: Matrix, size: usize) -> RefMatrix<Number> {
    let A_ref = A.to_c_format();
    let mut B_ref = vec![number!("0"); size * size];

    let t = Instant::now();
    unsafe {
        lrn(A_ref.as_ptr(), B_ref.as_mut_ptr(), size);
    }
    println!("C code: {:.2?}", t.elapsed());

    let B_ref = B_ref.chunks(size).map(|c| c.to_vec()).collect();

    RefMatrix::new(B_ref, size, size)
}
