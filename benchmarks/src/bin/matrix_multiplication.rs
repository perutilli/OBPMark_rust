/***
 * Matrix multiplication benchmark
 * It multiplies two square matrices with side length `size`
 */

#![allow(non_snake_case)]
use clap::Parser;
use core::panic;
use obpmark_library::{
    parallel_traits::ParallelMatMul, rayon_traits::RayonMatMul, BaseMatrix, MatMul,
};
use std::{path::Path, time::Instant};

use benchmarks::benchmark_utils::{CommonArgs, Implementation, Matrix, Number};
use benchmarks::{number, verify};

use benchmarks::reference_implementations::matrix_multiplication;
use obpmark_library::matrix_1d::Matrix1d as RefMatrix;
#[derive(Parser, Debug)]
#[command(about = "Matrix multiplication benchmark")]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,
}

fn main() {
    let args = Args::parse();

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

    C = Matrix::zeroes(args.common.size, args.common.size);

    let t0 = Instant::now();

    match (args.common.nthreads, args.common.implementation) {
        (None, Implementation::Rayon) => {
            A.rayon_multiply(&B, &mut C).unwrap();
        }
        (Some(_), Implementation::Rayon) => {
            panic!("Cannot specify number of threads for Rayon implementation")
        }
        (Some(n), Implementation::Sequential) if n != 1 => {
            panic!("Invalid parameter combination: sequential with nthreads != 1")
        }
        (_, Implementation::Sequential) => A.multiply(&B, &mut C).unwrap(),
        (Some(n_threads), Implementation::StdParallel) => {
            A.parallel_multiply(&B, &mut C, n_threads).unwrap()
        }
        (None, Implementation::StdParallel) => {
            // TODO: change 8 to number of cores
            A.parallel_multiply(&B, &mut C, 8).unwrap()
        }
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
            let C_ref = get_ref_result(A, B, args.common.size);
            verify!(C.get_data(), C_ref.get_data());
        }
        None => (),
    }
}

fn get_ref_result(A: Matrix, B: Matrix, size: usize) -> RefMatrix<Number> {
    let A_ref = A.to_c_format();
    let B_ref = B.to_c_format();

    let mut C_ref = vec![number!("0"); size * size];

    // TODO: this is for testing, remove
    let t = Instant::now();
    unsafe {
        matrix_multiplication(
            A_ref.as_ptr(),
            B_ref.as_ptr(),
            C_ref.as_mut_ptr(),
            size,
            size,
            size,
        );
    }
    println!("C code: {:.2?}", t.elapsed());
    let C_ref = C_ref.chunks(size).map(|c| c.to_vec()).collect();

    RefMatrix::new(C_ref, size, size)
}
