#![allow(non_snake_case)]
use benchmarks::reference_implementations::softmax;
use clap::Parser;
use core::panic;
use obpmark_library::parallel_traits::ParallelSoftmax;
use obpmark_library::{rayon_traits::RayonSoftmax, BaseMatrix, Softmax};
use std::time::Instant;

use benchmarks::benchmark_utils::{verify_toll, CommonArgs, Implementation, Matrix, Number};
use benchmarks::number;
use obpmark_library::matrix_2d::Matrix2d as RefMatrix;

#[derive(Parser, Debug)]
#[command(about = "Softmax function benchmark")]
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
            unimplemented!("Reading input from file not yet implemented");
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
            A.rayon_softmax(&mut B).unwrap();
        }
        (Some(_), Implementation::Rayon) => {
            panic!("Cannot specify number of threads for Rayon implementation")
        }
        (Some(n), Implementation::Sequential) if n != 1 => {
            panic!("Invalid parameter combination: sequential with nthreads != 1")
        }
        (_, Implementation::Sequential) => A.softmax(&mut B).unwrap(),
        (Some(n), Implementation::StdParallel) => A.parallel_softmax(&mut B, n).unwrap(),
        (None, Implementation::StdParallel) => {
            // TODO: use number of cores
            A.parallel_softmax(&mut B, 8).unwrap();
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
            let B_ref = get_ref_result(A, args.common.size);
            verify_toll(&B.get_data(), &B_ref.get_data(), 1e-5);
        }
        None => (),
    }
}

fn get_ref_result(A: Matrix, size: usize) -> RefMatrix<Number> {
    let A_ref = A.to_c_format();

    let mut B_ref = vec![number!("0"); size * size];

    // TODO: this is for testing, remove
    let t = Instant::now();
    unsafe {
        softmax(A_ref.as_ptr(), B_ref.as_mut_ptr(), size);
    }
    println!("C code: {:.2?}", t.elapsed());

    let B_ref = B_ref.chunks(size).map(|c| c.to_vec()).collect();

    RefMatrix::new(B_ref, size, size)
}
