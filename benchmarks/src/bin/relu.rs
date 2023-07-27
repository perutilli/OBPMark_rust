#![allow(non_snake_case)]
use clap::Parser;
use core::panic;
use obpmark_library::{parallel_traits::ParallelRelu, rayon_traits::RayonRelu, BaseMatrix, Relu};
use std::time::Instant;

use benchmarks::benchmark_utils::{CommonArgs, Implementation, Matrix, Number};
use benchmarks::{number, verify};

use benchmarks::reference_implementations::relu;
use obpmark_library::matrix_2d::Matrix2d as RefMatrix;

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
                number!("-10"),
                number!("10"),
            );
        }
    }

    if args.common.print_input {
        println!("A:");
        println!("{}", A);
    }

    let mut B = Matrix::zeroes(args.common.size, args.common.size);

    let t0 = Instant::now();

    match (args.common.nthreads, args.common.implementation) {
        (Some(n), Implementation::Sequential) if n != 1 => {
            panic!("Cannot run sequential implementation with more than 1 thread");
        }
        (_, Implementation::Sequential) => {
            A.relu(&mut B).unwrap();
        }
        (None, Implementation::Rayon) => {
            A.rayon_relu(&mut B).unwrap();
        }
        (Some(_), Implementation::Rayon) => {
            panic!("Cannot specify number of threads for Rayon implementation");
        }
        (Some(n), Implementation::StdParallel) => {
            A.parallel_relu(&mut B, n).unwrap();
        }
        (None, Implementation::StdParallel) => {
            // TODO: change n_threads
            A.parallel_relu(&mut B, 8).unwrap();
        }
    }

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
            let B_ref = get_ref_result(A, args.common.size);
            verify!(B.get_data(), B_ref.get_data());
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
        relu(A_ref.as_ptr(), B_ref.as_mut_ptr(), size);
    }
    println!("C code: {:.2?}", t.elapsed());

    let B_ref = B_ref.chunks(size).map(|c| c.to_vec()).collect();

    RefMatrix::new(B_ref, size, size)
}
